from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_core.output_parsers import PydanticOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pydantic import BaseModel, Field

from sql_functions import (
    get_table_names,
    describe_table,
    execute_query
)

load_dotenv()

DATABASE = "Student_db"
PERSIST_DIRECTORY = "embeddings"
COLLECTION_NAME = "sql_embeddings"

model = ChatOpenAI(
    base_url="http://localhost:11434/v1/",
    model="hf.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:Q4_K_M",
    temperature=0,
    streaming=False
)

model_embed = OllamaEmbeddings(
    model="qwen3-embedding:0.6b"
)

class SQLResponse(BaseModel):
    sql: str = Field(
        description=(
            "A MariaDB SELECT query, or '-1' if the "
            "question is irrelevant to the database schema."
        )
    )

    irrelevant: bool = Field(
        description=(
            "True if the question cannot be answered "
            "using the database schema; otherwise False."
        )
    )


parser = PydanticOutputParser(
    pydantic_object=SQLResponse
)

prompt1 = PromptTemplate(
    template="""
You are an expert MariaDB SQL query generator.

DATABASE SCHEMA:
{schema}

USER QUESTION:
{queary}

RULES:
1. Generate a valid MariaDB SELECT query.
2. Use only the tables and columns provided in the schema.
3. The table names are explicitly provided in the schema.
4. Use the correct table names and column names.
5. Use JOINs when the question requires multiple tables.
6. If the question is unrelated to the schema or cannot
   be answered using it, return sql="-1" and irrelevant=true.
7. Otherwise, return the SQL query and irrelevant=false.
8. Generate only read-only SELECT queries.
9. Never generate INSERT, UPDATE, DELETE, DROP, ALTER,
   CREATE, TRUNCATE, or other modifying statements.
10. Do not invent table names or columns.

Return the response in the required JSON format.

{format_instructions}
""",
    input_variables=["schema", "queary"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

chroma_db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=model_embed,
    collection_name=COLLECTION_NAME
)

tables = get_table_names(DATABASE)

table_data = {}

for table in tables:
    table_data[table] = describe_table(DATABASE, table)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

existing = chroma_db.get(limit=1)

if not existing["ids"]:

    for table, data in table_data.items():

        # The table name is part of the embedded document text.
        table_schema = (
            f"Table Name: {table}\n"
            f"Schema:\n{data}"
        )

        chunks = splitter.split_text(str(table_schema))

        for index, chunk in enumerate(chunks):

            chroma_db.add_texts(
                texts=[chunk],
                metadatas=[{"table": table}],
                ids=[f"{table}_{index}"]
            )

    print("Database schemas embedded successfully.")

else:
    print(
        "Existing Chroma collection found. "
        "If your schemas changed, rebuild the collection."
    )


retriever = chroma_db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 6,
        "fetch_k": 20,
        "lambda_mult": 0.5
    }
)


chain = prompt1 | model | parser


def is_safe_select(sql: str) -> bool:

    sql = sql.strip()

    if not sql or sql == "-1":
        return False

    # Reject multiple SQL statements.
    if ";" in sql.rstrip(";"):
        return False

    normalized = sql.upper()

    # Permit SELECT and WITH ... SELECT (CTEs).
    if not (
        normalized.startswith("SELECT ")
        or normalized.startswith("WITH ")
    ):
        return False

    forbidden = [
        "INSERT ",
        "UPDATE ",
        "DELETE ",
        "DROP ",
        "ALTER ",
        "CREATE ",
        "TRUNCATE ",
        "REPLACE ",
        "GRANT ",
        "REVOKE ",
        "INTO OUTFILE",
        "INTO DUMPFILE"
    ]

    padded_sql = " " + normalized + " "

    for keyword in forbidden:
        if keyword in padded_sql:
            return False

    return True


print("Natural Language to SQL chatbot")
print("Type /exit to quit.")

while True:

    inp = input("\nEnter your query: ").strip()

    if inp.lower() == "/exit":
        break

    if not inp:
        continue

    try:

        # Retrieve relevant schema documents.
        docs = retriever.invoke(inp)

        if not docs:
            print("No relevant schema found.")
            continue

        # Table names are already embedded in page_content.
        schema = "\n\n".join(
            doc.page_content for doc in docs
        )

        print("\nRetrieved schema:")
        print(schema)

        # Generate validated structured output.
        response = chain.invoke({
            "schema": schema,
            "queary": inp
        })

        print("\nStructured response:")
        print(response.model_dump_json(indent=2))

        if response.irrelevant or response.sql.strip() == "-1":
            print("\nQuestion is irrelevant to the schema.")
            continue

        if not is_safe_select(response.sql):
            print("\nRejected: SQL failed basic safety checks.")
            continue

        print("\nGenerated SQL:")
        print(response.sql)

        # Execute the query.
        result = execute_query(
            DATABASE,
            response.sql
        )

        print("\nQuery result:")
        print(result)

    except Exception as e:
        print(f"\nError: {e}")


print("\nChatbot terminated.")
