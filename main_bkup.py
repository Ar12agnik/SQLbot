from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings  
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel 
from langchain_chroma import Chroma
from sql_functions import get_table_names,describe_table,execute_query
from langchain_text_splitters import RecursiveCharacterTextSplitter


model = ChatOpenAI(
    base_url="http://localhost:11434/v1/",
    model="hf.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:Q4_K_M",
    streaming=True
)
model_embed = OllamaEmbeddings(
    model = "qwen3-embedding:0.6b"
)
prompt1 = PromptTemplate(template = "You are a Sql expert given this schema:{schema} and the queary {queary} write a sql queary(marina db) to get the relivant result.write only the sql querynothing else not even (```sql ```). if the queary is not relivant to schema please return -1",input_variables = ["schema","queary"])

class output(BaseModel):
    sql:str 
    irrelevant:bool

chroma_db = Chroma(persist_directory = "embeddings",embedding_function = model_embed,collection_name = "sql_embeddings")
##PreProssing
tables= get_table_names("Student_db")
table_data = {}
for table in tables:
    data=describe_table("Student_db",table)
    table_data[table]=data
# print(table_data)
###text Splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
for table, data in table_data.items():
    chunks = splitter.split_text(data)
    for chunk in chunks:
        chroma_db.add_texts([chunk],metadatas=[{"table":table}])
# print(chroma_db.get(include=['embeddings','documents', 'metadatas']))
retriever = chroma_db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.4
    }
)
inp = ""
while inp!="/exit":
    inp = input("enter your Queary and /exit to exit: ")
    if inp=="/exit":
        break
    docs = retriever.invoke(inp)
    prompt1.invoke({'schema': docs, 'queary': inp})
    model_response = model.invoke(prompt1.invoke({'schema': docs, 'queary': inp}))
    execute_response = execute_query("Student_db", str(model_response.content))
    print(execute_response)