import os
from typing import List

import mysql.connector
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Read environment variables
host = os.getenv("DB_HOST", "localhost")
user = os.getenv("DB_USER", "root")
password = os.getenv("DB_PASSWORD", "")


def get_table_names(database: str) -> List[str]:
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database,
    )

    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = %s;
        """,
        (database,),
    )

    tables = [row[0] for row in cursor.fetchall()]

    cursor.close()
    conn.close()

    return tables

def describe_table(database: str, table_name: str) -> str:
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database,
    )

    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_KEY, COLUMN_DEFAULT, EXTRA
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
        ORDER BY ORDINAL_POSITION;
        """,
        (database, table_name),
    )

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    if not rows:
        return f"Table {table_name} not found in database {database}."

    lines = []
    for column_name, column_type, is_nullable, column_key, column_default, extra in rows:
        default_value = "NULL" if column_default is None else str(column_default)
        lines.append(
            f"{column_name}: {column_type}, nullable={is_nullable}, key={column_key}, default={default_value}, extra={extra}"
        )

    return "\n".join(lines)

def execute_query(database: str, query: str) -> str:
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database,
    )

    cursor = conn.cursor()
    try:
        cursor.execute(query)
        
        # Check if query returns results (SELECT queries)
        if cursor.description:
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            lines = [", ".join(columns)]
            for row in rows:
                lines.append(", ".join(str(val) for val in row))
            
            result = "\n".join(lines)
        else:
            # For INSERT, UPDATE, DELETE queries
            conn.commit()
            result = f"Query executed successfully. Rows affected: {cursor.rowcount}"
    except Exception as e:
        result = f"Error executing query: {str(e)}"
    finally:
        cursor.close()
        conn.close()

    return result

# print(execute_query("Test_db","SELECT * FROM Customers;"))

