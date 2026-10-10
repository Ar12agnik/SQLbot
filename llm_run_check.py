from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
load_dotenv()
model = ChatOpenAI(
    base_url="http://localhost:11434/v1/",
    model="hf.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:Q4_K_M",
    streaming=True
)
model_embed = OllamaEmbeddings(
    model = "qwen3-embedding:0.6b"
)
print(model_embed.embed_query("Write a C program to display the number of digits in a number"))