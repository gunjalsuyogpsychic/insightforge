import os
from dataclasses import dataclass

@dataclass
class Settings:
    # Choose ONE provider:
    # 1) Groq LLM (cloud): set GROQ_API_KEY
    # 2) Ollama (local): install Ollama + pull a model (e.g., llama3)

    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")  # "groq" or "ollama"
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    DATA_SALES_CSV: str = os.getenv("DATA_SALES_CSV", "data/sales_data.csv")
    DATA_RECORDS_XLSX: str = os.getenv("DATA_RECORDS_XLSX", "data/records.xlsx")

    STORAGE_DIR: str = os.getenv("STORAGE_DIR", "storage")
    FAISS_DIR: str = os.getenv("FAISS_DIR", "storage/faiss_index")
    MEMORY_FILE: str = os.getenv("MEMORY_FILE", "storage/chat_memory.json")

settings = Settings()
