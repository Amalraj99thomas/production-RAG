import os
from dotenv import load_dotenv

load_dotenv()


DATA_DIR = os.getenv("RAG_DATA_DIR", "rag_store")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
INDEX_DIR = os.path.join(DATA_DIR, "indexes")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # 1536-d
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in env variables")