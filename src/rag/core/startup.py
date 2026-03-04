import os
from rag.core.config import UPLOAD_DIR, INDEX_DIR

def ensure_storage_dirs() -> None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)