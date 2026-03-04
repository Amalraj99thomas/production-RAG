import os
import json
import pickle
from typing import Dict, List, Tuple

import faiss
import numpy as np

from rag.core.config import INDEX_DIR

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def save_index(doc_id: str, index: faiss.IndexFlatIP, chunks: List[str], meta: Dict) -> None:
    doc_dir = os.path.join(INDEX_DIR, doc_id)
    os.makedirs(doc_dir, exist_ok=True)

    faiss.write_index(index, os.path.join(doc_dir, "index.faiss"))

    with open(os.path.join(doc_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    with open(os.path.join(doc_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_index(doc_id: str) -> Tuple[faiss.IndexFlatIP, List[str], Dict]:
    doc_dir = os.path.join(INDEX_DIR, doc_id)
    if not os.path.isdir(doc_dir):
        raise FileNotFoundError("Unknown doc_id.")

    index = faiss.read_index(os.path.join(doc_dir, "index.faiss"))

    with open(os.path.join(doc_dir, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)

    with open(os.path.join(doc_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    return index, chunks, meta