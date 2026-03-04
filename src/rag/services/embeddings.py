from typing import List
import numpy as np
from openai import OpenAI
from rag.core.config import OPENAI_API_KEY, EMBED_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Get embeddings for a list of strings. Returns (n, d) float32 array.
    """
    if not texts:
        return np.zeros((0, 1536), dtype="float32")

    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype="float32")