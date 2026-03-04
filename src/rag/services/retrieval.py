from typing import Dict, List
import faiss

from rag.services.embeddings import embed_texts

def retrieve(index: faiss.IndexFlatIP, query: str, chunks: List[str], k: int = 4) -> List[Dict]:
    q = embed_texts([query])
    if q.shape[0] == 0:
        return []

    faiss.normalize_L2(q)
    D, I = index.search(q, k)

    out = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        out.append({"score": float(score), "text": chunks[idx], "chunk_id": int(idx)})
    return out

def build_prompt(question: str, contexts: List[Dict]) -> str:
    context_block = "\n\n---\n\n".join(
        [f"[Chunk {c['chunk_id']}] {c['text']}" for c in contexts]
    )

    instructions = (
        "You are a precise document question-answering assistant.\n"
        "Answer ONLY using the provided context.\n\n"
        "Rules:\n"
        "1. Every factual statement must include a citation in the format (Chunk X).\n"
        "2. You may cite multiple chunks like (Chunk 2, Chunk 5).\n"
        "3. Do NOT use outside knowledge.\n"
        "4. If the answer is not explicitly supported by the context, say:\n"
        "   'I do not find this information in the document.'\n"
        "5. Do NOT invent citations.\n"
    )

    return (
        f"{instructions}\n\n"
        f"Context:\n{context_block}\n\n"
        f"User question: {question}\n\n"
        f"Answer:"
    )