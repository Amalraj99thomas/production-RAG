import os
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from rag.core.config import UPLOAD_DIR, EMBED_MODEL
from rag.models.schemas import AskIn
from rag.services.pdf import read_pdf_bytes
from rag.services.chunking import chunk_text
from rag.services.embeddings import embed_texts
from rag.storage.faiss_store import build_faiss_index, save_index, load_index
from rag.services.retrieval import retrieve, build_prompt
from rag.services.generation import generate_answer

router = APIRouter()

@router.get("/")
def health():
    return {"status": "running"}

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(..., description="A PDF file")):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    doc_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    with open(save_path, "wb") as f:
        f.write(pdf_bytes)

    text = read_pdf_bytes(pdf_bytes)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No usable text chunks from PDF.")

    embeddings = embed_texts(chunks)
    if embeddings.shape[0] != len(chunks):
        raise HTTPException(status_code=500, detail="Embedding failure.")

    index = build_faiss_index(embeddings)
    meta = {
        "doc_id": doc_id,
        "filename": file.filename,
        "num_chunks": len(chunks),
        "embedding_model": EMBED_MODEL,
    }

    save_index(doc_id, index, chunks, meta)
    return JSONResponse({"message": "Indexed PDF successfully.", "doc_id": doc_id, "meta": meta})

@router.post("/ask")
def ask(payload: AskIn):
    try:
        index, chunks, meta = load_index(payload.doc_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="doc_id not found. Upload first.")

    hits = retrieve(index, payload.question, chunks, k=4)
    if not hits:
        return JSONResponse({"answer": "I couldn't find this in the document.", "sources": []})

    prompt = build_prompt(payload.question, hits)
    answer = generate_answer(prompt)

    sources = [{"chunk_id": h["chunk_id"], "score": h["score"]} for h in hits]
    return JSONResponse({"answer": answer, "sources": sources, "meta": meta})