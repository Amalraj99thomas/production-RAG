from fastapi import FastAPI
from rag.api.routes import router
from rag.core.startup import ensure_storage_dirs

def create_app() -> FastAPI:
    app = FastAPI(title="PDF RAG API", version="1.0")
    ensure_storage_dirs()
    app.include_router(router)
    return app

app = create_app()