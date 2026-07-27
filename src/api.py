from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request
from pathlib import Path

from src.search import (
    load_chunks,
    extract_texts,
    build_bm25,
    build_dense_index,
    hybrid_search,
)

from src.llm_clients import LlamaCppClient
from src.rag import GroundedAnswer, answer_question

CHUNKS_PATH = Path(
    "data/processed/chunks/Sripetch_vs_SEC_chunks_20260726_145257.json"
)

EMBEDDING_CACHE_DIR = Path(
    "data/cache"
)

class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )


class AnswerRequest(BaseModel):
    query: str
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    chunks = load_chunks(CHUNKS_PATH)
    texts = extract_texts(chunks)

    bm25 = build_bm25(texts)

    embedding_model, chunk_embeddings = build_dense_index(
        texts,
        cache_dir=EMBEDDING_CACHE_DIR,
    )

    llm_client = LlamaCppClient()

    app.state.chunks = chunks
    app.state.texts = texts
    app.state.bm25 = bm25
    app.state.embedding_model = embedding_model
    app.state.chunk_embeddings = chunk_embeddings
    app.state.llm_client = llm_client

    print(f"Loaded {len(chunks)} chunks")

    yield


app = FastAPI(
    title="Audio Search API",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "chunks_loaded": len(request.app.state.chunks),
    }

@app.post("/search")
def search(
    search_request: SearchRequest,
    request: Request,
):
    state = request.app.state

    results = hybrid_search(
        query=search_request.query,
        chunks=state.chunks,
        bm25=state.bm25,
        embedding_model=state.embedding_model,
        chunk_embeddings=state.chunk_embeddings,
        top_k=search_request.top_k,
    )

    return {
        "query": search_request.query,
        "top_k": search_request.top_k,
        "results": results,
    }

@app.post(
    "/answer",
    response_model=GroundedAnswer,
)
def answer(
    answer_request: AnswerRequest,
    request: Request,
):
    state = request.app.state

    retrieved_chunks = hybrid_search(
        query=answer_request.query,
        chunks=state.chunks,
        bm25=state.bm25,
        embedding_model=state.embedding_model,
        chunk_embeddings=state.chunk_embeddings,
        top_k=answer_request.top_k,
    )

    return answer_question(
        query=answer_request.query,
        retrieved_chunks=retrieved_chunks,
        llm_client=state.llm_client,
    )