from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request
from pathlib import Path

from src.search import (
    load_chunks,
    extract_texts,
    build_bm25,
    build_dense_index,
    rank_bm25,
    rank_dense,
    reciprocal_rank_fusion,
)

CHUNKS_PATH = Path(
    "data/processed/chunks/Sripetch_vs_SEC_chunks.json"
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    chunks = load_chunks(CHUNKS_PATH)
    texts = extract_texts(chunks)

    bm25 = build_bm25(texts)

    embedding_model, chunk_embeddings = build_dense_index(
        texts,
        cache_dir=EMBEDDING_CACHE_DIR,
    )

    app.state.chunks = chunks
    app.state.texts = texts
    app.state.bm25 = bm25
    app.state.embedding_model = embedding_model
    app.state.chunk_embeddings = chunk_embeddings

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

    bm25_ranked_indices = rank_bm25(
        search_request.query,
        state.bm25,
    )

    dense_ranked_indices = rank_dense(
        search_request.query,
        state.embedding_model,
        state.chunk_embeddings,
    )

    fused_ranked_indices, rrf_scores = reciprocal_rank_fusion(
        bm25_ranked_indices,
        dense_ranked_indices,
    )

    results = []

    for rank, chunk_index in enumerate(
        fused_ranked_indices[:search_request.top_k],
        start=1,
    ):
        result = dict(state.chunks[chunk_index])

        result["rank"] = rank
        result["rrf_score"] = rrf_scores[chunk_index]

        results.append(result)

    return {
        "query": search_request.query,
        "top_k": search_request.top_k,
        "results": results,
    }