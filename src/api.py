from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from src.config import load_settings
from src.llm_clients import LlamaCppClient
from src.rag import GroundedAnswer, answer_question
from collections import Counter
from typing import Literal

from src.corpus import (
    build_corpus_index,
    search_corpus,
)


class SearchRequest(BaseModel):
    query: str

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    source_keys: list[str] | None = None

    retrieval_mode: Literal[
        "global",
        "per_source",
    ] = "global"

    top_k_per_source: int = Field(
        default=3,
        ge=1,
        le=10,
    )


class AnswerRequest(BaseModel):
    query: str

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    source_keys: list[str] | None = None

    retrieval_mode: Literal[
        "global",
        "per_source",
    ] = "global"

    top_k_per_source: int = Field(
        default=3,
        ge=1,
        le=10,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()

    corpus_index = build_corpus_index(
        settings
    )

    if corpus_index is None:
        raise RuntimeError(
            "No processed corpus sources found."
        )

    llm_client = LlamaCppClient(
        base_url=settings.llm.base_url,
        model=settings.llm.model,
        timeout=settings.llm.timeout_seconds,
    )

    app.state.settings = settings
    app.state.corpus_index = corpus_index
    app.state.llm_client = llm_client

    source_counts = Counter(
        chunk["source_key"]
        for chunk in corpus_index.chunks
    )

    print(
        f"Loaded {len(corpus_index.chunks)} "
        f"chunks from {len(source_counts)} sources"
    )

    yield


app = FastAPI(
    title="Audio Search API",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    state = request.app.state
    index = state.corpus_index

    source_counts = Counter(
        chunk["source_key"]
        for chunk in index.chunks
    )

    return {
        "status": "ok",
        "sources_loaded": len(source_counts),
        "chunks_loaded": len(index.chunks),
        "chunks_by_source": dict(source_counts),
        "embedding_model": (
            state.settings.models.embedding_model
        ),
    }

@app.post("/search")
def search(
    search_request: SearchRequest,
    request: Request,
):
    index = request.app.state.corpus_index

    results = search_corpus(
        query=search_request.query,
        index=index,
        top_k=search_request.top_k,
        source_keys=search_request.source_keys,
        retrieval_mode=(
            search_request.retrieval_mode
        ),
        top_k_per_source=(
            search_request.top_k_per_source
        ),
    )

    return {
        "query": search_request.query,
        "source_keys": search_request.source_keys,
        "retrieval_mode": (
            search_request.retrieval_mode
        ),
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

    retrieved_chunks = search_corpus(
        query=answer_request.query,
        index=state.corpus_index,
        top_k=answer_request.top_k,
        source_keys=answer_request.source_keys,
        retrieval_mode=(
            answer_request.retrieval_mode
        ),
        top_k_per_source=(
            answer_request.top_k_per_source
        ),
    )

    return answer_question(
        query=answer_request.query,
        retrieved_chunks=retrieved_chunks,
        llm_client=state.llm_client,
    )