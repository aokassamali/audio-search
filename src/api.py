from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.audio_ingest import AudioIngestManager
from src.chunk import create_chunks
from src.config import PROJECT_ROOT, load_settings
from src.corpus import SourceIndex, build_corpus_index
from src.llm_clients import LlamaCppClient
from src.rag import GroundedAnswer, answer_question
from src.retrieval_planner import retrieve_with_plan
from src.search import build_bm25, build_dense_index, extract_texts, load_chunks
from src.transcript_import import parse_transcript_bytes

FRONTEND_DIR = PROJECT_ROOT / "frontend"
IMPORT_ROOT = PROJECT_ROOT / "data" / "demo_imports"
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    source_keys: list[str] | None = None
    retrieval_mode: Literal["global", "per_source"] = "global"
    top_k_per_source: int = Field(default=3, ge=1, le=10)


class AnswerRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    source_keys: list[str] | None = None
    retrieval_mode: Literal["global", "per_source"] = "global"
    top_k_per_source: int = Field(default=3, ge=1, le=10)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    return slug or "imported-source"


def _unique_import_source_key(state, display_name: str) -> str:
    base = _slugify(display_name)
    key = base
    suffix = 2
    while (
        key in state.corpus_index.sources
        or key in state.settings.sources
        or key in state.runtime_sources
        or (IMPORT_ROOT / key).exists()
    ):
        key = f"{base}-{suffix}"
        suffix += 1
    return key


def _source_audio_path(state, source_key: str) -> Path | None:
    runtime = state.runtime_sources.get(source_key)
    if runtime and runtime.get("audio_path"):
        path = Path(runtime["audio_path"])
        if path.exists():
            return path
    configured = state.settings.sources.get(source_key)
    if configured is None:
        return None
    if configured.normalized_audio_path.exists():
        return configured.normalized_audio_path
    if configured.raw_audio_path.exists():
        return configured.raw_audio_path
    return None


def _source_display_name(state, source_key: str) -> str:
    runtime = state.runtime_sources.get(source_key)
    if runtime:
        return runtime.get("display_name", source_key)
    configured = state.settings.sources.get(source_key)
    if configured:
        return Path(configured.audio_filename).stem.replace("_", " ")
    return source_key.replace("_", " ")


def _source_summary(state, source_key: str) -> dict:
    source_index = state.corpus_index.sources[source_key]
    chunks = source_index.chunks
    timed = [chunk for chunk in chunks if chunk.get("start", -1) >= 0 and chunk.get("end", -1) >= 0]
    speakers = []
    for chunk in chunks:
        for label in chunk.get("speaker_labels", {}).values():
            if label not in speakers:
                speakers.append(label)
    runtime = state.runtime_sources.get(source_key, {})
    return {
        "source_key": source_key,
        "source_id": chunks[0]["source_id"] if chunks else source_key,
        "display_name": _source_display_name(state, source_key),
        "chunk_count": len(chunks),
        "has_audio": _source_audio_path(state, source_key) is not None,
        "has_timestamps": len(timed) == len(chunks) and bool(chunks),
        "speakers": speakers[:12],
        "source_type": runtime.get("source_type", "processed_audio"),
        "status": "ready",
    }


def _source_catalog(state) -> list[dict]:
    return [
        {"source_key": key, "display_name": _source_display_name(state, key)}
        for key in state.corpus_index.sources
    ]


def _rebuild_global_index(index) -> None:
    source_indexes = list(index.sources.values())
    index.chunks = [chunk for source_index in source_indexes for chunk in source_index.chunks]
    index.texts = extract_texts(index.chunks)
    index.bm25 = build_bm25(index.texts)
    index.chunk_embeddings = np.concatenate(
        [source_index.chunk_embeddings for source_index in source_indexes],
        axis=0,
    )


def _add_runtime_source(
    state,
    source_key: str,
    source_id: str,
    chunks: list[dict],
    cache_dir: Path | None = None,
) -> None:
    normalized_chunks = []
    for chunk in chunks:
        normalized = dict(chunk)
        normalized["source_key"] = source_key
        normalized["source_id"] = source_id
        normalized_chunks.append(normalized)
    texts = extract_texts(normalized_chunks)
    if cache_dir is None:
        cache_dir = state.settings.paths.embedding_cache_root / "imports" / source_key
    _, embeddings = build_dense_index(
        texts,
        cache_dir=cache_dir,
        embedding_model=state.corpus_index.embedding_model,
        model_name=state.settings.models.embedding_model,
    )
    state.corpus_index.sources[source_key] = SourceIndex(
        chunks=normalized_chunks,
        texts=texts,
        bm25=build_bm25(texts),
        chunk_embeddings=embeddings,
    )
    _rebuild_global_index(state.corpus_index)


def _attach_completed_audio_job(state, job: dict) -> None:
    if job.get("status") != "complete":
        return
    source_key = job["source_key"]
    if source_key in state.corpus_index.sources:
        return
    source = state.settings.sources.get(source_key)
    if source is None or not source.active_chunks_path.exists():
        return
    chunks = load_chunks(source.active_chunks_path)
    _add_runtime_source(
        state,
        source_key=source_key,
        source_id=source.source_id,
        chunks=chunks,
        cache_dir=source.embedding_cache_dir,
    )
    playback_path = (
        source.normalized_audio_path
        if source.normalized_audio_path.exists()
        else Path(job["audio_path"])
    )
    state.runtime_sources[source_key] = {
        "display_name": job["display_name"],
        "source_type": "processed_audio",
        # Chunk timestamps are derived from normalized PCM. Serving that same
        # file avoids the small seek offsets that can happen with MP3 playback.
        "audio_path": str(playback_path),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    corpus_index = build_corpus_index(settings)
    if corpus_index is None:
        raise RuntimeError("No processed corpus sources found.")
    llm_client = LlamaCppClient(
        base_url=settings.llm.base_url,
        model=settings.llm.model,
        timeout=settings.llm.timeout_seconds,
    )
    app.state.settings = settings
    app.state.corpus_index = corpus_index
    app.state.llm_client = llm_client
    app.state.runtime_sources = {}
    app.state.audio_ingest = AudioIngestManager(settings)
    source_counts = Counter(chunk["source_key"] for chunk in corpus_index.chunks)
    print(f"Loaded {len(corpus_index.chunks)} chunks from {len(source_counts)} sources")
    print(
        "Audio ingestion workers: "
        f"{app.state.audio_ingest.max_workers}; "
        f"Dagster home: {app.state.audio_ingest.dagster_home}"
    )
    try:
        yield
    finally:
        app.state.audio_ingest.shutdown()


app = FastAPI(title="Audio Search API", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    state = request.app.state
    index = state.corpus_index
    source_counts = Counter(chunk["source_key"] for chunk in index.chunks)
    return {
        "status": "ok",
        "sources_loaded": len(source_counts),
        "chunks_loaded": len(index.chunks),
        "chunks_by_source": dict(source_counts),
        "embedding_model": state.settings.models.embedding_model,
        "ingest_workers": state.audio_ingest.max_workers,
        "dagster_home": str(state.audio_ingest.dagster_home),
    }


@app.get("/sources")
def sources(request: Request):
    state = request.app.state
    return {"sources": [_source_summary(state, key) for key in state.corpus_index.sources]}


@app.get("/sources/{source_key}/chunks")
def source_chunks(
    source_key: str,
    request: Request,
    query: str | None = None,
    chunk_id: int | None = None,
    limit: int = 500,
):
    state = request.app.state
    source_index = state.corpus_index.sources.get(source_key)
    if source_index is None:
        raise HTTPException(status_code=404, detail="Unknown source")
    chunks = source_index.chunks
    if chunk_id is not None:
        chunks = [chunk for chunk in chunks if chunk.get("chunk_id") == chunk_id]
    if query:
        needle = query.lower().strip()
        chunks = [
            chunk
            for chunk in chunks
            if needle in chunk.get("text", "").lower()
            or needle in chunk.get("speaker_text", "").lower()
            or any(needle in str(label).lower() for label in chunk.get("speaker_labels", {}).values())
        ]
    return {
        "source": _source_summary(state, source_key),
        "total": len(chunks),
        "chunks": chunks[: max(1, min(limit, 2000))],
    }


@app.get("/sources/{source_key}/audio")
def source_audio(source_key: str, request: Request):
    state = request.app.state
    if source_key not in state.corpus_index.sources:
        raise HTTPException(status_code=404, detail="Unknown source")
    audio_path = _source_audio_path(state, source_key)
    if audio_path is None:
        raise HTTPException(status_code=404, detail="No audio is attached to this source")
    return FileResponse(audio_path)


@app.post("/search")
def search(search_request: SearchRequest, request: Request):
    state = request.app.state
    results, plan = retrieve_with_plan(
        query=search_request.query,
        index=state.corpus_index,
        selected_source_keys=search_request.source_keys,
        source_catalog=_source_catalog(state),
        llm_client=state.llm_client,
        top_k=search_request.top_k,
        top_k_per_source=search_request.top_k_per_source,
    )
    return {
        "query": search_request.query,
        "source_keys": search_request.source_keys,
        "retrieval_mode": "llm_planned_hybrid",
        "retrieval_plan": plan.model_dump(),
        "results": results,
    }


@app.post("/answer", response_model=GroundedAnswer)
def answer(answer_request: AnswerRequest, request: Request):
    state = request.app.state
    retrieved_chunks, plan = retrieve_with_plan(
        query=answer_request.query,
        index=state.corpus_index,
        selected_source_keys=answer_request.source_keys,
        source_catalog=_source_catalog(state),
        llm_client=state.llm_client,
        top_k=answer_request.top_k,
        top_k_per_source=answer_request.top_k_per_source,
    )
    if plan.needs_clarification:
        clarification = plan.clarification_question.strip()
        message = (
            "I don't understand the question well enough to answer it reliably. "
            + (clarification or "Could you clarify or rewrite it more clearly?")
        )
        return GroundedAnswer(answerable=False, answer=message, citations=[])
    return answer_question(
        query=answer_request.query,
        retrieved_chunks=retrieved_chunks,
        llm_client=state.llm_client,
    )


@app.post("/ingest/audio")
async def ingest_audio(
    request: Request,
    audio: UploadFile = File(...),
    source_name: str | None = Form(default=None),
):
    extension = Path(audio.filename or "audio").suffix.lower()
    if extension not in AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    display_name = (
        source_name.strip()
        if source_name and source_name.strip()
        else Path(audio.filename or "Audio source").stem
    )
    raw_dir = request.app.state.settings.paths.raw_audio_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / f"{_slugify(display_name)}-{uuid.uuid4().hex[:8]}{extension}"
    target.write_bytes(await audio.read())
    return request.app.state.audio_ingest.submit(audio_path=target, display_name=display_name)


@app.get("/ingest/jobs")
def ingest_jobs(request: Request):
    state = request.app.state
    jobs = state.audio_ingest.list_jobs()
    for job in jobs:
        _attach_completed_audio_job(state, job)
    return {"jobs": jobs}


@app.get("/ingest/jobs/{job_id}")
def ingest_job(job_id: str, request: Request):
    state = request.app.state
    try:
        job = state.audio_ingest.status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown ingestion job") from exc
    _attach_completed_audio_job(state, job)
    return job


@app.post("/ingest/jobs/{job_id}/cancel")
def cancel_ingest_job(job_id: str, request: Request):
    try:
        return request.app.state.audio_ingest.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown ingestion job") from exc


@app.post("/ingest/cancel-all")
def cancel_all_ingest(request: Request):
    return {
        "status": "cancellation_requested",
        "jobs": request.app.state.audio_ingest.cancel_all(),
    }


@app.post("/ingest/transcript")
async def ingest_transcript(
    request: Request,
    transcript: UploadFile = File(...),
    source_name: str | None = Form(default=None),
    audio: UploadFile | None = File(default=None),
):
    state = request.app.state
    payload = await transcript.read()
    try:
        segments, metadata = parse_transcript_bytes(transcript.filename or "transcript.txt", payload)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    display_name = (
        source_name.strip()
        if source_name and source_name.strip()
        else Path(transcript.filename or "Imported transcript").stem
    )
    source_key = _unique_import_source_key(state, display_name)
    source_id = source_key

    speaker_labels = {
        segment["speaker"]: segment["speaker"]
        for segment in segments
        if segment.get("speaker")
    }
    chunks = create_chunks(
        segments,
        source_id=source_id,
        speaker_labels=speaker_labels,
    )

    source_dir = IMPORT_ROOT / source_key
    source_dir.mkdir(parents=True, exist_ok=False)
    transcript_path = source_dir / "transcript.json"
    chunks_path = source_dir / "chunks.json"
    transcript_path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")
    chunks_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    audio_path = None
    if audio is not None and audio.filename:
        extension = Path(audio.filename).suffix.lower()
        if extension not in AUDIO_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Unsupported audio format")
        audio_path = source_dir / f"audio{extension}"
        audio_path.write_bytes(await audio.read())

    _add_runtime_source(state, source_key=source_key, source_id=source_id, chunks=chunks)
    state.runtime_sources[source_key] = {
        "display_name": display_name,
        "source_type": "imported_transcript",
        "audio_path": str(audio_path) if audio_path else None,
        "transcript_path": str(transcript_path),
        "chunks_path": str(chunks_path),
        **metadata,
    }
    return {
        "status": "ready",
        "source": _source_summary(state, source_key),
        "import": metadata,
    }


@app.get("/", include_in_schema=False)
def frontend():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not built")
    return FileResponse(index_path)


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
