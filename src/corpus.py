from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import Settings, SourceSettings
from src.search import (
    build_bm25,
    build_dense_index,
    extract_texts,
    hybrid_search,
    load_chunks,
    rank_bm25,
    rank_dense,
    reciprocal_rank_fusion,
)

from typing import Literal

@dataclass(frozen=True)
class CorpusSource:
    source: SourceSettings
    chunks_path: Path


@dataclass
class SourceIndex:
    chunks: list[dict]
    texts: list[str]
    bm25: object
    chunk_embeddings: np.ndarray


@dataclass
class CorpusIndex:
    chunks: list[dict]
    texts: list[str]
    bm25: object
    embedding_model: SentenceTransformer
    chunk_embeddings: np.ndarray
    sources: dict[str, SourceIndex]


def discover_corpus_sources(
    settings: Settings,
) -> list[CorpusSource]:
    corpus_sources = []

    for source in settings.sources.values():
        chunk_variant = source.chunk_variant

        if chunk_variant == "plain":
            if not source.chunks_path.exists():
                print(
                    f"Skipping '{source.key}': "
                    "plain chunks are required but "
                    "were not found"
                )
                continue

            chunks_path = source.chunks_path

        elif chunk_variant == "prefer_speaker":
            if source.speaker_chunks_path.exists():
                chunks_path = (
                    source.speaker_chunks_path
                )

            elif source.chunks_path.exists():
                print(
                    f"WARNING: '{source.key}' has no "
                    "speaker chunks; falling back to "
                    "plain chunks"
                )

                chunks_path = source.chunks_path

            else:
                print(
                    f"Skipping '{source.key}': "
                    "no speaker or plain chunks found"
                )
                continue

        elif chunk_variant == "require_speaker":
            if not source.speaker_chunks_path.exists():
                print(
                    f"Skipping '{source.key}': "
                    "speaker chunks are required but "
                    "were not found"
                )
                continue

            chunks_path = (
                source.speaker_chunks_path
            )

        else:
            print(
                f"Skipping '{source.key}': "
                f"unknown chunk variant "
                f"'{chunk_variant}'"
            )
            continue

        corpus_sources.append(
            CorpusSource(
                source=source,
                chunks_path=chunks_path,
            )
        )

    return corpus_sources


def build_corpus_index(
    settings: Settings,
) -> CorpusIndex | None:
    corpus_sources = discover_corpus_sources(
        settings
    )

    if not corpus_sources:
        print("No processed corpus sources found.")
        return None

    embedding_model = SentenceTransformer(
        settings.models.embedding_model
    )

    all_chunks = []
    source_embedding_matrices = []
    source_indexes = {}

    for corpus_source in corpus_sources:
        source = corpus_source.source

        source_chunks = load_chunks(
            corpus_source.chunks_path
        )

        normalized_chunks = []

        for chunk in source_chunks:
            normalized_chunk = dict(chunk)

            normalized_chunk["source_key"] = (
                source.key
            )

            normalized_chunk["source_id"] = (
                source.source_id
            )

            normalized_chunks.append(
                normalized_chunk
            )

        source_texts = extract_texts(
            normalized_chunks
        )

        source_bm25 = build_bm25(
            source_texts
        )

        _, source_embeddings = build_dense_index(
            source_texts,
            cache_dir=source.embedding_cache_dir,
            embedding_model=embedding_model,
            model_name=settings.models.embedding_model,
        )

        source_indexes[source.key] = SourceIndex(
            chunks=normalized_chunks,
            texts=source_texts,
            bm25=source_bm25,
            chunk_embeddings=source_embeddings,
        )

        all_chunks.extend(
            normalized_chunks
        )

        source_embedding_matrices.append(
            source_embeddings
        )

        print(
            f"Loaded {len(normalized_chunks)} "
            f"chunks from '{source.key}'"
        )

    all_texts = extract_texts(all_chunks)

    bm25 = build_bm25(all_texts)

    chunk_embeddings = np.concatenate(
        source_embedding_matrices,
        axis=0,
    )

    return CorpusIndex(
        chunks=all_chunks,
        texts=all_texts,
        bm25=bm25,
        embedding_model=embedding_model,
        chunk_embeddings=chunk_embeddings,
        sources=source_indexes,
    )

RetrievalMode = Literal[
    "global",
    "per_source",
]


def search_source(
    query: str,
    index: CorpusIndex,
    source_key: str,
    top_k: int,
) -> list[dict]:
    source_index = index.sources[source_key]

    return hybrid_search(
        query=query,
        chunks=source_index.chunks,
        bm25=source_index.bm25,
        embedding_model=index.embedding_model,
        chunk_embeddings=(
            source_index.chunk_embeddings
        ),
        top_k=top_k,
    )

def search_filtered_global(
    query: str,
    index: CorpusIndex,
    source_keys: list[str],
    top_k: int,
) -> list[dict]:
    selected_source_keys = set(source_keys)

    bm25_ranked_indices = rank_bm25(
        query,
        index.bm25,
    )

    dense_ranked_indices = rank_dense(
        query,
        index.embedding_model,
        index.chunk_embeddings,
    )

    filtered_bm25_indices = [
        int(chunk_index)
        for chunk_index in bm25_ranked_indices
        if index.chunks[int(chunk_index)][
            "source_key"
        ] in selected_source_keys
    ]

    filtered_dense_indices = [
        int(chunk_index)
        for chunk_index in dense_ranked_indices
        if index.chunks[int(chunk_index)][
            "source_key"
        ] in selected_source_keys
    ]

    fused_ranked_indices, rrf_scores = (
        reciprocal_rank_fusion(
            filtered_bm25_indices,
            filtered_dense_indices,
        )
    )

    results = []

    for rank, chunk_index in enumerate(
        fused_ranked_indices[:top_k],
        start=1,
    ):
        result = index.chunks[
            chunk_index
        ].copy()

        result["rank"] = rank
        result["rrf_score"] = (
            rrf_scores[chunk_index]
        )

        results.append(result)

    return results

def search_corpus(
    query: str,
    index: CorpusIndex,
    top_k: int = 5,
    source_keys: list[str] | None = None,
    retrieval_mode: RetrievalMode = "global",
    top_k_per_source: int = 3,
) -> list[dict]:
    available_source_keys = list(
        index.sources
    )

    if source_keys:
        selected_source_keys = [
            source_key
            for source_key in source_keys
            if source_key in index.sources
        ]
    else:
        selected_source_keys = (
            available_source_keys
        )

    if not selected_source_keys:
        return []

    if retrieval_mode == "per_source":
        results = []

        for source_key in selected_source_keys:
            source_results = search_source(
                query=query,
                index=index,
                source_key=source_key,
                top_k=top_k_per_source,
            )

            for result in source_results:
                result["source_rank"] = (
                    result["rank"]
                )

                result["rank"] = (
                    len(results) + 1
                )

                results.append(result)

        return results

    if (
        len(selected_source_keys)
        == len(available_source_keys)
    ):
        return hybrid_search(
            query=query,
            chunks=index.chunks,
            bm25=index.bm25,
            embedding_model=(
                index.embedding_model
            ),
            chunk_embeddings=(
                index.chunk_embeddings
            ),
            top_k=top_k,
        )

    if len(selected_source_keys) == 1:
        return search_source(
            query=query,
            index=index,
            source_key=selected_source_keys[0],
            top_k=top_k,
        )

    return search_filtered_global(
        query=query,
        index=index,
        source_keys=selected_source_keys,
        top_k=top_k,
    )