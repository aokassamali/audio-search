from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import Settings, SourceSettings
from src.search import (
    build_bm25,
    build_dense_index,
    extract_texts,
    load_chunks,
)

from typing import Literal

from src.search import hybrid_search

@dataclass(frozen=True)
class CorpusSource:
    source: SourceSettings
    chunks_path: Path


@dataclass
class CorpusIndex:
    chunks: list[dict]
    texts: list[str]
    bm25: object
    embedding_model: SentenceTransformer
    chunk_embeddings: np.ndarray


def discover_corpus_sources(
    settings: Settings,
) -> list[CorpusSource]:
    corpus_sources = []

    for source in settings.sources.values():
        if (
            source.chunk_variant == "speaker"
            and source.speaker_chunks_path.exists()
        ):
            chunks_path = source.speaker_chunks_path

        elif source.chunks_path.exists():
            chunks_path = source.chunks_path

        else:
            print(
                f"Skipping '{source.key}': "
                "no chunk file found"
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

        _, source_embeddings = build_dense_index(
            source_texts,
            cache_dir=source.embedding_cache_dir,
            embedding_model=embedding_model,
            model_name=settings.models.embedding_model,
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
    )

RetrievalMode = Literal[
    "global",
    "per_source",
]


def search_corpus_subset(
    query: str,
    index: CorpusIndex,
    chunk_indices: list[int],
    top_k: int,
) -> list[dict]:
    if not chunk_indices:
        return []

    subset_chunks = [
        index.chunks[i]
        for i in chunk_indices
    ]

    subset_texts = [
        index.texts[i]
        for i in chunk_indices
    ]

    subset_embeddings = (
        index.chunk_embeddings[chunk_indices]
    )

    if len(chunk_indices) == len(index.chunks):
        subset_bm25 = index.bm25
    else:
        subset_bm25 = build_bm25(
            subset_texts
        )

    return hybrid_search(
        query=query,
        chunks=subset_chunks,
        bm25=subset_bm25,
        embedding_model=index.embedding_model,
        chunk_embeddings=subset_embeddings,
        top_k=top_k,
    )


def search_corpus(
    query: str,
    index: CorpusIndex,
    top_k: int = 5,
    source_keys: list[str] | None = None,
    retrieval_mode: RetrievalMode = "global",
    top_k_per_source: int = 3,
) -> list[dict]:
    available_source_keys = list(
        dict.fromkeys(
            chunk["source_key"]
            for chunk in index.chunks
        )
    )

    if source_keys:
        selected_source_keys = [
            source_key
            for source_key in source_keys
            if source_key in available_source_keys
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
            source_indices = [
                i
                for i, chunk in enumerate(
                    index.chunks
                )
                if chunk["source_key"]
                == source_key
            ]

            source_results = (
                search_corpus_subset(
                    query=query,
                    index=index,
                    chunk_indices=source_indices,
                    top_k=top_k_per_source,
                )
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

    selected_indices = [
        i
        for i, chunk in enumerate(
            index.chunks
        )
        if chunk["source_key"]
        in selected_source_keys
    ]

    return search_corpus_subset(
        query=query,
        index=index,
        chunk_indices=selected_indices,
        top_k=top_k,
    )