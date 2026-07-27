import argparse
import json
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import hashlib
from pathlib import Path
import numpy as np


EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


def create_embedding_cache_key(texts):
    cache_input = {
        "model_name": EMBEDDING_MODEL_NAME,
        "texts": texts,
    }

    serialized_input = json.dumps(
        cache_input,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    cache_key = hashlib.sha256(
        serialized_input.encode("utf-8")
    ).hexdigest()

    return cache_key


def load_chunks(chunks_path):
    with open(chunks_path, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    return chunks


def extract_texts(chunks):
    texts = []

    for chunk in chunks:
        texts.append(chunk["text"])
    
    return texts


def build_bm25(texts):
    tokenized_texts = []

    for text in texts:
        tokens = text.lower().split()
        tokenized_texts.append(tokens)

    bm25 = BM25Okapi(tokenized_texts)

    return bm25


def rank_bm25(query, bm25):
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_ranked_indices = bm25_scores.argsort()[::-1]

    return bm25_ranked_indices


def build_dense_index(
    texts,
    cache_dir=None,
    embedding_model=None,
):
    if embedding_model is None:
        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )
    if cache_dir is None:
        chunk_embeddings = embedding_model.encode(
            texts,
            normalize_embeddings=True,
        )

        return embedding_model, chunk_embeddings

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings_path = cache_dir / "chunk_embeddings.npy"
    metadata_path = cache_dir / "chunk_embeddings_metadata.json"

    current_cache_key = create_embedding_cache_key(
        texts
    )

    cached_metadata = {}

    if metadata_path.exists():
        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            cached_metadata = json.load(file)

    cache_is_valid = (
        embeddings_path.exists()
        and cached_metadata.get("cache_key")
        == current_cache_key
    )

    if cache_is_valid:
        chunk_embeddings = np.load(
            embeddings_path,
            allow_pickle=False,
        )

        print("Loaded chunk embeddings from cache")

    else:
        chunk_embeddings = embedding_model.encode(
            texts,
            normalize_embeddings=True,
        )

        np.save(
            embeddings_path,
            chunk_embeddings,
        )

        metadata = {
            "cache_key": current_cache_key,
            "model_name": EMBEDDING_MODEL_NAME,
            "chunk_count": len(texts),
        }

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=2,
            )

        print("Built and cached chunk embeddings")

    return embedding_model, chunk_embeddings


def rank_dense(query, embedding_model, chunk_embeddings):
    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True,
    )

    dense_scores = chunk_embeddings @ query_embedding

    dense_ranked_indices = dense_scores.argsort()[::-1]

    return dense_ranked_indices


def reciprocal_rank_fusion(
    bm25_ranked_indices,
    dense_ranked_indices,
    candidate_count=20,
    rrf_k=60,
):
    rrf_scores = {}

    bm25_top = bm25_ranked_indices[:candidate_count]
    dense_top = dense_ranked_indices[:candidate_count]

    for ranking in [bm25_top, dense_top]:
        for rank, chunk_index in enumerate(ranking, start=1):
            chunk_index = int(chunk_index)
            contribution = 1 / (rrf_k + rank)

            current_score = rrf_scores.get(chunk_index, 0)
            rrf_scores[chunk_index] = current_score + contribution

    fused_ranked_indices = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True,
    )

    return fused_ranked_indices, rrf_scores


def hybrid_search(
    query: str,
    chunks: list[dict],
    bm25,
    embedding_model,
    chunk_embeddings,
    top_k: int = 5,
) -> list[dict]:
    bm25_ranked_indices = rank_bm25(
        query,
        bm25,
    )

    dense_ranked_indices = rank_dense(
        query,
        embedding_model,
        chunk_embeddings,
    )

    fused_ranked_indices, rrf_scores = (
        reciprocal_rank_fusion(
            bm25_ranked_indices,
            dense_ranked_indices,
        )
    )

    results = []

    for rank, chunk_index in enumerate(
        fused_ranked_indices[:top_k],
        start=1,
    ):
        result = chunks[chunk_index].copy()

        result["rank"] = rank
        result["rrf_score"] = rrf_scores[
            chunk_index
        ]

        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("chunks")
    parser.add_argument("query")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    texts = extract_texts(chunks)
    
    bm25 = build_bm25(texts)
    bm25_ranked_indices = rank_bm25(args.query, bm25)

    embedding_model, chunk_embeddings = build_dense_index(texts)
    dense_ranked_indices = rank_dense(args.query, embedding_model, chunk_embeddings)

    fused_ranked_indices, rrf_scores = reciprocal_rank_fusion(bm25_ranked_indices, dense_ranked_indices)

    fused_top_5 = fused_ranked_indices[:5]

    print("Fused top five:", fused_top_5)

    for rank, chunk_index in enumerate(fused_top_5, start=1):
        chunk = chunks[chunk_index]

        print(f"\nFused rank: {rank}")
        print(f"Chunk index: {chunk_index}")
        print(f"RRF score: {rrf_scores[chunk_index]}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Start: {chunk['start']}")
        print(f"End: {chunk['end']}")
        print(f"Text: {chunk['text']}")

if __name__ == "__main__":
    main()

