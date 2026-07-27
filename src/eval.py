import argparse
import json

from src.config import load_settings
from src.search import (
    build_bm25,
    build_dense_index,
    extract_texts,
    load_chunks,
    rank_bm25,
    rank_dense,
    reciprocal_rank_fusion,
)


def load_eval_queries(queries_path):
    with open(queries_path, "r", encoding="utf-8") as file:
        eval_queries = json.load(file)

    return eval_queries


def recall_at_k(retrieved_ids, relevant_ids, k=5):
    retrieved_top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    relevant_retrieved = retrieved_top_k.intersection(relevant_set)

    recall = len(relevant_retrieved) / len(relevant_set)

    return recall


def reciprocal_rank(retrieved_ids, relevant_ids):
    relevant_set = set(relevant_ids)

    for rank, retrieved_id in enumerate(retrieved_ids, start=1):
        if retrieved_id in relevant_set:
            return 1 / rank

    return 0


def indices_to_chunk_ids(ranked_indices, chunks):
    ranked_ids = []

    for chunk_index in ranked_indices:
        chunk_id = chunks[chunk_index]["chunk_id"]
        ranked_ids.append(chunk_id)

    return ranked_ids


def main():
    settings = load_settings()

    parser = argparse.ArgumentParser()
    parser.add_argument("chunks")
    parser.add_argument("queries")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    eval_queries = load_eval_queries(args.queries)

    texts = extract_texts(chunks)

    bm25 = build_bm25(texts)

    embedding_model, chunk_embeddings = (
    build_dense_index(
        texts,
        model_name=(
            settings.models.embedding_model
        ),
    )
)

    bm25_recalls = []
    dense_recalls = []
    rrf_recalls = []

    bm25_reciprocal_ranks = []
    dense_reciprocal_ranks = []
    rrf_reciprocal_ranks = []

    for eval_item in eval_queries:
        eval_query = eval_item["query"]
        relevant_chunk_ids = eval_item["relevant_chunk_ids"]

        bm25_ranked_indices = rank_bm25(
            eval_query,
            bm25,
        )

        dense_ranked_indices = rank_dense(
            eval_query,
            embedding_model,
            chunk_embeddings,
        )

        fused_ranked_indices, _ = reciprocal_rank_fusion(
            bm25_ranked_indices,
            dense_ranked_indices,
        )

        bm25_ranked_ids = indices_to_chunk_ids(
            bm25_ranked_indices,
            chunks,
        )

        dense_ranked_ids = indices_to_chunk_ids(
            dense_ranked_indices,
            chunks,
        )

        fused_ranked_ids = indices_to_chunk_ids(
            fused_ranked_indices,
            chunks,
        )

        bm25_recall = recall_at_k(
            bm25_ranked_ids,
            relevant_chunk_ids,
        )

        dense_recall = recall_at_k(
            dense_ranked_ids,
            relevant_chunk_ids,
        )

        rrf_recall = recall_at_k(
            fused_ranked_ids,
            relevant_chunk_ids,
        )

        bm25_rr = reciprocal_rank(
            bm25_ranked_ids,
            relevant_chunk_ids,
        )

        dense_rr = reciprocal_rank(
            dense_ranked_ids,
            relevant_chunk_ids,
        )

        rrf_rr = reciprocal_rank(
            fused_ranked_ids,
            relevant_chunk_ids,
        )

        bm25_recalls.append(bm25_recall)
        dense_recalls.append(dense_recall)
        rrf_recalls.append(rrf_recall)

        bm25_reciprocal_ranks.append(bm25_rr)
        dense_reciprocal_ranks.append(dense_rr)
        rrf_reciprocal_ranks.append(rrf_rr)

    bm25_average_recall = sum(bm25_recalls) / len(bm25_recalls)
    dense_average_recall = sum(dense_recalls) / len(dense_recalls)
    rrf_average_recall = sum(rrf_recalls) / len(rrf_recalls)

    bm25_mrr = (
        sum(bm25_reciprocal_ranks)
        / len(bm25_reciprocal_ranks)
    )

    dense_mrr = (
        sum(dense_reciprocal_ranks)
        / len(dense_reciprocal_ranks)
    )

    rrf_mrr = (
        sum(rrf_reciprocal_ranks)
        / len(rrf_reciprocal_ranks)
    )

    print("\nRetrieval Evaluation")
    print(f"{'Mode':<10} {'Recall@5':>10} {'MRR':>10}")
    print("-" * 32)
    print(
        f"{'BM25':<10} "
        f"{bm25_average_recall:>10.3f} "
        f"{bm25_mrr:>10.3f}"
    )
    print(
        f"{'Dense':<10} "
        f"{dense_average_recall:>10.3f} "
        f"{dense_mrr:>10.3f}"
    )
    print(
        f"{'RRF':<10} "
        f"{rrf_average_recall:>10.3f} "
        f"{rrf_mrr:>10.3f}"
    )


if __name__ == "__main__":
    main()