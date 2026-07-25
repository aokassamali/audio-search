import argparse
import json
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("chunks")
    parser.add_argument("query")
    args = parser.parse_args()

    query_tokens = args.query.lower().split()

    with open(args.chunks, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    texts = []

    for chunk in chunks:
        texts.append(chunk["text"])

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    chunk_embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
    )

    query_embedding = embedding_model.encode(
        args.query,
        normalize_embeddings=True,
    )

    dense_scores = chunk_embeddings @ query_embedding

    dense_ranked_indices = dense_scores.argsort()[::-1]

    tokenized_texts = []

    for text in texts:
        tokens = text.lower().split()
        tokenized_texts.append(tokens)

    bm25 = BM25Okapi(tokenized_texts)

    bm25_scores = bm25.get_scores(query_tokens)

    bm25_ranked_indices = bm25_scores.argsort()[::-1]

    bm25_top_20 = bm25_ranked_indices[:20]
    dense_top_20 = dense_ranked_indices[:20]
    
    rrf_scores = {}

    RRF_K = 60

    for ranking in [bm25_top_20, dense_top_20]:
        for rank, chunk_index in enumerate(ranking, start=1):
            contribution = 1 / (RRF_K + rank)

            current_score = rrf_scores.get(chunk_index, 0)
            rrf_scores[chunk_index] = current_score + contribution

    fused_ranked_indices = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True,
    )

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

