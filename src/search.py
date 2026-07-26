import argparse
import json
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

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


def build_dense_index(texts):
    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    chunk_embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
    )

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

