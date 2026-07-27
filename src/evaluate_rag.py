import json
from pathlib import Path

from src.llm_clients import LlamaCppClient
from src.rag import answer_question, create_citation_id
from src.search import (
    build_bm25,
    build_dense_index,
    extract_texts,
    hybrid_search,
    load_chunks,
)


CHUNKS_PATH = Path(
    "data/processed/chunks/Sripetch_vs_SEC_chunks_20260726_145257.json"
)

EMBEDDING_CACHE_DIR = Path(
    "data/cache"
)

EVAL_PATH = Path(
    "data/eval/rag_eval.json"
)

RESULTS_PATH = Path(
    "data/eval/rag_eval_results.json"
)


def evaluate_rag() -> None:
    with EVAL_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        eval_cases = json.load(file)

    chunks = load_chunks(CHUNKS_PATH)
    texts = extract_texts(chunks)

    bm25 = build_bm25(texts)

    embedding_model, chunk_embeddings = (
        build_dense_index(
            texts,
            cache_dir=EMBEDDING_CACHE_DIR,
        )
    )

    llm_client = LlamaCppClient()

    results = []

    for case in eval_cases:
        print(f"Running: {case['id']}")

        retrieved_chunks = hybrid_search(
            query=case["query"],
            chunks=chunks,
            bm25=bm25,
            embedding_model=embedding_model,
            chunk_embeddings=chunk_embeddings,
            top_k=5,
        )

        answer = answer_question(
            query=case["query"],
            retrieved_chunks=retrieved_chunks,
            llm_client=llm_client,
        )

        result = {
            "id": case["id"],
            "query": case["query"],
            "expected_answerable": (
                case["expected_answerable"]
            ),
            "actual_answerable": answer.answerable,
            "answerability_match": (
                answer.answerable
                == case["expected_answerable"]
            ),
            "answer": answer.answer,
            "citations": [
                citation.model_dump()
                for citation in answer.citations
            ],
            "retrieved_citation_ids": [
                create_citation_id(chunk)
                for chunk in retrieved_chunks
            ],
            "notes": case.get("notes", ""),
        }

        results.append(result)

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    passed = sum(
        result["answerability_match"]
        for result in results
    )

    print(
        f"\nAnswerability: "
        f"{passed}/{len(results)} correct"
    )

    print(
        f"Results written to: {RESULTS_PATH}"
    )


if __name__ == "__main__":
    evaluate_rag()