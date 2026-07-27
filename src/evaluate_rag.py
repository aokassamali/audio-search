import json

from sentence_transformers import SentenceTransformer

from src.config import load_settings
from src.llm_clients import LlamaCppClient
from src.rag import answer_question, create_citation_id
from src.search import (
    build_bm25,
    build_dense_index,
    extract_texts,
    hybrid_search,
    load_chunks,
)


def evaluate_rag(
    source_key: str | None = None,
) -> None:
    settings = load_settings()
    source = settings.get_source(source_key)

    eval_path = settings.rag_eval_path
    results_path = settings.rag_eval_results_path

    with eval_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        eval_cases = json.load(file)

    chunks = load_chunks(
        source.active_chunks_path
    )

    texts = extract_texts(chunks)
    bm25 = build_bm25(texts)

    configured_embedding_model = SentenceTransformer(
        settings.models.embedding_model
    )

    embedding_model, chunk_embeddings = (
        build_dense_index(
            texts,
            cache_dir=source.embedding_cache_dir,
            embedding_model=configured_embedding_model,
            model_name=settings.models.embedding_model,
        )
    )

    llm_client = LlamaCppClient(
        base_url=settings.llm.base_url,
        model=settings.llm.model,
        timeout=settings.llm.timeout_seconds,
    )

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
            "source_key": source.key,
            "source_id": source.source_id,
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

    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with results_path.open(
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
        f"\nSource: {source.key}"
        f"\nAnswerability: "
        f"{passed}/{len(results)} correct"
    )

    print(
        f"Results written to: {results_path}"
    )


if __name__ == "__main__":
    evaluate_rag()