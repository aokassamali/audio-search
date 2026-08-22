from __future__ import annotations

import time
from threading import Lock

from pydantic import BaseModel, Field, ValidationError

from src.corpus import CorpusIndex, search_corpus


PLANNER_SYSTEM_PROMPT = """
You are a retrieval planner for a grounded question-answering system over audio transcripts.

Your job is NOT to answer the user's question. Your job is to decide which selected recordings are relevant and write a small set of search queries that will retrieve the evidence needed to answer it.

Rules:
1. You may choose only source_keys listed in the selected source catalog.
2. Resolve obvious misspellings, abbreviations, punctuation differences, and approximate references to source titles. A user may type a source title imperfectly.
3. If the user clearly refers to one or more source titles, choose those sources. Otherwise keep all selected sources in scope.
4. Remove source-title wording from semantic retrieval queries when it is merely identifying the recording.
5. Use the short transcript previews only to understand what each recording is about and to choose useful retrieval language. Do not answer from the previews.
6. For synthesis, comparison, "positions of each side", arguments, disagreements, causes, timelines, or other multi-part questions, decompose the request into 2-4 complementary retrieval queries so the evidence covers the distinct parts of the answer.
7. Write concise retrieval queries using language likely to appear in the transcript. It is fine to use concepts revealed by the source preview, but do not invent facts.
8. Return JSON only.
""".strip()


class RetrievalPlan(BaseModel):
    source_keys: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)


_PLAN_CACHE: dict[tuple, tuple[float, RetrievalPlan]] = {}
_PLAN_CACHE_LOCK = Lock()
_PLAN_CACHE_SECONDS = 90.0


def _plan_schema(allowed_source_keys: list[str]) -> dict:
    schema = RetrievalPlan.model_json_schema()
    schema["required"] = ["source_keys", "queries"]
    schema["additionalProperties"] = False
    schema["properties"]["source_keys"]["items"] = {
        "type": "string",
        "enum": allowed_source_keys,
    }
    schema["properties"]["queries"]["items"] = {"type": "string"}
    return schema


def _fallback_plan(query: str, selected_source_keys: list[str]) -> RetrievalPlan:
    return RetrievalPlan(
        source_keys=list(selected_source_keys),
        queries=[query.strip()] if query.strip() else [],
    )


def plan_retrieval(
    query: str,
    selected_source_keys: list[str],
    source_catalog: list[dict],
    llm_client,
) -> RetrievalPlan:
    allowed = [
        item["source_key"]
        for item in source_catalog
        if item["source_key"] in selected_source_keys
    ]
    if not allowed:
        return _fallback_plan(query, selected_source_keys)

    catalog = [
        {
            "source_key": item["source_key"],
            "display_name": item["display_name"],
            "preview": item.get("preview", ""),
        }
        for item in source_catalog
        if item["source_key"] in allowed
    ]

    cache_key = (
        query.strip().lower(),
        tuple(allowed),
        tuple((item["source_key"], item["display_name"], item["preview"]) for item in catalog),
    )
    now = time.time()

    # Hold the lock while planning so the frontend's concurrent /search and
    # /answer calls share one planner request instead of hitting the local LLM twice.
    with _PLAN_CACHE_LOCK:
        cached = _PLAN_CACHE.get(cache_key)
        if cached and now - cached[0] <= _PLAN_CACHE_SECONDS:
            return cached[1]

        catalog_text = "\n\n".join(
            (
                f'- source_key: {item["source_key"]}\n'
                f'  title: {item["display_name"]}\n'
                f'  preview: {item["preview"]}'
            )
            for item in catalog
        )
        prompt = (
            f"User question:\n{query}\n\n"
            f"Selected source catalog:\n{catalog_text}\n\n"
            "Return the retrieval plan."
        )

        try:
            raw = llm_client.generate(
                system_prompt=PLANNER_SYSTEM_PROMPT,
                user_prompt=prompt,
                response_schema=_plan_schema(allowed),
                max_tokens=256,
            )
            plan = RetrievalPlan.model_validate_json(raw)
        except Exception:
            # Retrieval must remain usable even if the planner model is unavailable.
            plan = _fallback_plan(query, allowed)

        valid_sources = [key for key in plan.source_keys if key in allowed]
        if not valid_sources:
            valid_sources = allowed

        queries = []
        seen_queries = set()
        for item in plan.queries:
            cleaned = item.strip()
            normalized = cleaned.lower()
            if cleaned and normalized not in seen_queries:
                queries.append(cleaned)
                seen_queries.add(normalized)
            if len(queries) == 4:
                break
        if not queries:
            queries = [query.strip()]

        normalized_plan = RetrievalPlan(
            source_keys=valid_sources,
            queries=queries,
        )
        _PLAN_CACHE[cache_key] = (now, normalized_plan)
        return normalized_plan


def _catalog_with_previews(
    index: CorpusIndex,
    source_catalog: list[dict],
) -> list[dict]:
    enriched = []
    for item in source_catalog:
        source_index = index.sources.get(item["source_key"])
        preview_parts = []
        if source_index is not None:
            for chunk in source_index.chunks[:3]:
                text = chunk.get("speaker_text") or chunk.get("text") or ""
                if text:
                    preview_parts.append(str(text).strip())
        preview = " ".join(preview_parts)
        if len(preview) > 1800:
            preview = preview[:1800] + "…"
        enriched.append({**item, "preview": preview})
    return enriched


def retrieve_with_plan(
    query: str,
    index: CorpusIndex,
    selected_source_keys: list[str] | None,
    source_catalog: list[dict],
    llm_client,
    top_k: int,
    top_k_per_source: int = 3,
) -> tuple[list[dict], RetrievalPlan]:
    source_catalog = _catalog_with_previews(index, source_catalog)
    available = [item["source_key"] for item in source_catalog]
    selected = (
        [key for key in selected_source_keys if key in available]
        if selected_source_keys is not None
        else available
    )
    if not selected:
        return [], RetrievalPlan(source_keys=[], queries=[])

    plan = plan_retrieval(
        query=query,
        selected_source_keys=selected,
        source_catalog=source_catalog,
        llm_client=llm_client,
    )

    # Retrieve several candidates for each sub-question, then round-robin them.
    # This prevents a broad synthesis question from spending all of its evidence
    # budget on one side of the issue.
    per_query_k = max(3, min(6, top_k))
    ranked_lists = []
    for retrieval_query in plan.queries:
        ranked_lists.append(
            search_corpus(
                query=retrieval_query,
                index=index,
                top_k=per_query_k,
                source_keys=plan.source_keys,
                retrieval_mode="global",
                top_k_per_source=top_k_per_source,
            )
        )

    results = []
    seen = set()
    cursor = 0
    while len(results) < top_k and any(cursor < len(items) for items in ranked_lists):
        for items in ranked_lists:
            if cursor >= len(items):
                continue
            chunk = items[cursor]
            key = (chunk.get("source_key"), chunk.get("chunk_id"))
            if key not in seen:
                seen.add(key)
                results.append(dict(chunk))
                if len(results) >= top_k:
                    break
        cursor += 1

    if len(results) < top_k:
        fallback = search_corpus(
            query=query,
            index=index,
            top_k=top_k,
            source_keys=plan.source_keys,
            retrieval_mode="global",
            top_k_per_source=top_k_per_source,
        )
        for chunk in fallback:
            key = (chunk.get("source_key"), chunk.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            results.append(dict(chunk))
            if len(results) >= top_k:
                break

    display_names = {
        item["source_key"]: item["display_name"]
        for item in source_catalog
    }
    for chunk in results:
        chunk["source_display_name"] = display_names.get(
            chunk.get("source_key"),
            chunk.get("source_id", "Source"),
        )

    return results, plan
