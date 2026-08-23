from __future__ import annotations

import re
import time
from difflib import SequenceMatcher
from threading import Lock

from pydantic import BaseModel, Field

from src.corpus import CorpusIndex, search_corpus


PLANNER_SYSTEM_PROMPT = """
You are a retrieval planner for a grounded question-answering system over audio transcripts.

Infer the user's information need from their wording, the selected-source context, and query-specific transcript hints, then plan retrieval that is likely to gather enough transcript evidence to answer it. Do not answer the question yourself.

Rules:
1. You may choose only source_keys listed in the selected source catalog.
2. Treat source titles as metadata. Resolve approximate references to them from the catalog rather than requiring exact wording.
3. If the user is clearly referring to one or more selected recordings, scope retrieval to those recordings. Otherwise keep all selected recordings in scope.
4. Transcript hints are subordinate context for resolving ambiguous wording, typos, abbreviations, and shorthand. Preserve explicit subject matter in the user's question even when the selected corpus does not discuss it. A lack of matching evidence is a retrieval outcome, not permission to reinterpret the question so it fits unrelated corpus material.
5. Use transcript hints to normalize an unclear token only when the user's wording and corpus context reasonably support that normalization. Never replace a clear topic, entity, relationship, or requested concept with a different topic merely because that different topic appears in the hints.
6. Produce interpreted_question as a concise, neutral restatement of the user's intended information need in clear language. Preserve uncertainty if the request is genuinely ambiguous. Do not answer it and do not add facts.
7. Clarification is only for a genuinely ambiguous token or reference whose meaning is required before retrieval. If the user's topic or information need is understandable but the selected recordings may not contain relevant evidence, set needs_clarification to false and preserve the request; the grounded answer stage will handle insufficient evidence.
8. A clarification_question may ask what an ambiguous term or reference means, but it must not suggest candidate meanings, entities, topics, or facts from world knowledge. Never ask whether a clear user topic is a typo merely because the selected corpus discusses something else.
9. Set needs_clarification to false when the intent is reasonably clear from context, even if the wording contains typos, slang, abbreviations, shorthand, or poor grammar. Clarification is for genuine unresolved ambiguity, not imperfect writing.
10. Generate one or more concise semantic retrieval queries that together cover the user's information need. Decompose the request when doing so would improve evidence coverage.
11. Prefer language supported by the user's wording first, then use source metadata and transcript hints to normalize or expand it. Do not invent named entities, acronym expansions, or subject matter that are unsupported by the user and selected corpus context.
12. Return JSON only.
""".strip()


class RetrievalPlan(BaseModel):
    interpreted_question: str = ""
    needs_clarification: bool = False
    clarification_question: str = ""
    source_keys: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)


_PLAN_CACHE: dict[tuple, tuple[float, RetrievalPlan]] = {}
_PLAN_CACHE_LOCK = Lock()
_PLAN_CACHE_SECONDS = 90.0


_COMMON_SHORT_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does",
    "for", "from", "had", "has", "have", "hav", "he", "her", "him", "his", "how",
    "i", "if", "in", "is", "it", "its", "me", "my", "of", "on", "or", "our", "she",
    "so", "that", "the", "their", "them", "then", "they", "this", "to", "was", "we",
    "were", "what", "when", "where", "which", "who", "why", "will", "with", "wit",
    "you", "your", "all", "about", "tell", "give", "know", "say", "says", "said",
    "wdyk", "wdyt",
}

_CONTENT_STOPWORDS = _COMMON_SHORT_WORDS | {
    "being", "these", "those", "more", "less", "than", "very", "really", "just",
    "kind", "sort", "thing", "things", "something", "someone", "people", "guys",
}


def _plan_schema(allowed_source_keys: list[str]) -> dict:
    schema = RetrievalPlan.model_json_schema()
    schema["required"] = [
        "interpreted_question",
        "needs_clarification",
        "clarification_question",
        "source_keys",
        "queries",
    ]
    schema["additionalProperties"] = False
    schema["properties"]["source_keys"]["items"] = {
        "type": "string",
        "enum": allowed_source_keys,
    }
    schema["properties"]["queries"]["items"] = {"type": "string"}
    return schema


def _fallback_plan(query: str, selected_source_keys: list[str]) -> RetrievalPlan:
    cleaned = query.strip()
    return RetrievalPlan(
        interpreted_question=cleaned,
        needs_clarification=False,
        clarification_question="",
        source_keys=list(selected_source_keys),
        queries=[cleaned] if cleaned else [],
    )


def _hint_text(evidence_hints: list[dict] | None) -> str:
    if not evidence_hints:
        return "(none)"

    blocks = []
    for chunk in evidence_hints[:10]:
        source_name = chunk.get(
            "source_display_name",
            chunk.get("source_key", chunk.get("source_id", "Source")),
        )
        text = str(chunk.get("speaker_text") or chunk.get("text") or "").strip()
        if len(text) > 700:
            text = text[:700] + "…"
        blocks.append(
            f'- source: {source_name}\n'
            f'  chunk_id: {chunk.get("chunk_id")}\n'
            f'  excerpt: {text}'
        )
    return "\n\n".join(blocks) if blocks else "(none)"


def _support_text(
    source_catalog: list[dict],
    evidence_hints: list[dict] | None,
) -> str:
    parts = []
    for item in source_catalog:
        parts.append(str(item.get("display_name", "")))
        parts.append(str(item.get("preview", "")))
    for chunk in evidence_hints or []:
        parts.append(str(chunk.get("source_display_name", "")))
        parts.append(str(chunk.get("speaker_text") or chunk.get("text") or ""))
    return " ".join(parts).lower()


def _unknown_short_tokens(query: str, support: str) -> list[str]:
    tokens = re.findall(r"\b[a-zA-Z]{2,5}\b", query.lower())
    unknown = []
    for token in tokens:
        if token in _COMMON_SHORT_WORDS:
            continue
        if re.search(rf"\b{re.escape(token)}\b", support):
            continue
        if token not in unknown:
            unknown.append(token)
    return unknown


def _content_tokens(text: str) -> list[str]:
    tokens = []
    for token in re.findall(r"\b[a-zA-Z]{4,}\b", text.lower()):
        if token in _CONTENT_STOPWORDS:
            continue
        if token not in tokens:
            tokens.append(token)
    return tokens


def _tokens_match(left: str, right: str) -> bool:
    if left == right:
        return True
    if min(len(left), len(right)) < 4:
        return False
    return SequenceMatcher(None, left, right).ratio() >= 0.72


def _interpretation_drifted(query: str, interpreted_question: str) -> bool:
    """Detect large topic drift while still allowing typo correction/paraphrase."""
    query_tokens = _content_tokens(query)
    if len(query_tokens) < 3:
        return False

    interpreted_tokens = _content_tokens(interpreted_question)
    if not interpreted_tokens:
        return True

    matched = sum(
        any(_tokens_match(token, candidate) for candidate in interpreted_tokens)
        for token in query_tokens
    )
    return (matched / len(query_tokens)) < 0.34


def _unsupported_interpretation(
    query: str,
    interpreted_question: str,
    source_catalog: list[dict],
    evidence_hints: list[dict] | None,
) -> str | None:
    """Return the ambiguous raw token when the planner invents unsupported entities."""
    support = _support_text(source_catalog, evidence_hints)
    unknown_tokens = _unknown_short_tokens(query, support)
    if not unknown_tokens:
        return None

    expansion_pattern = re.compile(
        r"\b([A-Z][A-Za-z]+(?:\s+(?:[A-Z][A-Za-z]+|of|the|and|for)){1,6})\s*\(([A-Z]{2,10})\)"
    )
    for phrase, acronym in expansion_pattern.findall(interpreted_question):
        if acronym.lower() not in unknown_tokens:
            continue
        if phrase.lower() not in support:
            return acronym.lower()

    proper_name_pattern = re.compile(
        r"\b(?:[A-Z][a-z]{2,})(?:\s+(?:[A-Z][a-z]{2,})){1,4}\b"
    )
    for phrase in proper_name_pattern.findall(interpreted_question):
        lowered = phrase.lower()
        if lowered in support or lowered in query.lower():
            continue
        return unknown_tokens[0]

    return None


def plan_retrieval(
    query: str,
    selected_source_keys: list[str],
    source_catalog: list[dict],
    llm_client,
    evidence_hints: list[dict] | None = None,
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

    hint_signature = tuple(
        (
            str(item.get("source_key", "")),
            str(item.get("chunk_id", "")),
        )
        for item in (evidence_hints or [])[:10]
    )
    cache_key = (
        query.strip().lower(),
        tuple(allowed),
        tuple((item["source_key"], item["display_name"], item["preview"]) for item in catalog),
        hint_signature,
    )
    now = time.time()

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
            f"Query-specific transcript hints:\n{_hint_text(evidence_hints)}\n\n"
            "Important: the transcript hints may be irrelevant nearest-neighbor results. "
            "Preserve clear subject matter from the user even when those hints discuss something else. "
            "If the request itself is understandable but unsupported by the selected recordings, do not clarify; retrieve it faithfully and let the answer stage refuse. "
            "Never offer world-knowledge candidate meanings in a clarification.\n\n"
            "Return the retrieval plan."
        )

        try:
            raw = llm_client.generate(
                system_prompt=PLANNER_SYSTEM_PROMPT,
                user_prompt=prompt,
                response_schema=_plan_schema(allowed),
                max_tokens=384,
            )
            plan = RetrievalPlan.model_validate_json(raw)
        except Exception:
            plan = _fallback_plan(query, allowed)

        valid_sources = [key for key in plan.source_keys if key in allowed]
        if not valid_sources:
            valid_sources = allowed

        interpreted_question = plan.interpreted_question.strip() or query.strip()
        needs_clarification = bool(plan.needs_clarification)
        clarification_question = plan.clarification_question.strip()

        unsupported_token = _unsupported_interpretation(
            query=query,
            interpreted_question=interpreted_question,
            source_catalog=catalog,
            evidence_hints=evidence_hints,
        )
        force_raw_query = False
        if unsupported_token:
            needs_clarification = True
            clarification_question = (
                f'What do you mean by "{unsupported_token}"? '
                "I can’t reliably infer that term from the selected recordings."
            )
        elif needs_clarification and len(_content_tokens(query)) >= 2:
            # For a search product, a readable out-of-corpus question should be
            # searched faithfully and refused if unsupported. Do not turn corpus
            # mismatch into speculative chatbot-style clarification.
            needs_clarification = False
            clarification_question = ""
            interpreted_question = query.strip()
            force_raw_query = True

        if not needs_clarification and _interpretation_drifted(query, interpreted_question):
            interpreted_question = query.strip()
            force_raw_query = True

        if needs_clarification and not clarification_question:
            clarification_question = (
                "Could you clarify the ambiguous term or reference in your question?"
            )
        if not needs_clarification:
            clarification_question = ""

        queries = []
        seen_queries = set()
        if force_raw_query:
            cleaned = query.strip()
            if cleaned:
                queries = [cleaned]
        else:
            for item in plan.queries:
                cleaned = item.strip()
                normalized = cleaned.lower()
                if cleaned and normalized not in seen_queries:
                    queries.append(cleaned)
                    seen_queries.add(normalized)
                if len(queries) == 4:
                    break
        if not queries:
            queries = [interpreted_question or query.strip()]

        normalized_plan = RetrievalPlan(
            interpreted_question=interpreted_question,
            needs_clarification=needs_clarification,
            clarification_question=clarification_question,
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
        return [], RetrievalPlan(
            interpreted_question=query.strip(),
            needs_clarification=False,
            clarification_question="",
            source_keys=[],
            queries=[],
        )

    display_names = {
        item["source_key"]: item["display_name"]
        for item in source_catalog
    }

    bootstrap = search_corpus(
        query=query,
        index=index,
        top_k=max(8, min(12, top_k * 2)),
        source_keys=selected,
        retrieval_mode="global",
        top_k_per_source=top_k_per_source,
    )
    for chunk in bootstrap:
        chunk["source_display_name"] = display_names.get(
            chunk.get("source_key"),
            chunk.get("source_id", "Source"),
        )

    plan = plan_retrieval(
        query=query,
        selected_source_keys=selected,
        source_catalog=source_catalog,
        llm_client=llm_client,
        evidence_hints=bootstrap,
    )

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
            query=plan.interpreted_question or query,
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

    for chunk in results:
        chunk["source_display_name"] = display_names.get(
            chunk.get("source_key"),
            chunk.get("source_id", "Source"),
        )
        chunk["retrieval_interpretation"] = plan.interpreted_question

    return results, plan
