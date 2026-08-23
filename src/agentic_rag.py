from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import json
import re
import time
from typing import Literal

from pydantic import BaseModel, Field

from src.corpus import CorpusIndex, search_corpus
from src.rag import (
    Citation,
    GroundedAnswer,
    LLMAnswerDraft,
    REFUSAL_TEXT,
    create_citation_id,
    finalize_answer,
)


AGENT_SYSTEM_PROMPT = """
You are the retrieval controller and grounded answerer for an audio-transcript search system.

You do not know the recordings from model memory. The only factual evidence you may use in an answer is transcript evidence returned by the retrieval tools in this session. Source titles and metadata may be used to decide where to search, but they are not factual evidence for the answer.

You receive:
- the user's natural-language question,
- the selected recording catalog,
- transcript evidence gathered so far,
- the number of retrieval rounds remaining.

Available retrieval tools:
1. search_transcripts
   Semantic + lexical search over transcript chunks. Use for names, topics, paraphrases, comparisons, and targeted questions.
2. read_source
   Read one ordered slice of a recording by chunk offset. Use when nearby chronology or a particular section matters.
3. sample_source
   Read chunks distributed across an entire recording. Use when broad coverage of a recording is useful and the current evidence is too narrow.
4. read_context
   Read chunks immediately around a known chunk. Use when a retrieved passage needs neighboring context.

Available final actions:
- answer: answer only from gathered transcript evidence and cite evidence IDs.
- refuse: use when reasonable retrieval has not found enough evidence.
- clarify: use only when the user's meaning remains materially ambiguous after consulting the selected catalog and available transcript evidence.

Behavior rules:
- Do not classify the request into a fixed intent taxonomy. Decide directly what information would help and which retrieval tools can obtain it.
- Resolve obvious typos, shorthand, and approximate recording-title references from the selected source catalog. Do not clarify when one catalog match is clearly best.
- If the question is understandable but unsupported by the selected recordings, retrieve reasonably and then refuse. Do not reinterpret it toward unrelated corpus content.
- Clarification is a last resort. Never propose world-knowledge meanings for an unclear term.
- Stay inside the selected-source boundary. Never request or cite an unselected source.
- Prefer one purposeful gather step containing multiple retrieval requests over serially requesting one small slice at a time.
- For a broad question about one recording, sample the source or request a few strategically different sections in the same gather step instead of walking offsets 0, 8, 16, ... across multiple model turns.
- If the bootstrap evidence is already sufficient, answer immediately.
- When no retrieval rounds remain, choose answer, refuse, or clarify. Do not request more retrieval.
- Keep final answers concise: usually 3-6 sentences and under about 160 words unless the user explicitly asks for more detail.
- Return JSON only.

Transport note: when action is gather, use the flat tool slots tool_1/tool_2/tool_3 and their matching argument fields. Set unused tool slots to "none". Do not return nested tool-call objects.
""".strip()


ToolName = Literal[
    "search_transcripts",
    "read_source",
    "sample_source",
    "read_context",
]
ToolSlot = Literal[
    "none",
    "search_transcripts",
    "read_source",
    "sample_source",
    "read_context",
]
ActionName = Literal[
    "gather",
    "answer",
    "refuse",
    "clarify",
]


class RetrievalRequest(BaseModel):
    tool: ToolName
    query: str = ""
    source_keys: list[str] = Field(default_factory=list)
    source_key: str = ""
    offset: int = 0
    limit: int = 5
    chunk_id: int = -1
    radius: int = 2


class AgentDecision(BaseModel):
    action: ActionName

    tool_1: ToolSlot = "none"
    query_1: str = ""
    source_key_1: str = ""
    offset_1: int = 0
    limit_1: int = 5
    chunk_id_1: int = -1
    radius_1: int = 2

    tool_2: ToolSlot = "none"
    query_2: str = ""
    source_key_2: str = ""
    offset_2: int = 0
    limit_2: int = 5
    chunk_id_2: int = -1
    radius_2: int = 2

    tool_3: ToolSlot = "none"
    query_3: str = ""
    source_key_3: str = ""
    offset_3: int = 0
    limit_3: int = 5
    chunk_id_3: int = -1
    radius_3: int = 2

    answer: str = ""
    citation_ids: list[str] = Field(default_factory=list)
    clarification_question: str = ""


class AgentTraceStep(BaseModel):
    iteration: int
    action: str
    detail: str
    result_count: int = 0
    elapsed_ms: int = 0


class AskResult(BaseModel):
    outcome: Literal["answer", "refusal", "clarification", "error"]
    answerable: bool
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[dict] = Field(default_factory=list)
    trace: list[AgentTraceStep] = Field(default_factory=list)


@dataclass
class _AgentState:
    evidence: dict[str, dict]
    trace: list[AgentTraceStep]


class ControllerParseError(ValueError):
    def __init__(self, message: str, elapsed_ms: int = 0):
        super().__init__(message)
        self.elapsed_ms = elapsed_ms


def _decision_schema(
    allowed_source_keys: list[str],
    evidence_ids: list[str],
) -> dict:
    """Flat schema on purpose: local llama.cpp structured output is more reliable
    with top-level primitive fields than nested arrays of tool-call objects."""
    schema = AgentDecision.model_json_schema()
    schema["required"] = ["action"]
    schema["additionalProperties"] = False

    for slot in (1, 2, 3):
        schema["properties"][f"source_key_{slot}"] = {
            "type": "string",
            "enum": ["", *allowed_source_keys],
        }

    schema["properties"]["citation_ids"]["items"] = (
        {"type": "string", "enum": evidence_ids}
        if evidence_ids
        else {"type": "string"}
    )
    return schema


def _coerce_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_tool_name(value: object) -> str:
    name = str(value or "").strip().lower()
    aliases = {
        "search": "search_transcripts",
        "search_transcript": "search_transcripts",
        "read": "read_source",
        "sample": "sample_source",
        "context": "read_context",
    }
    return aliases.get(name, name)


def _extract_json_object(raw: str) -> dict:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                raise exc
        else:
            raise
    if not isinstance(data, dict):
        raise ValueError("controller response was not a JSON object")
    return data


def _parse_decision(raw: str) -> AgentDecision:
    """Accept the flat wire format and a few harmless legacy/common variants.

    The model is still constrained by the selected-source and citation checks in
    Python; normalization here only prevents benign JSON-shape differences from
    turning into a false product failure.
    """
    data = _extract_json_object(raw)

    nested_requests = data.pop("requests", None)
    if nested_requests is None:
        nested_requests = data.pop("tool_calls", None)

    action = str(data.get("action") or "").strip().lower()
    direct_tool = _normalize_tool_name(action)
    if direct_tool in {"search_transcripts", "read_source", "sample_source", "read_context"}:
        nested_requests = [dict(data, tool=direct_tool)]
        action = "gather"

    if not action:
        if nested_requests:
            action = "gather"
        elif data.get("clarification_question"):
            action = "clarify"
        elif data.get("answer"):
            action = "answer"
        elif data.get("answerable") is False:
            action = "refuse"
    action_aliases = {
        "retrieve": "gather",
        "search": "gather",
        "not_enough_evidence": "refuse",
        "not enough evidence": "refuse",
    }
    data["action"] = action_aliases.get(action, action)

    if nested_requests:
        if isinstance(nested_requests, dict):
            nested_requests = [nested_requests]
        if isinstance(nested_requests, list):
            for slot, item in enumerate(nested_requests[:3], start=1):
                if not isinstance(item, dict):
                    continue
                function = item.get("function")
                if isinstance(function, dict):
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    merged = dict(arguments) if isinstance(arguments, dict) else {}
                    merged["tool"] = function.get("name", item.get("tool", ""))
                    item = merged

                source_keys = item.get("source_keys")
                source_key = item.get("source_key")
                if not source_key and isinstance(source_keys, list) and len(source_keys) == 1:
                    source_key = source_keys[0]

                data[f"tool_{slot}"] = _normalize_tool_name(
                    item.get("tool") or item.get("name") or item.get("action")
                ) or "none"
                data[f"query_{slot}"] = str(item.get("query") or "")
                data[f"source_key_{slot}"] = str(source_key or "")
                data[f"offset_{slot}"] = _coerce_int(item.get("offset"), 0)
                data[f"limit_{slot}"] = _coerce_int(item.get("limit"), 5)
                data[f"chunk_id_{slot}"] = _coerce_int(item.get("chunk_id"), -1)
                data[f"radius_{slot}"] = _coerce_int(item.get("radius"), 2)

    for slot in (1, 2, 3):
        data[f"tool_{slot}"] = _normalize_tool_name(data.get(f"tool_{slot}")) or "none"
        data[f"query_{slot}"] = str(data.get(f"query_{slot}") or "")
        data[f"source_key_{slot}"] = str(data.get(f"source_key_{slot}") or "")
        data[f"offset_{slot}"] = _coerce_int(data.get(f"offset_{slot}"), 0)
        data[f"limit_{slot}"] = _coerce_int(data.get(f"limit_{slot}"), 5)
        data[f"chunk_id_{slot}"] = _coerce_int(data.get(f"chunk_id_{slot}"), -1)
        data[f"radius_{slot}"] = _coerce_int(data.get(f"radius_{slot}"), 2)

    citation_ids = data.get("citation_ids", [])
    if isinstance(citation_ids, str):
        citation_ids = [citation_ids]
    if not isinstance(citation_ids, list):
        citation_ids = []
    data["citation_ids"] = [str(value) for value in citation_ids if value is not None]
    data["answer"] = str(data.get("answer") or "")
    data["clarification_question"] = str(data.get("clarification_question") or "")

    allowed_fields = set(AgentDecision.model_fields)
    data = {key: value for key, value in data.items() if key in allowed_fields}
    return AgentDecision.model_validate(data)


def _decision_requests(decision: AgentDecision) -> list[RetrievalRequest]:
    requests: list[RetrievalRequest] = []
    for slot in (1, 2, 3):
        tool = getattr(decision, f"tool_{slot}")
        if tool == "none":
            continue
        source_key = getattr(decision, f"source_key_{slot}")
        requests.append(
            RetrievalRequest(
                tool=tool,
                query=getattr(decision, f"query_{slot}"),
                source_keys=[source_key] if source_key else [],
                source_key=source_key,
                offset=getattr(decision, f"offset_{slot}"),
                limit=getattr(decision, f"limit_{slot}"),
                chunk_id=getattr(decision, f"chunk_id_{slot}"),
                radius=getattr(decision, f"radius_{slot}"),
            )
        )
    return requests


def _normalized_words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _title_match_score(query: str, title: str) -> float:
    """Fuzzy metadata match without interpreting the user's substantive intent."""
    query_words = _normalized_words(query)
    title_words = _normalized_words(title)
    if not query_words or not title_words:
        return 0.0

    title_text = " ".join(title_words)
    candidates = [" ".join(query_words)]
    width = len(title_words)
    for window in range(max(1, width - 1), min(len(query_words), width + 2) + 1):
        for start in range(0, len(query_words) - window + 1):
            candidates.append(" ".join(query_words[start: start + window]))

    return max(
        SequenceMatcher(None, title_text, candidate).ratio()
        for candidate in candidates
    )


def _obvious_source_reference(
    query: str,
    source_catalog: list[dict],
    selected_source_keys: list[str],
) -> str | None:
    selected = set(selected_source_keys)
    scored = sorted(
        (
            (_title_match_score(query, item["display_name"]), item["source_key"])
            for item in source_catalog
            if item["source_key"] in selected
        ),
        reverse=True,
    )
    if not scored:
        return None
    best_score, best_key = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    if best_score >= 0.78 and best_score - second_score >= 0.10:
        return best_key
    return None


def _catalog_text(
    source_catalog: list[dict],
    selected_source_keys: list[str],
    index: CorpusIndex,
) -> str:
    selected = set(selected_source_keys)
    blocks = []
    for item in source_catalog:
        key = item["source_key"]
        if key not in selected or key not in index.sources:
            continue
        chunk_count = len(index.sources[key].chunks)
        blocks.append(
            f"- source_key: {key}\n"
            f"  title: {item['display_name']}\n"
            f"  chunk_count: {chunk_count}"
        )
    return "\n\n".join(blocks) if blocks else "(none)"


def _chunk_excerpt(chunk: dict, max_chars: int = 260) -> str:
    text = str(chunk.get("speaker_text") or chunk.get("text") or "").strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return text


def _evidence_text(evidence: dict[str, dict]) -> str:
    if not evidence:
        return "(none)"
    blocks = []
    for citation_id, chunk in evidence.items():
        blocks.append(
            f"[{citation_id}]\n"
            f"Source: {chunk.get('source_display_name', chunk.get('source_key', chunk.get('source_id', 'Source')))}\n"
            f"Timestamp: {float(chunk.get('start', -1)):.1f}s–{float(chunk.get('end', -1)):.1f}s\n"
            f"Transcript: {_chunk_excerpt(chunk)}"
        )
    return "\n\n".join(blocks)


def _prompt(
    query: str,
    source_catalog: list[dict],
    selected_source_keys: list[str],
    index: CorpusIndex,
    evidence: dict[str, dict],
    remaining_rounds: int,
    max_requests_per_round: int,
    previous_steps: list[AgentTraceStep],
) -> str:
    step_text = "\n".join(
        f"- {step.action}: {step.detail}"
        for step in previous_steps[-8:]
    ) or "(none)"
    return (
        f"User question:\n{query}\n\n"
        f"Selected recording catalog:\n{_catalog_text(source_catalog, selected_source_keys, index)}\n\n"
        f"Transcript evidence gathered so far:\n{_evidence_text(evidence)}\n\n"
        f"Retrieval steps already taken:\n{step_text}\n\n"
        f"Evidence chunks gathered: {len(evidence)} / 16\n"
        f"Retrieval rounds remaining: {remaining_rounds}\n"
        f"Maximum retrieval requests in one gather step: {max_requests_per_round}\n\n"
        "Choose the next action. If you choose gather, fill up to three flat tool slots "
        "(tool_1/query_1/source_key_1/etc., then tool_2, tool_3) and set unused tools to none. "
        "If you answer, cite only evidence IDs shown above. If evidence is sufficient, answer now."
    )


def _normalize_selected_sources(
    selected_source_keys: list[str] | None,
    source_catalog: list[dict],
    index: CorpusIndex,
) -> list[str]:
    available = [
        item["source_key"]
        for item in source_catalog
        if item["source_key"] in index.sources
    ]
    if selected_source_keys is None:
        return available
    return [key for key in selected_source_keys if key in index.sources and key in available]


def _add_evidence(
    state: _AgentState,
    chunks: list[dict],
    display_names: dict[str, str],
    max_evidence: int = 16,
) -> int:
    added = 0
    for raw_chunk in chunks:
        if len(state.evidence) >= max_evidence:
            break
        chunk = dict(raw_chunk)
        source_key = str(chunk.get("source_key", ""))
        chunk["source_display_name"] = display_names.get(
            source_key,
            chunk.get("source_id", "Source"),
        )
        citation_id = create_citation_id(chunk)
        if citation_id in state.evidence:
            continue
        state.evidence[citation_id] = chunk
        added += 1
    return added


def _authorized_sources(requested: list[str], selected: list[str]) -> list[str]:
    if not requested:
        return list(selected)
    selected_set = set(selected)
    return [key for key in requested if key in selected_set]


def _execute_search(
    request: RetrievalRequest,
    index: CorpusIndex,
    selected: list[str],
) -> tuple[list[dict], str]:
    source_keys = _authorized_sources(request.source_keys, selected)
    if request.source_keys and not source_keys:
        return [], "blocked: search requested only unselected sources"
    query = request.query.strip()
    if not query:
        return [], "skipped: search request had no query"
    limit = max(1, min(int(request.limit), 6))
    chunks = search_corpus(
        query=query,
        index=index,
        top_k=limit,
        source_keys=source_keys,
        retrieval_mode="global",
        top_k_per_source=min(4, limit),
    )
    return chunks, f'search "{query}" in {", ".join(source_keys)}'


def _execute_read_source(
    request: RetrievalRequest,
    index: CorpusIndex,
    selected: list[str],
) -> tuple[list[dict], str]:
    key = request.source_key
    if key not in selected:
        return [], "blocked: source read requested an unselected source"
    source = index.sources.get(key)
    if source is None:
        return [], "failed: source does not exist"
    offset = max(0, int(request.offset))
    limit = max(1, min(int(request.limit), 6))
    chunks = source.chunks[offset: offset + limit]
    return chunks, f"read {len(chunks)} ordered chunks from {key} starting at offset {offset}"


def _execute_sample_source(
    request: RetrievalRequest,
    index: CorpusIndex,
    selected: list[str],
) -> tuple[list[dict], str]:
    key = request.source_key
    if key not in selected:
        return [], "blocked: source sample requested an unselected source"
    source = index.sources.get(key)
    if source is None:
        return [], "failed: source does not exist"
    chunks = source.chunks
    if not chunks:
        return [], f"sampled 0 chunks from {key}"

    limit = max(2, min(int(request.limit), 8))
    if len(chunks) <= limit:
        sampled = list(chunks)
    else:
        last = len(chunks) - 1
        positions: list[int] = []
        for i in range(limit):
            position = round(i * last / (limit - 1))
            if position not in positions:
                positions.append(position)
        sampled = [chunks[position] for position in positions]
    return sampled, f"sampled {len(sampled)} chunks across all {len(chunks)} chunks in {key}"


def _execute_read_context(
    request: RetrievalRequest,
    index: CorpusIndex,
    selected: list[str],
) -> tuple[list[dict], str]:
    key = request.source_key
    if key not in selected:
        return [], "blocked: context read requested an unselected source"
    source = index.sources.get(key)
    if source is None:
        return [], "failed: source does not exist"
    radius = max(1, min(int(request.radius), 3))
    target_index = next(
        (
            i
            for i, chunk in enumerate(source.chunks)
            if int(chunk.get("chunk_id", -999999)) == int(request.chunk_id)
        ),
        None,
    )
    if target_index is None:
        return [], f"failed: chunk {request.chunk_id} was not found in {key}"
    start = max(0, target_index - radius)
    end = min(len(source.chunks), target_index + radius + 1)
    return source.chunks[start:end], f"read context around chunk {request.chunk_id} from {key}"


def _execute_request(
    request: RetrievalRequest,
    index: CorpusIndex,
    selected: list[str],
) -> tuple[list[dict], str]:
    if request.tool == "search_transcripts":
        return _execute_search(request, index, selected)
    if request.tool == "read_source":
        return _execute_read_source(request, index, selected)
    if request.tool == "sample_source":
        return _execute_sample_source(request, index, selected)
    return _execute_read_context(request, index, selected)


def _clarification_result(question: str, trace: list[AgentTraceStep]) -> AskResult:
    cleaned = question.strip() or "Could you clarify what you mean?"
    return AskResult(
        outcome="clarification",
        answerable=False,
        answer="I don't understand the question well enough to answer it reliably. " + cleaned,
        trace=trace,
    )


def _refusal_result(trace: list[AgentTraceStep]) -> AskResult:
    return AskResult(
        outcome="refusal",
        answerable=False,
        answer=REFUSAL_TEXT,
        trace=trace,
    )


def _error_result(
    trace: list[AgentTraceStep],
    detail: str,
    elapsed_ms: int = 0,
) -> AskResult:
    safe_trace = list(trace)
    safe_trace.append(
        AgentTraceStep(
            iteration=len(safe_trace),
            action="error",
            detail=detail,
            elapsed_ms=elapsed_ms,
        )
    )
    return AskResult(
        outcome="error",
        answerable=False,
        answer=(
            "I couldn't complete the grounded retrieval process reliably. "
            "Please try the question again."
        ),
        trace=safe_trace,
    )


def _answer_result(decision: AgentDecision, state: _AgentState) -> AskResult:
    draft = LLMAnswerDraft(
        answerable=True,
        answer=decision.answer.strip(),
        citation_ids=decision.citation_ids,
    )
    grounded: GroundedAnswer = finalize_answer(
        draft=draft,
        retrieved_chunks=list(state.evidence.values()),
    )
    if not grounded.answerable:
        return _refusal_result(state.trace)

    evidence_lookup = {
        create_citation_id(chunk): chunk
        for chunk in state.evidence.values()
    }
    cited_evidence = [
        evidence_lookup[citation.citation_id]
        for citation in grounded.citations
        if citation.citation_id in evidence_lookup
    ]
    return AskResult(
        outcome="answer",
        answerable=True,
        answer=grounded.answer,
        citations=grounded.citations,
        evidence=cited_evidence,
        trace=state.trace,
    )


def _get_decision(
    *,
    query: str,
    state: _AgentState,
    selected: list[str],
    source_catalog: list[dict],
    index: CorpusIndex,
    llm_client,
    remaining_rounds: int,
    max_requests_per_round: int,
) -> tuple[AgentDecision, int]:
    prompt = _prompt(
        query=query,
        source_catalog=source_catalog,
        selected_source_keys=selected,
        index=index,
        evidence=state.evidence,
        remaining_rounds=remaining_rounds,
        max_requests_per_round=max_requests_per_round,
        previous_steps=state.trace,
    )
    started = time.perf_counter()
    raw = llm_client.generate(
        system_prompt=AGENT_SYSTEM_PROMPT,
        user_prompt=prompt,
        response_schema=_decision_schema(selected, list(state.evidence)),
        max_tokens=560,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    try:
        return _parse_decision(raw), elapsed_ms
    except Exception as exc:
        preview = re.sub(r"\s+", " ", str(raw or "")).strip()[:360]
        raise ControllerParseError(
            f"{type(exc).__name__}; raw={preview!r}",
            elapsed_ms=elapsed_ms,
        ) from exc


def agentic_ask(
    query: str,
    index: CorpusIndex,
    selected_source_keys: list[str] | None,
    source_catalog: list[dict],
    llm_client,
    initial_top_k: int = 4,
    max_tool_calls: int = 3,
    max_retrieval_rounds: int = 2,
) -> AskResult:
    selected = _normalize_selected_sources(selected_source_keys, source_catalog, index)
    if not selected:
        return _refusal_result([])

    display_names = {
        item["source_key"]: item["display_name"]
        for item in source_catalog
        if item["source_key"] in selected
    }
    state = _AgentState(evidence={}, trace=[])

    bootstrap_started = time.perf_counter()
    bootstrap = search_corpus(
        query=query,
        index=index,
        top_k=max(1, min(initial_top_k, 6)),
        source_keys=selected,
        retrieval_mode="global",
        top_k_per_source=3,
    )
    bootstrap_ms = round((time.perf_counter() - bootstrap_started) * 1000)
    added = _add_evidence(state, bootstrap, display_names)
    state.trace.append(
        AgentTraceStep(
            iteration=0,
            action="bootstrap_search",
            detail=f'initial search for "{query}"',
            result_count=added,
            elapsed_ms=bootstrap_ms,
        )
    )

    matched_source = _obvious_source_reference(query, source_catalog, selected)
    if matched_source:
        sample_request = RetrievalRequest(
            tool="sample_source",
            source_key=matched_source,
            limit=8,
        )
        sample_started = time.perf_counter()
        sample_chunks, sample_detail = _execute_sample_source(
            sample_request,
            index,
            selected,
        )
        sample_ms = round((time.perf_counter() - sample_started) * 1000)
        sample_added = _add_evidence(state, sample_chunks, display_names)
        state.trace.append(
            AgentTraceStep(
                iteration=0,
                action="metadata_sample",
                detail=f"obvious title match: {sample_detail}",
                result_count=sample_added,
                elapsed_ms=sample_ms,
            )
        )

    retrieval_rounds_used = 0
    decision_number = 0

    while True:
        remaining_rounds = max(0, max_retrieval_rounds - retrieval_rounds_used)
        if len(state.evidence) >= 16:
            remaining_rounds = 0
        decision_number += 1

        try:
            decision, controller_ms = _get_decision(
                query=query,
                state=state,
                selected=selected,
                source_catalog=source_catalog,
                index=index,
                llm_client=llm_client,
                remaining_rounds=remaining_rounds,
                max_requests_per_round=max_tool_calls,
            )
        except ControllerParseError as exc:
            return _error_result(
                state.trace,
                f"controller parse failure: {exc}",
                elapsed_ms=exc.elapsed_ms,
            )
        except Exception as exc:
            return _error_result(
                state.trace,
                f"controller failure: {type(exc).__name__}: {str(exc)[:240]}",
            )

        state.trace.append(
            AgentTraceStep(
                iteration=decision_number,
                action="controller",
                detail=f"controller chose {decision.action}",
                elapsed_ms=controller_ms,
            )
        )

        if decision.action == "answer":
            state.trace.append(
                AgentTraceStep(
                    iteration=decision_number,
                    action="answer",
                    detail="answered from gathered transcript evidence",
                    result_count=len(decision.citation_ids),
                )
            )
            return _answer_result(decision, state)

        if decision.action == "refuse":
            state.trace.append(
                AgentTraceStep(
                    iteration=decision_number,
                    action="refuse",
                    detail="selected transcript evidence did not support the question",
                )
            )
            return _refusal_result(state.trace)

        if decision.action == "clarify":
            state.trace.append(
                AgentTraceStep(
                    iteration=decision_number,
                    action="clarify",
                    detail="material ambiguity remained after retrieval",
                )
            )
            return _clarification_result(decision.clarification_question, state.trace)

        if remaining_rounds <= 0:
            return _error_result(
                state.trace,
                "controller requested more retrieval after the retrieval budget was exhausted",
            )

        requests = _decision_requests(decision)[:max_tool_calls]
        if not requests:
            return _error_result(
                state.trace,
                "controller chose gather without any retrieval requests",
            )

        retrieval_rounds_used += 1
        total_added = 0
        for request_number, retrieval_request in enumerate(requests, start=1):
            request_started = time.perf_counter()
            chunks, detail = _execute_request(retrieval_request, index, selected)
            request_ms = round((time.perf_counter() - request_started) * 1000)
            added = _add_evidence(state, chunks, display_names)
            total_added += added
            state.trace.append(
                AgentTraceStep(
                    iteration=decision_number,
                    action=retrieval_request.tool,
                    detail=f"batch {retrieval_rounds_used}.{request_number}: {detail}",
                    result_count=added,
                    elapsed_ms=request_ms,
                )
            )

        if total_added == 0 and len(state.evidence) >= 16:
            state.trace.append(
                AgentTraceStep(
                    iteration=decision_number,
                    action="evidence_budget",
                    detail="evidence budget full; further duplicate/source-walk reads are suppressed",
                )
            )
