from __future__ import annotations

from dataclasses import dataclass
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
- Return JSON only.
""".strip()


ToolName = Literal[
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
    requests: list[RetrievalRequest] = Field(default_factory=list)
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


def _decision_schema(
    allowed_source_keys: list[str],
    evidence_ids: list[str],
) -> dict:
    schema = AgentDecision.model_json_schema()
    schema["required"] = ["action"]
    schema["additionalProperties"] = False

    request_schema = schema.get("$defs", {}).get("RetrievalRequest", {})
    if request_schema:
        request_schema["required"] = ["tool"]
        request_schema["additionalProperties"] = False
        request_schema["properties"]["source_keys"]["items"] = {
            "type": "string",
            "enum": allowed_source_keys,
        }
        request_schema["properties"]["source_key"] = {
            "type": "string",
            "enum": ["", *allowed_source_keys],
        }

    schema["properties"]["citation_ids"]["items"] = (
        {"type": "string", "enum": evidence_ids}
        if evidence_ids
        else {"type": "string"}
    )
    return schema


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
    text = str(
        chunk.get("speaker_text")
        or chunk.get("text")
        or ""
    ).strip()
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
        f"Selected recording catalog:\n"
        f"{_catalog_text(source_catalog, selected_source_keys, index)}\n\n"
        f"Transcript evidence gathered so far:\n"
        f"{_evidence_text(evidence)}\n\n"
        f"Retrieval steps already taken:\n{step_text}\n\n"
        f"Evidence chunks gathered: {len(evidence)} / 16\n"
        f"Retrieval rounds remaining: {remaining_rounds}\n"
        f"Maximum retrieval requests in one gather step: {max_requests_per_round}\n\n"
        "Choose the next action. "
        "If you choose gather, put all useful retrieval requests for this round in requests. "
        "If you answer, cite only evidence IDs shown above. "
        "If the question is broad and current evidence is narrow, prefer sample_source or multiple strategic requests in one gather step. "
        "If evidence is sufficient, answer now."
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
    return [
        key
        for key in selected_source_keys
        if key in index.sources and key in available
    ]


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


def _authorized_sources(
    requested: list[str],
    selected: list[str],
) -> list[str]:
    if not requested:
        return list(selected)
    selected_set = set(selected)
    return [
        key
        for key in requested
        if key in selected_set
    ]


def _execute_search(
    request: RetrievalRequest,
    index: CorpusIndex,
    selected: list[str],
) -> tuple[list[dict], str]:
    source_keys = _authorized_sources(
        request.source_keys,
        selected,
    )
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
    return chunks, (
        f'search "{query}" in {", ".join(source_keys)}'
    )


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
    return chunks, (
        f"read {len(chunks)} ordered chunks from {key} "
        f"starting at offset {offset}"
    )


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
        positions = []
        for i in range(limit):
            position = round(i * last / (limit - 1))
            if position not in positions:
                positions.append(position)
        sampled = [chunks[position] for position in positions]

    return sampled, (
        f"sampled {len(sampled)} chunks across all {len(chunks)} chunks in {key}"
    )


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
            index_value
            for index_value, chunk in enumerate(source.chunks)
            if int(chunk.get("chunk_id", -999999)) == int(request.chunk_id)
        ),
        None,
    )
    if target_index is None:
        return [], f"failed: chunk {request.chunk_id} was not found in {key}"
    start = max(0, target_index - radius)
    end = min(len(source.chunks), target_index + radius + 1)
    chunks = source.chunks[start:end]
    return chunks, (
        f"read context around chunk {request.chunk_id} from {key}"
    )


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


def _clarification_result(
    question: str,
    trace: list[AgentTraceStep],
) -> AskResult:
    cleaned = question.strip() or "Could you clarify what you mean?"
    return AskResult(
        outcome="clarification",
        answerable=False,
        answer=(
            "I don't understand the question well enough to answer it reliably. "
            + cleaned
        ),
        trace=trace,
    )


def _refusal_result(
    trace: list[AgentTraceStep],
) -> AskResult:
    return AskResult(
        outcome="refusal",
        answerable=False,
        answer=REFUSAL_TEXT,
        trace=trace,
    )


def _error_result(
    trace: list[AgentTraceStep],
    detail: str,
) -> AskResult:
    safe_trace = list(trace)
    safe_trace.append(
        AgentTraceStep(
            iteration=len(safe_trace),
            action="error",
            detail=detail,
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


def _answer_result(
    decision: AgentDecision,
    state: _AgentState,
) -> AskResult:
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
        response_schema=_decision_schema(
            selected,
            list(state.evidence),
        ),
        max_tokens=420,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    return AgentDecision.model_validate_json(raw), elapsed_ms


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
    selected = _normalize_selected_sources(
        selected_source_keys,
        source_catalog,
        index,
    )
    if not selected:
        return _refusal_result([])

    display_names = {
        item["source_key"]: item["display_name"]
        for item in source_catalog
        if item["source_key"] in selected
    }
    state = _AgentState(
        evidence={},
        trace=[],
    )

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
    added = _add_evidence(
        state,
        bootstrap,
        display_names,
    )
    state.trace.append(
        AgentTraceStep(
            iteration=0,
            action="bootstrap_search",
            detail=f'initial search for "{query}"',
            result_count=added,
            elapsed_ms=bootstrap_ms,
        )
    )

    retrieval_rounds_used = 0
    decision_number = 0

    while True:
        remaining_rounds = max(
            0,
            max_retrieval_rounds - retrieval_rounds_used,
        )
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
        except Exception as exc:
            return _error_result(
                state.trace,
                f"controller failure: {type(exc).__name__}",
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
            return _clarification_result(
                decision.clarification_question,
                state.trace,
            )

        if remaining_rounds <= 0:
            return _error_result(
                state.trace,
                "controller requested more retrieval after the retrieval budget was exhausted",
            )

        requests = decision.requests[:max_tool_calls]
        if not requests:
            return _error_result(
                state.trace,
                "controller chose gather without any retrieval requests",
            )

        retrieval_rounds_used += 1
        total_added = 0

        for request_number, retrieval_request in enumerate(requests, start=1):
            request_started = time.perf_counter()
            chunks, detail = _execute_request(
                retrieval_request,
                index,
                selected,
            )
            request_ms = round((time.perf_counter() - request_started) * 1000)
            added = _add_evidence(
                state,
                chunks,
                display_names,
            )
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
                    result_count=0,
                )
            )
