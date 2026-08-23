from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

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

You do not know the recordings from model memory. The only facts you may use in an answer are facts present in transcript evidence returned by the retrieval tools in this session. Source titles and metadata may be used to decide where to search, but they are not factual evidence for the answer.

You receive:
- the user's natural-language question,
- the selected recording catalog,
- transcript evidence gathered so far,
- the number of retrieval tool calls remaining.

Available actions:
1. search_transcripts
   Search transcript chunks semantically and lexically. Use this for targeted questions, names, topics, comparisons, paraphrases, and reformulations.
2. read_source
   Read an ordered slice of one recording's transcript by chunk offset. Use this when current search results are too narrow, when chronology matters, or when the user asks for an explanation/synthesis that needs broader coverage.
3. read_context
   Read chunks immediately around one known chunk. Use this when a retrieved passage needs neighboring context.
4. answer
   Answer using only gathered transcript evidence. Every factual claim must be supported by at least one citation_id from the evidence.
5. refuse
   Choose this when reasonable retrieval has not found evidence that supports the user's question.
6. clarify
   Choose this only when the user's meaning remains materially ambiguous after using the selected source catalog and retrieval tools.

Behavior rules:
- Do not classify the request into a fixed intent taxonomy. Decide directly what information you need and which retrieval action will obtain it.
- Resolve obvious typos, shorthand, and approximate recording-title references from the selected source catalog. Do not ask for clarification when one interpretation is clearly the best match.
- If a question is understandable but unsupported by the selected recordings, retrieve reasonably and then refuse. Do not invent a different interpretation just to fit the corpus.
- Clarification is a last resort. Use it only when multiple materially different interpretations remain plausible and choosing among them would change the answer.
- Never use world knowledge as evidence. Never propose world-knowledge meanings for an unclear term.
- Stay inside the selected source boundary. Never request or cite an unselected source.
- Do not answer broad questions from a narrow handful of passages if more source coverage is needed; retrieve more first.
- Prefer a small number of purposeful retrieval actions. If the evidence is already sufficient, answer without another tool call.
- When no retrieval tool calls remain, choose answer, refuse, or clarify.
- Return JSON only.
""".strip()


ActionName = Literal[
    "search_transcripts",
    "read_source",
    "read_context",
    "answer",
    "refuse",
    "clarify",
]


class AgentDecision(BaseModel):
    action: ActionName
    query: str = ""
    source_keys: list[str] = Field(default_factory=list)
    source_key: str = ""
    offset: int = 0
    limit: int = 5
    chunk_id: int = -1
    radius: int = 2
    answer: str = ""
    citation_ids: list[str] = Field(default_factory=list)
    clarification_question: str = ""


class AgentTraceStep(BaseModel):
    iteration: int
    action: str
    detail: str
    result_count: int = 0


class AskResult(BaseModel):
    outcome: Literal["answer", "refusal", "clarification"]
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
    schema["required"] = list(schema["properties"])
    schema["additionalProperties"] = False
    schema["properties"]["source_keys"]["items"] = {
        "type": "string",
        "enum": allowed_source_keys,
    }
    schema["properties"]["source_key"] = {
        "type": "string",
        "enum": ["", *allowed_source_keys],
    }
    schema["properties"]["citation_ids"]["items"] = {
        "type": "string",
        "enum": evidence_ids,
    }
    return schema


def _catalog_text(
    source_catalog: list[dict],
    selected_source_keys: list[str],
    index: CorpusIndex,
) -> str:
    selected = set(selected_source_keys)
    lines = []
    for item in source_catalog:
        key = item["source_key"]
        if key not in selected or key not in index.sources:
            continue
        chunk_count = len(index.sources[key].chunks)
        lines.append(
            f'- source_key: {key}\n'
            f'  title: {item["display_name"]}\n'
            f'  chunk_count: {chunk_count}'
        )
    return "\n\n".join(lines) if lines else "(none)"


def _chunk_excerpt(chunk: dict, max_chars: int = 340) -> str:
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
    remaining_tool_calls: int,
    previous_steps: list[AgentTraceStep],
) -> str:
    step_text = "\n".join(
        f"- {step.action}: {step.detail}"
        for step in previous_steps[-6:]
    ) or "(none)"
    return (
        f"User question:\n{query}\n\n"
        f"Selected recording catalog:\n"
        f"{_catalog_text(source_catalog, selected_source_keys, index)}\n\n"
        f"Transcript evidence gathered so far:\n"
        f"{_evidence_text(evidence)}\n\n"
        f"Retrieval steps already taken:\n{step_text}\n\n"
        f"Retrieval tool calls remaining: {remaining_tool_calls}\n\n"
        "Choose the next action. If you answer, cite only evidence IDs shown above. "
        "If evidence is insufficient but another retrieval action could reasonably help, use the tool rather than refusing."
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
    max_evidence: int = 18,
) -> int:
    added = 0
    for raw_chunk in chunks:
        chunk = dict(raw_chunk)
        source_key = str(chunk.get("source_key", ""))
        chunk["source_display_name"] = display_names.get(
            source_key,
            chunk.get("source_id", "Source"),
        )
        citation_id = create_citation_id(chunk)
        if citation_id in state.evidence:
            continue
        if len(state.evidence) >= max_evidence:
            break
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
    decision: AgentDecision,
    index: CorpusIndex,
    selected: list[str],
) -> tuple[list[dict], str]:
    source_keys = _authorized_sources(
        decision.source_keys,
        selected,
    )
    if decision.source_keys and not source_keys:
        return [], "Search was blocked because it requested only unselected sources."
    query = decision.query.strip()
    if not query:
        return [], "Search was skipped because no retrieval query was supplied."
    limit = max(1, min(int(decision.limit), 8))
    chunks = search_corpus(
        query=query,
        index=index,
        top_k=limit,
        source_keys=source_keys,
        retrieval_mode="global",
        top_k_per_source=min(4, limit),
    )
    return chunks, (
        f'searched "{query}" in '
        f'{", ".join(source_keys)}'
    )


def _execute_read_source(
    decision: AgentDecision,
    index: CorpusIndex,
    selected: list[str],
) -> tuple[list[dict], str]:
    key = decision.source_key
    if key not in selected:
        return [], "Source read was blocked because the source is not selected."
    source = index.sources.get(key)
    if source is None:
        return [], "Source read failed because the source does not exist."
    offset = max(0, int(decision.offset))
    limit = max(1, min(int(decision.limit), 8))
    chunks = source.chunks[offset: offset + limit]
    return chunks, (
        f"read {len(chunks)} ordered chunks from {key} "
        f"starting at offset {offset}"
    )


def _execute_read_context(
    decision: AgentDecision,
    index: CorpusIndex,
    selected: list[str],
) -> tuple[list[dict], str]:
    key = decision.source_key
    if key not in selected:
        return [], "Context read was blocked because the source is not selected."
    source = index.sources.get(key)
    if source is None:
        return [], "Context read failed because the source does not exist."
    radius = max(1, min(int(decision.radius), 4))
    target_index = next(
        (
            index_value
            for index_value, chunk in enumerate(source.chunks)
            if int(chunk.get("chunk_id", -999999)) == int(decision.chunk_id)
        ),
        None,
    )
    if target_index is None:
        return [], f"Chunk {decision.chunk_id} was not found in {key}."
    start = max(0, target_index - radius)
    end = min(len(source.chunks), target_index + radius + 1)
    chunks = source.chunks[start:end]
    return chunks, (
        f"read context around chunk {decision.chunk_id} "
        f"from {key}"
    )


def _clarification_result(
    question: str,
    trace: list[AgentTraceStep],
) -> AskResult:
    cleaned = question.strip() or "Could you clarify what you mean?"
    message = (
        "I don't understand the question well enough to answer it reliably. "
        + cleaned
    )
    return AskResult(
        outcome="clarification",
        answerable=False,
        answer=message,
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


def agentic_ask(
    query: str,
    index: CorpusIndex,
    selected_source_keys: list[str] | None,
    source_catalog: list[dict],
    llm_client,
    initial_top_k: int = 6,
    max_tool_calls: int = 3,
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

    bootstrap = search_corpus(
        query=query,
        index=index,
        top_k=max(1, min(initial_top_k, 8)),
        source_keys=selected,
        retrieval_mode="global",
        top_k_per_source=3,
    )
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
        )
    )

    tool_calls_used = 0
    # One final decision is allowed after the retrieval budget has been spent.
    for iteration in range(1, max_tool_calls + 2):
        remaining = max(0, max_tool_calls - tool_calls_used)
        prompt = _prompt(
            query=query,
            source_catalog=source_catalog,
            selected_source_keys=selected,
            index=index,
            evidence=state.evidence,
            remaining_tool_calls=remaining,
            previous_steps=state.trace,
        )
        try:
            raw = llm_client.generate(
                system_prompt=AGENT_SYSTEM_PROMPT,
                user_prompt=prompt,
                response_schema=_decision_schema(
                    selected,
                    list(state.evidence),
                ),
                max_tokens=512,
            )
            decision = AgentDecision.model_validate_json(raw)
        except (ValidationError, ValueError, TypeError):
            return _refusal_result(state.trace)

        if decision.action == "answer":
            state.trace.append(
                AgentTraceStep(
                    iteration=iteration,
                    action="answer",
                    detail="answered from gathered transcript evidence",
                    result_count=len(decision.citation_ids),
                )
            )
            return _answer_result(decision, state)

        if decision.action == "refuse":
            state.trace.append(
                AgentTraceStep(
                    iteration=iteration,
                    action="refuse",
                    detail="model determined the selected evidence does not support the question",
                )
            )
            return _refusal_result(state.trace)

        if decision.action == "clarify":
            state.trace.append(
                AgentTraceStep(
                    iteration=iteration,
                    action="clarify",
                    detail="material ambiguity remained after retrieval",
                )
            )
            return _clarification_result(
                decision.clarification_question,
                state.trace,
            )

        if remaining <= 0:
            state.trace.append(
                AgentTraceStep(
                    iteration=iteration,
                    action="refuse",
                    detail="retrieval budget exhausted without a grounded final answer",
                )
            )
            return _refusal_result(state.trace)

        if decision.action == "search_transcripts":
            chunks, detail = _execute_search(
                decision,
                index,
                selected,
            )
        elif decision.action == "read_source":
            chunks, detail = _execute_read_source(
                decision,
                index,
                selected,
            )
        else:
            chunks, detail = _execute_read_context(
                decision,
                index,
                selected,
            )

        added = _add_evidence(
            state,
            chunks,
            display_names,
        )
        tool_calls_used += 1
        state.trace.append(
            AgentTraceStep(
                iteration=iteration,
                action=decision.action,
                detail=detail,
                result_count=added,
            )
        )

    return _refusal_result(state.trace)
