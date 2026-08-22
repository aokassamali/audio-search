from pydantic import BaseModel, Field
from typing import Protocol

from pydantic import ValidationError


SYSTEM_PROMPT = """
You are the grounded answer-synthesis stage of a question-answering system over audio transcripts.

The retrieval planner has already interpreted the user's wording and resolved ordinary typos, shorthand, source references, and conversational phrasing. You receive that planner-resolved question below. Do not reinterpret the user's spelling or invent alternate meanings for it. If the question was genuinely too ambiguous to interpret reliably, the system handles clarification before this stage.

Grounding rules:
1. Answer the planner-resolved question using only the supplied evidence.
2. State only factual claims supported by the supplied evidence. Do not use outside knowledge.
3. Every factual claim in the answer must be supported by at least one citation_id.
4. Only cite citation_ids that appear in the supplied evidence.
5. Synthesize across multiple evidence chunks and make ordinary inferences when the cited evidence jointly supports them.
6. For comparison, relationship, summary, argument, or "what is going on" questions, answer at the level supported by the evidence. The transcript does not need to contain a verbatim sentence naming the requested relationship if the relationship can be directly summarized from supported comparisons or descriptions in the evidence.
7. Distinguish a speaker's own position from questions, hypotheticals, and descriptions of another person's position. If attribution is uncertain, describe the point neutrally.
8. Refuse only when the supplied evidence is genuinely insufficient to give a meaningful answer to the planner-resolved question.
9. When refusing, briefly explain what evidence is missing or unsupported and use an empty citation_ids list.
10. Return JSON only, with no Markdown or additional commentary.

Grounding constrains what facts you may state; it should not force you to demand exact wording that the evidence already supports semantically.

Return exactly this structure:
{
  "answerable": true or false,
  "answer": "your answer or a brief refusal",
  "citation_ids": ["source_id:chunk_id"]
}
""".strip()


class LLMClient(Protocol):
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict,
        max_tokens: int = 512,
    ) -> str:
        ...


class LLMAnswerDraft(BaseModel):
    answerable: bool
    answer: str
    citation_ids: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    citation_id: str
    source_id: str
    chunk_id: int
    start: float
    end: float


class GroundedAnswer(BaseModel):
    answerable: bool
    answer: str
    citations: list[Citation] = Field(default_factory=list)


def create_citation_id(chunk: dict) -> str:
    return f"{chunk['source_id']}:{chunk['chunk_id']}"


def build_context(
    chunks: list[dict],
) -> str:
    context_blocks = []

    for chunk in chunks:
        citation_id = create_citation_id(chunk)
        source_name = chunk.get(
            "source_display_name",
            chunk.get("source_key", chunk.get("source_id", "Source")),
        )

        context_text = chunk.get(
            "speaker_text",
            chunk["text"],
        )

        context_block = (
            f"[{citation_id}]\n"
            f"Source: {source_name}\n"
            f"Timestamp: "
            f"{chunk['start']:.1f}s–"
            f"{chunk['end']:.1f}s\n"
            f"Transcript:\n{context_text}"
        )

        context_blocks.append(context_block)

    return "\n\n".join(context_blocks)


def finalize_answer(
    draft: LLMAnswerDraft,
    retrieved_chunks: list[dict],
) -> GroundedAnswer:
    chunk_lookup = {
        create_citation_id(chunk): chunk
        for chunk in retrieved_chunks
    }

    if not draft.answerable:
        refusal = draft.answer.strip()

        if not refusal:
            refusal = "I don't find this discussed in the audio."

        return GroundedAnswer(
            answerable=False,
            answer=refusal,
        )

    invalid_citation_ids = [
        citation_id
        for citation_id in draft.citation_ids
        if citation_id not in chunk_lookup
    ]

    if not draft.citation_ids or invalid_citation_ids:
        return GroundedAnswer(
            answerable=False,
            answer=(
                "I couldn't verify an answer from the "
                "retrieved audio evidence."
            ),
        )

    citations = []

    for citation_id in draft.citation_ids:
        chunk = chunk_lookup[citation_id]

        citations.append(
            Citation(
                citation_id=citation_id,
                source_id=chunk["source_id"],
                chunk_id=chunk["chunk_id"],
                start=chunk["start"],
                end=chunk["end"],
            )
        )

    return GroundedAnswer(
        answerable=True,
        answer=draft.answer,
        citations=citations,
    )


def build_prompt(
    query: str,
    retrieved_chunks: list[dict],
) -> str:
    context = build_context(retrieved_chunks)
    interpreted = next(
        (
            str(chunk.get("retrieval_interpretation", "")).strip()
            for chunk in retrieved_chunks
            if str(chunk.get("retrieval_interpretation", "")).strip()
        ),
        query.strip(),
    )

    return (
        f"Planner-resolved question:\n{interpreted}\n\n"
        "The question above represents the user's intended information need. "
        "Do not require any misspelled or shorthand form from the original wording to appear in the transcript.\n\n"
        f"Audio evidence:\n{context}\n\n"
        "Answer the planner-resolved question using only this evidence and return the required JSON."
    )


def parse_llm_answer(
    raw_response: str,
) -> LLMAnswerDraft:
    try:
        return LLMAnswerDraft.model_validate_json(
            raw_response
        )
    except ValidationError:
        return LLMAnswerDraft(
            answerable=False,
            answer=(
                "The language model returned an "
                "invalid response."
            ),
            citation_ids=[],
        )


def answer_question(
    query: str,
    retrieved_chunks: list[dict],
    llm_client: LLMClient,
) -> GroundedAnswer:
    user_prompt = build_prompt(
        query=query,
        retrieved_chunks=retrieved_chunks,
    )

    response_schema = build_answer_schema(
        retrieved_chunks
    )

    raw_response = llm_client.generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=response_schema,
    )

    draft = parse_llm_answer(raw_response)

    return finalize_answer(
        draft=draft,
        retrieved_chunks=retrieved_chunks,
    )


def build_answer_schema(
    retrieved_chunks: list[dict],
) -> dict:
    allowed_citation_ids = [
        create_citation_id(chunk)
        for chunk in retrieved_chunks
    ]

    schema = LLMAnswerDraft.model_json_schema()

    schema["required"] = [
        "answerable",
        "answer",
        "citation_ids",
    ]

    schema["additionalProperties"] = False

    schema["properties"]["citation_ids"]["items"] = {
        "type": "string",
        "enum": allowed_citation_ids,
    }

    return schema
