from pydantic import BaseModel, Field
from typing import Protocol

from pydantic import ValidationError


class LLMClient(Protocol):
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema:dict,
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

        context_text = chunk.get(
            "speaker_text",
            chunk["text"],
        )

        context_block = (
            f"[{citation_id}]\n"
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

    return (
        f"Question:\n{query}\n\n"
        f"Audio evidence:\n{context}\n\n"
        "Produce the required JSON response."
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


SYSTEM_PROMPT = """
You answer questions about an audio recording using only the supplied evidence.

Rules:
1. Use only facts explicitly supported by the evidence.
2. Do not use outside knowledge, even when you know the answer.
3. Every factual claim in the answer must be supported by at least one citation_id.
4. Only cite citation_ids that appear in the supplied evidence.
5. If the evidence does not answer the question, set answerable to false.
6. When answerable is false, use an empty citation_ids list.
7. Return JSON only, with no Markdown or additional commentary.
8. Do not attribute a claim to a person or party merely because another
speaker describes that person's position. If attribution is uncertain,
describe the disagreement neutrally.
9.When answerable is false, briefly explain whether:
- the topic is absent from the evidence, or
- the question contains a premise that the evidence does not support.
10. Distinguish between a speaker's own position, a question, a hypothetical,
and their description of another speaker's position. Do not describe a
question or hypothetical as that speaker's argument unless the evidence
clearly supports that interpretation.

Do not answer using outside knowledge.

Return exactly this structure:
{
  "answerable": true or false,
  "answer": "your answer or a brief refusal",
  "citation_ids": ["source_id:chunk_id"]
}
""".strip()


