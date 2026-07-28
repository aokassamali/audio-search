import json
from typing import Literal

from pydantic import BaseModel, Field

from src.rag import LLMClient
from src.register_gold import RegisterLabel


TaxonomyVariant = Literal[
    "permissive",
    "strict",
]


class RegisterPrediction(BaseModel):
    turn_id: str
    label: RegisterLabel
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


REGISTER_SYSTEM_PROMPT = """
Classify the dominant speech act of one speaker turn.

Use exactly one label:

- assertion: the speaker states their own position as held
- hypothetical: a proposition is entertained but not asserted
- question: the main function is seeking information or testing a response
- characterization: the speaker primarily describes another person's,
  institution's, or prior decision's position
- hyperbole: a clear overstatement not intended literally
- joke: clearly humorous or non-serious

Judge the dominant function of the whole turn, not merely its final sentence.

A turn can contain several forms. For example:
- a question may contain asserted premises
- an assertion may cite or discuss another source
- an example supporting the speaker's actual position is usually an assertion
- use characterization only when reporting another position is the primary act

Do not infer from the speaker's job or identity. Use only the supplied text.

Return JSON only.
""".strip()


def build_register_schema(
    turn_id: str,
) -> dict:
    return {
        "type": "object",
        "properties": {
            "turn_id": {
                "type": "string",
                "const": turn_id,
            },
            "label": {
                "type": "string",
                "enum": [
                    "assertion",
                    "hypothetical",
                    "question",
                    "characterization",
                    "hyperbole",
                    "joke",
                ],
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
        "required": [
            "turn_id",
            "label",
            "confidence",
        ],
        "additionalProperties": False,
    }


def classify_text_only(
    turn_id: str,
    text: str,
    taxonomy_variant: TaxonomyVariant,
    llm_client: LLMClient,
) -> RegisterPrediction:
    if taxonomy_variant == "strict":
        hypothetical_rule = (
    "Label hypothetical only when the turn "
    "contains an explicit marker such as "
    "suppose, assume, imagine, what if, or "
    "let's say. An explicit marker is necessary "
    "but not sufficient: if the hypothetical "
    "mainly sets up a substantive question, "
    "label the dominant speech act as question."
    )
    else:
        hypothetical_rule = (
            "Label hypothetical whenever the "
            "speaker entertains a proposition "
            "without asserting it, even when no "
            "explicit counterfactual marker appears."
        )

    user_prompt = (
        f"Taxonomy variant: {taxonomy_variant}\n"
        f"{hypothetical_rule}\n\n"
        f"Turn ID: {turn_id}\n"
        f"Turn text:\n{text}"
    )

    raw_response = llm_client.generate(
        system_prompt=REGISTER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=build_register_schema(
            turn_id
        ),
        max_tokens=128,
    )

    return RegisterPrediction.model_validate(
        json.loads(raw_response)
    )