from typing import Literal

from pydantic import BaseModel, Field


RegisterLabel = Literal[
    "assertion",
    "hypothetical",
    "question",
    "characterization",
    "hyperbole",
    "joke",
]

Difficulty = Literal[
    "easy",
    "hard",
]


class RegisterGoldItem(BaseModel):
    turn_id: str
    source_id: str
    turn_index: int

    speaker: str
    speaker_label: str

    start: float
    end: float
    duration: float

    text: str
    word_count: int

    label_permissive: RegisterLabel | None = None
    label_strict: RegisterLabel | None = None
    difficulty: Difficulty | None = None
    notes: str = ""


class SamplingMetadata(BaseModel):
    method: str
    seed: int
    requested_count: int
    minimum_words: int


class RegisterGoldArtifact(BaseModel):
    schema_version: str = "1.0"
    source_id: str
    taxonomy_version: str = "register_v1"

    sampling: SamplingMetadata
    items: list[RegisterGoldItem] = Field(
        default_factory=list
    )

    limitation: str = (
        "Single annotator; no inter-annotator "
        "agreement. Intra-annotator agreement "
        "is unavailable because repeated "
        "annotation would involve strong "
        "recognition memory."
    )