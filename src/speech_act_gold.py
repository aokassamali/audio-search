from pydantic import BaseModel, Field

from src.register_gold import RegisterGoldItem


class SpeechActGoldItem(RegisterGoldItem):
    source_key: str
    frozen: bool = False


class SourceSamplingRecord(BaseModel):
    source_key: str
    source_id: str

    requested_count: int
    eligible_count: int
    excluded_existing_count: int


class SpeechActSamplingMetadata(BaseModel):
    method: str
    seed: int
    minimum_words: int
    target_count: int

    source_quotas: dict[str, int]

    source_records: list[
        SourceSamplingRecord
    ] = Field(default_factory=list)


class SpeechActGoldArtifact(BaseModel):
    schema_version: str = "2.0"
    taxonomy_version: str = (
        "speech_act_sincerity_v1"
    )

    sampling: SpeechActSamplingMetadata

    items: list[SpeechActGoldItem] = Field(
        default_factory=list
    )

    limitation: str = (
        "Single annotator. No inter-annotator "
        "agreement. The original 80 items are "
        "frozen from v2.3."
    )