import json
from collections import defaultdict
from pathlib import Path

from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
)

from src.rag import LLMClient

from src.chunk import build_speaker_turns

SPEAKER_ROLE_SYSTEM_PROMPT = """
You infer conversational roles and identities from samples of a recorded
conversation.

Use only contextual evidence contained in the supplied target turns and their
neighboring turns. Do not rely on outside knowledge about the recording.

For every anonymous speaker, infer two separate fields:

1. role:
   The speaker's function in the conversation. Return a concise label grounded
   in the dialogue. Use "unknown" when the evidence is insufficient.

2. identity:
   The speaker's proper name or another explicitly established unique identity.
   Use "unknown" unless the supplied dialogue clearly establishes it.

Do not guess a proper name from subject matter, speaking style, or outside
knowledge.

A name appearing in a neighboring turn may establish the target speaker's
identity only when the conversational context clearly shows that the name is
being used to introduce or address that target speaker.

Confidence must be one of:
- high: directly or very strongly established
- medium: supported but requires meaningful inference
- low: weak, ambiguous, or unknown

For every non-unknown role, cite one or more role evidence sample IDs.
For every non-unknown identity, cite one or more identity evidence sample IDs.

Only use supplied sample IDs.

Return JSON only.
""".strip()


class ContextTurn(BaseModel):
    speaker: str
    start: float
    end: float
    text: str


SampleType = Literal[
    "first_substantive",
    "longest",
]


class SpeakerSample(BaseModel):
    sample_id: str
    speaker: str
    sample_type: SampleType
    start: float
    end: float
    text: str
    word_count: int
    previous_turn: ContextTurn | None = None
    next_turn: ContextTurn | None = None


RoleConfidence = Literal[
    "high",
    "medium",
    "low",
]


class SpeakerRoleDraft(BaseModel):
    speaker: str

    role: str
    role_confidence: RoleConfidence

    role_evidence_sample_ids: list[str] = Field(
        default_factory=list
    )

    identity: str
    identity_confidence: RoleConfidence

    identity_evidence_sample_ids: list[str] = Field(
        default_factory=list
    )


LabelSource = Literal[
    "manual_override",
    "inferred_identity",
    "inferred_role",
    "raw_speaker_id",
]


class SpeakerRoleEvidence(BaseModel):
    sample_id: str
    sample_type: SampleType
    start: float
    end: float
    text: str
    previous_turn: ContextTurn | None = None
    next_turn: ContextTurn | None = None


class SpeakerRoleResult(BaseModel):
    speaker: str

    inferred_role: str
    role_confidence: RoleConfidence
    role_evidence: list[
        SpeakerRoleEvidence
    ] = Field(default_factory=list)

    inferred_identity: str
    identity_confidence: RoleConfidence
    identity_evidence: list[
        SpeakerRoleEvidence
    ] = Field(default_factory=list)

    effective_label: str
    label_source: LabelSource


class SpeakerRolesArtifact(BaseModel):
    source_id: str
    roles: dict[str, SpeakerRoleResult]


class SpeakerRolesDraft(BaseModel):
    speakers: list[SpeakerRoleDraft]

def select_speaker_samples(
    segments: list[dict],
    samples_per_speaker: int = 4,
    minimum_words: int = 8,
) -> dict[str, list[SpeakerSample]]:
    speaker_turns = build_speaker_turns(
        segments
    )

    eligible_indices = defaultdict(list)

    for turn_index, turn in enumerate(
        speaker_turns
    ):
        word_count = len(
            turn["text"].split()
        )

        if word_count < minimum_words:
            continue

        eligible_indices[
            turn["speaker"]
        ].append(turn_index)

    samples_by_speaker = {}

    for speaker, turn_indices in (
        eligible_indices.items()
    ):
        first_turn_index = turn_indices[0]

        longest_indices = sorted(
            turn_indices,
            key=lambda index: len(
                speaker_turns[index][
                    "text"
                ].split()
            ),
            reverse=True,
        )

        selected_turns = [
            (
                first_turn_index,
                "first_substantive",
            )
        ]

        for turn_index in longest_indices:
            if turn_index == first_turn_index:
                continue

            selected_turns.append(
                (
                    turn_index,
                    "longest",
                )
            )

            if (
                len(selected_turns)
                >= samples_per_speaker
            ):
                break

        samples = []

        for sample_number, (
            turn_index,
            sample_type,
        ) in enumerate(
            selected_turns,
            start=1,
        ):
            turn = speaker_turns[turn_index]

            previous_turn = None
            next_turn = None

            if turn_index > 0:
                previous = speaker_turns[
                    turn_index - 1
                ]

                previous_turn = ContextTurn(
                    speaker=previous["speaker"],
                    start=previous["start"],
                    end=previous["end"],
                    text=previous["text"],
                )

            if (
                turn_index
                < len(speaker_turns) - 1
            ):
                following = speaker_turns[
                    turn_index + 1
                ]

                next_turn = ContextTurn(
                    speaker=following["speaker"],
                    start=following["start"],
                    end=following["end"],
                    text=following["text"],
                )

            samples.append(
                SpeakerSample(
                    sample_id=(
                        f"{speaker}:sample_"
                        f"{sample_number}"
                    ),
                    speaker=speaker,
                    sample_type=sample_type,
                    start=turn["start"],
                    end=turn["end"],
                    text=turn["text"],
                    word_count=len(
                        turn["text"].split()
                    ),
                    previous_turn=previous_turn,
                    next_turn=next_turn,
                )
            )

        samples_by_speaker[speaker] = samples

    return samples_by_speaker


def truncate_text(
    text: str,
    max_words: int,
) -> str:
    words = text.split()
    displayed_text = " ".join(
        words[:max_words]
    )

    if len(words) > max_words:
        displayed_text += " ..."

    return displayed_text


def load_speaker_samples(
    transcript_path: str | Path,
    samples_per_speaker: int = 4,
    minimum_words: int = 8,
) -> dict[str, list[SpeakerSample]]:
    transcript_path = Path(
        transcript_path
    )

    with transcript_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        segments = json.load(file)

    return select_speaker_samples(
        segments=segments,
        samples_per_speaker=(
            samples_per_speaker
        ),
        minimum_words=minimum_words,
    )


def build_speaker_role_prompt(
    samples_by_speaker: dict[
        str,
        list[SpeakerSample],
    ],
    max_words_per_sample: int = 80,
    max_context_words: int = 30,
) -> str:
    sections = [
        (
            "Infer the conversational role and, "
            "when explicitly established, the "
            "identity of every anonymous speaker."
        )
    ]

    for speaker, samples in (
        samples_by_speaker.items()
    ):
        sections.append(
            f"\nSpeaker: {speaker}"
        )

        for sample in samples:
            sections.append(
                (
                    f"\n[{sample.sample_id}] "
                    f"type={sample.sample_type}"
                )
            )

            if sample.previous_turn:
                previous = sample.previous_turn

                previous_text = truncate_text(
                    previous.text,
                    max_context_words,
                )

                sections.append(
                    (
                        "Previous turn — "
                        f"{previous.speaker}: "
                        f"{previous_text}"
                    )
                )

            target_text = truncate_text(
                sample.text,
                max_words_per_sample,
            )

            sections.append(
                (
                    f"Target turn — "
                    f"{speaker}: "
                    f"{target_text}"
                )
            )

            if sample.next_turn:
                following = sample.next_turn

                following_text = truncate_text(
                    following.text,
                    max_context_words,
                )

                sections.append(
                    (
                        "Next turn — "
                        f"{following.speaker}: "
                        f"{following_text}"
                    )
                )

    return "\n".join(sections)


def build_speaker_role_schema(
    samples_by_speaker: dict[
        str,
        list[SpeakerSample],
    ],
) -> dict:
    speaker_ids = list(
        samples_by_speaker
    )

    valid_sample_ids = [
        sample.sample_id
        for samples in samples_by_speaker.values()
        for sample in samples
    ]

    confidence_schema = {
        "type": "string",
        "enum": [
            "high",
            "medium",
            "low",
        ],
    }

    evidence_schema = {
        "type": "array",
        "items": {
            "type": "string",
            "enum": valid_sample_ids,
        },
        "uniqueItems": True,
    }

    return {
        "type": "object",
        "properties": {
            "speakers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {
                            "type": "string",
                            "enum": speaker_ids,
                        },
                        "role": {
                            "type": "string",
                        },
                        "role_confidence": (
                            confidence_schema
                        ),
                        "role_evidence_sample_ids": (
                            evidence_schema
                        ),
                        "identity": {
                            "type": "string",
                        },
                        "identity_confidence": (
                            confidence_schema
                        ),
                        "identity_evidence_sample_ids": (
                            evidence_schema
                        ),
                    },
                    "required": [
                        "speaker",
                        "role",
                        "role_confidence",
                        "role_evidence_sample_ids",
                        "identity",
                        "identity_confidence",
                        "identity_evidence_sample_ids",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["speakers"],
        "additionalProperties": False,
    }

def parse_speaker_roles(
    raw_response: str,
) -> SpeakerRolesDraft:
    try:
        response_data = json.loads(
            raw_response
        )

        return SpeakerRolesDraft.model_validate(
            response_data
        )

    except (
        json.JSONDecodeError,
        ValidationError,
    ):
        return SpeakerRolesDraft(
            speakers=[]
        )


def resolve_role_evidence(
    evidence_sample_ids: list[str],
    speaker: str,
    sample_lookup: dict[
        str,
        SpeakerSample,
    ],
) -> list[SpeakerRoleEvidence]:
    evidence = []
    seen_sample_ids = set()

    for sample_id in evidence_sample_ids:
        if sample_id in seen_sample_ids:
            continue

        sample = sample_lookup.get(
            sample_id
        )

        if sample is None:
            continue

        if sample.speaker != speaker:
            continue

        seen_sample_ids.add(sample_id)

        evidence.append(
            SpeakerRoleEvidence(
                sample_id=sample.sample_id,
                sample_type=sample.sample_type,
                start=sample.start,
                end=sample.end,
                text=sample.text,
                previous_turn=(
                    sample.previous_turn
                ),
                next_turn=sample.next_turn,
            )
        )

    return evidence


def finalize_speaker_roles(
    draft: SpeakerRolesDraft,
    samples_by_speaker: dict[
        str,
        list[SpeakerSample],
    ],
    source_id: str,
    manual_labels: dict[str, str] | None = None,
) -> SpeakerRolesArtifact:
    manual_labels = manual_labels or {}

    sample_lookup = {
        sample.sample_id: sample
        for samples
        in samples_by_speaker.values()
        for sample in samples
    }

    draft_roles = {}

    for draft_role in draft.speakers:
        speaker = draft_role.speaker

        if speaker not in samples_by_speaker:
            continue

        if speaker in draft_roles:
            continue

        draft_roles[speaker] = draft_role

    finalized_roles = {}

    for speaker in samples_by_speaker:
        draft_role = draft_roles.get(
            speaker
        )

        if draft_role is None:
            inferred_role = "unknown"
            role_confidence: RoleConfidence = (
                "low"
            )
            role_evidence = []

            inferred_identity = "unknown"
            identity_confidence: RoleConfidence = (
                "low"
            )
            identity_evidence = []

        else:
            inferred_role = (
                draft_role.role.strip()
                or "unknown"
            )

            role_confidence = (
                draft_role.role_confidence
            )

            role_evidence = resolve_role_evidence(
                evidence_sample_ids=(
                    draft_role
                    .role_evidence_sample_ids
                ),
                speaker=speaker,
                sample_lookup=sample_lookup,
            )

            if (
                inferred_role.lower() != "unknown"
                and not role_evidence
            ):
                inferred_role = "unknown"
                role_confidence = "low"

            inferred_identity = (
                draft_role.identity.strip()
                or "unknown"
            )

            identity_confidence = (
                draft_role.identity_confidence
            )

            identity_evidence = (
                resolve_role_evidence(
                    evidence_sample_ids=(
                        draft_role
                        .identity_evidence_sample_ids
                    ),
                    speaker=speaker,
                    sample_lookup=sample_lookup,
                )
            )

            if (
                inferred_identity.lower()
                != "unknown"
                and not identity_evidence
            ):
                inferred_identity = "unknown"
                identity_confidence = "low"

        manual_label = manual_labels.get(
            speaker
        )

        if manual_label and manual_label.strip():
            effective_label = (
                manual_label.strip()
            )

            label_source: LabelSource = (
                "manual_override"
            )

        elif (
            inferred_identity.lower()
            != "unknown"
            and identity_confidence == "high"
        ):
            effective_label = (
                inferred_identity
            )

            label_source = (
                "inferred_identity"
            )

        elif (
            inferred_role.lower()
            != "unknown"
            and role_confidence == "high"
        ):
            effective_label = inferred_role
            label_source = "inferred_role"

        else:
            effective_label = speaker
            label_source = "raw_speaker_id"

        finalized_roles[speaker] = (
            SpeakerRoleResult(
                speaker=speaker,
                inferred_role=inferred_role,
                role_confidence=(
                    role_confidence
                ),
                role_evidence=role_evidence,
                inferred_identity=(
                    inferred_identity
                ),
                identity_confidence=(
                    identity_confidence
                ),
                identity_evidence=(
                    identity_evidence
                ),
                effective_label=(
                    effective_label
                ),
                label_source=label_source,
            )
        )

    return SpeakerRolesArtifact(
        source_id=source_id,
        roles=finalized_roles,
    )


def infer_speaker_roles(
    samples_by_speaker: dict[
        str,
        list[SpeakerSample],
    ],
    source_id: str,
    llm_client: LLMClient,
    manual_labels: dict[str, str] | None = None,
) -> SpeakerRolesArtifact:
    prompt = build_speaker_role_prompt(
        samples_by_speaker
    )

    response_schema = build_speaker_role_schema(
        samples_by_speaker
    )

    raw_response = llm_client.generate(
        system_prompt=(
            SPEAKER_ROLE_SYSTEM_PROMPT
        ),
        user_prompt=prompt,
        response_schema=response_schema,
        max_tokens=2048,
    )

    draft = parse_speaker_roles(
        raw_response
    )

    return finalize_speaker_roles(
        draft=draft,
        samples_by_speaker=samples_by_speaker,
        source_id=source_id,
        manual_labels=manual_labels,
    )

def save_speaker_roles(
    artifact: SpeakerRolesArtifact,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            artifact.model_dump(),
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path