import json
from collections import defaultdict
from pathlib import Path


def overlap_duration(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    return max(
        0.0,
        min(end_a, end_b) - max(start_a, start_b),
    )


def assign_speaker(
    segment: dict,
    diarization_turns: list[dict],
) -> dict:
    overlap_by_speaker = defaultdict(float)

    for turn in diarization_turns:
        overlap = overlap_duration(
            segment["start"],
            segment["end"],
            turn["start"],
            turn["end"],
        )

        if overlap > 0:
            overlap_by_speaker[
                turn["speaker"]
            ] += overlap

    if not overlap_by_speaker:
        return {
            "speaker": "UNKNOWN",
            "speaker_overlap_ratio": 0.0,
            "diarization_coverage": 0.0,
        }

    best_speaker = max(
        overlap_by_speaker,
        key=overlap_by_speaker.get,
    )

    best_overlap = overlap_by_speaker[
        best_speaker
    ]

    total_speaker_overlap = sum(
        overlap_by_speaker.values()
    )

    segment_duration = (
        segment["end"] - segment["start"]
    )

    speaker_overlap_ratio = (
        best_overlap / total_speaker_overlap
        if total_speaker_overlap > 0
        else 0.0
    )

    diarization_coverage = (
        total_speaker_overlap / segment_duration
        if segment_duration > 0
        else 0.0
    )

    return {
        "speaker": best_speaker,
        "speaker_overlap_ratio": (
            speaker_overlap_ratio
        ),
        "diarization_coverage": (
            diarization_coverage
        ),
    }


def align_transcript_speakers(
    transcript_path: str | Path,
    diarization_path: str | Path,
    output_path: str | Path,
) -> Path:
    transcript_path = Path(transcript_path)
    diarization_path = Path(diarization_path)
    output_path = Path(output_path)

    with transcript_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        transcript_segments = json.load(file)

    with diarization_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        diarization_turns = json.load(file)

    aligned_segments = []

    for segment in transcript_segments:
        speaker_assignment = assign_speaker(
            segment,
            diarization_turns,
        )

        aligned_segment = {
            **segment,
            **speaker_assignment,
        }

        aligned_segments.append(
            aligned_segment
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            aligned_segments,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path