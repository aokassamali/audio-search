import argparse
import json
from pathlib import Path

from src.config import load_settings


SOFT_TARGET = 120
HARD_CAP = 200


def load_speaker_labels(
    speaker_roles_path: str | Path,
) -> dict[str, str]:
    speaker_roles_path = Path(
        speaker_roles_path
    )

    with speaker_roles_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        artifact = json.load(file)

    labels = {}

    for speaker, role in artifact.get(
        "roles",
        {},
    ).items():
        effective_label = role.get(
            "effective_label"
        )

        if (
            isinstance(effective_label, str)
            and effective_label.strip()
        ):
            labels[speaker] = (
                effective_label.strip()
            )

    return labels


def create_output_path(
    transcription: str | Path,
    output_directory: str | Path,
) -> Path:
    transcription_path = Path(transcription)
    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        output_directory
        / f"{transcription_path.stem}_chunks.json"
    )


def build_chunk(
    segments,
    chunk_id,
    source_id,
    speaker_labels=None,
):
    text = " ".join(
        segment["text"].strip()
        for segment in segments
    )

    chunk = {
        "source_id": source_id,
        "chunk_id": chunk_id,
        "text": text,
        "start": segments[0]["start"],
        "end": segments[-1]["end"],
        "word_count": len(text.split()),
    }

    has_speaker_data = any(
        "speaker" in segment
        for segment in segments
    )

    if has_speaker_data:
        speaker_turns = build_speaker_turns(
            segments,
            speaker_labels=speaker_labels,
        )

        chunk["speakers"] = list(
            dict.fromkeys(
                turn["speaker"]
                for turn in speaker_turns
            )
        )

        chunk["speaker_labels"] = {
            speaker: (
                speaker_labels or {}
            ).get(
                speaker,
                speaker,
            )
            for speaker in chunk["speakers"]
        }

        chunk["speaker_turns"] = speaker_turns

        speaker_lines = []

        for turn in speaker_turns:
            speaker = turn["speaker"]
            speaker_label = turn[
                "speaker_label"
            ]

            if speaker_label == speaker:
                display_name = speaker
            else:
                display_name = (
                    f"{speaker_label} "
                    f"[{speaker}]"
                )

            speaker_lines.append(
                f"{display_name}: "
                f"{turn['text']}"
            )

        chunk["speaker_text"] = "\n".join(
            speaker_lines
        )

    return chunk


def create_chunks(
    segments,
    source_id,
    soft_target=SOFT_TARGET,
    hard_cap=HARD_CAP,
    speaker_labels=None,
):
    current_segments = []
    current_word_count = 0
    all_chunks = []
    chunk_id = 0
    just_closed_chunk = False

    for segment in segments:
        just_closed_chunk = False
        segment_text = segment["text"].strip()

        current_segments.append(segment)
        current_word_count += len(segment_text.split())

        reached_soft_target = (
            current_word_count >= soft_target
            and segment_text.endswith((".", "?", "!"))
        )

        reached_hard_cap = (
            current_word_count >= hard_cap
        )

        if reached_soft_target or reached_hard_cap:
            all_chunks.append(
                build_chunk(
                    current_segments,
                    chunk_id,
                    source_id,
                    speaker_labels=speaker_labels,
                )
            )

            chunk_id += 1

            overlap_segment = current_segments[-1]
            current_segments = [overlap_segment]
            current_word_count = len(
                overlap_segment["text"].split()
            )

            just_closed_chunk = True

    if current_segments and not just_closed_chunk:
        all_chunks.append(
            build_chunk(
                current_segments,
                chunk_id,
                source_id,
                speaker_labels=speaker_labels,
            )
        )

    return all_chunks


def chunk_transcript(
    transcription,
    output_path=None,
    source_id=None,
    soft_target=SOFT_TARGET,
    hard_cap=HARD_CAP,
    speaker_labels=None,
):
    transcription_path = Path(transcription)
    if source_id is None:
        source_id = transcription_path.stem

    with transcription_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        segments = json.load(file)

    all_chunks = create_chunks(
        segments,
        source_id=source_id,
        soft_target=soft_target,
        hard_cap=hard_cap,
        speaker_labels=speaker_labels,
    )

    if output_path is None:
        settings = load_settings()

        output_path = create_output_path(
            transcription=transcription_path,
            output_directory=(
                settings.paths.chunks_dir
            ),
        )
    else:
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
            all_chunks,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path, len(all_chunks)


def build_speaker_turns(
    segments: list[dict],
    speaker_labels: dict[str, str] | None = None,
) -> list[dict]:
    speaker_labels = speaker_labels or {}

    speaker_turns = []

    for segment in segments:
        text = segment["text"].strip()

        if not text:
            continue

        speaker = segment.get(
            "speaker",
            "UNKNOWN",
        )

        speaker_label = speaker_labels.get(
            speaker,
            speaker,
        )

        if (
            speaker_turns
            and speaker_turns[-1]["speaker"] == speaker
        ):
            speaker_turns[-1]["end"] = segment["end"]
            speaker_turns[-1]["text"] += f" {text}"

        else:
            speaker_turns.append(
                {
                    "speaker": speaker,
                    "speaker_label": speaker_label,
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": text,
                }
            )

    return speaker_turns


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "transcription",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--source-id",
        default=None,
    )

    parser.add_argument(
        "--speaker-roles",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    speaker_labels = None

    if args.speaker_roles is not None:
        speaker_labels = load_speaker_labels(
            args.speaker_roles
        )

    output_path, chunk_count = chunk_transcript(
        transcription=args.transcription,
        output_path=args.output,
        source_id=args.source_id,
        speaker_labels=speaker_labels,
    )

    print(f"Created {chunk_count} chunks.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()