import json
from collections import Counter

import numpy as np

from src.chunk import (
    build_speaker_turns,
    load_speaker_labels,
)
from src.config import load_settings
from src.register_gold import RegisterGoldArtifact


MINIMUM_WORDS = 8
TARGET_COUNT = 300


def summarize_durations(
    durations: list[float],
) -> dict:
    if not durations:
        return {
            "minimum": None,
            "median": None,
            "p90": None,
            "maximum": None,
        }

    values = np.asarray(
        durations,
        dtype=float,
    )

    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "maximum": float(np.max(values)),
    }


def main() -> None:
    settings = load_settings()
    eval_dir = settings.paths.eval_dir

    gold_path = eval_dir / "register_gold.json"

    existing_turn_ids = set()

    if gold_path.exists():
        with gold_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            gold = RegisterGoldArtifact.model_validate(
                json.load(file)
            )

        existing_turn_ids = {
            item.turn_id
            for item in gold.items
        }

    source_results = {}

    total_eligible = 0
    total_remaining = 0

    for source_key, source in settings.sources.items():
        transcript_path = (
            source.speaker_transcript_path
        )

        if not transcript_path.exists():
            source_results[source_key] = {
                "status": (
                    "missing_speaker_transcript"
                )
            }

            print(
                f"{source_key}: "
                "missing speaker transcript"
            )

            continue

        with transcript_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            segments = json.load(file)

        if source.speaker_roles_path.exists():
            speaker_labels = load_speaker_labels(
                source.speaker_roles_path
            )
        else:
            speaker_labels = {}

        turns = build_speaker_turns(
            segments,
            speaker_labels=speaker_labels,
        )

        eligible_turn_ids = []
        durations = []
        speaker_counts = Counter()

        for turn_index, turn in enumerate(turns):
            text = turn["text"].strip()
            word_count = len(text.split())

            if word_count < MINIMUM_WORDS:
                continue

            turn_id = (
                f"{source.source_id}:"
                f"turn_{turn_index:04d}"
            )

            duration = (
                float(turn["end"])
                - float(turn["start"])
            )

            eligible_turn_ids.append(turn_id)
            durations.append(duration)

            speaker_key = (
                f"{source.source_id}:"
                f"{turn['speaker']}"
            )

            speaker_counts[speaker_key] += 1

        already_sampled = sum(
            turn_id in existing_turn_ids
            for turn_id in eligible_turn_ids
        )

        remaining = (
            len(eligible_turn_ids)
            - already_sampled
        )

        total_eligible += len(eligible_turn_ids)
        total_remaining += remaining

        source_results[source_key] = {
            "status": "ready",
            "source_id": source.source_id,
            "eligible_turns": (
                len(eligible_turn_ids)
            ),
            "already_sampled": already_sampled,
            "remaining_turns": remaining,
            "speaker_count": len(speaker_counts),
            "speaker_turn_counts": dict(
                speaker_counts.most_common()
            ),
            "duration_seconds": (
                summarize_durations(durations)
            ),
        }

        duration_summary = (
            source_results[source_key][
                "duration_seconds"
            ]
        )

        print(f"\n{source_key}")
        print(
            f"  eligible: "
            f"{len(eligible_turn_ids)}"
        )
        print(
            f"  already sampled: "
            f"{already_sampled}"
        )
        print(
            f"  remaining: {remaining}"
        )
        print(
            f"  speakers: "
            f"{len(speaker_counts)}"
        )
        print(
            f"  duration median: "
            f"{duration_summary['median']:.1f}s"
        )
        print(
            f"  duration p90: "
            f"{duration_summary['p90']:.1f}s"
        )
        print(
            f"  duration max: "
            f"{duration_summary['maximum']:.1f}s"
        )

    current_count = len(existing_turn_ids)

    needed = max(
        0,
        TARGET_COUNT - current_count,
    )

    results = {
        "minimum_words": MINIMUM_WORDS,
        "target_count": TARGET_COUNT,
        "current_gold_count": current_count,
        "additional_turns_needed": needed,
        "total_eligible_turns": total_eligible,
        "total_remaining_turns": total_remaining,
        "enough_remaining_turns": (
            total_remaining >= needed
        ),
        "sources": source_results,
    }

    output_path = (
        eval_dir
        / "speech_act_pool_audit.json"
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
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nTOTAL")
    print(f"  current gold: {current_count}")
    print(f"  target: {TARGET_COUNT}")
    print(f"  additional needed: {needed}")
    print(
        f"  remaining candidates: "
        f"{total_remaining}"
    )
    print(
        f"  enough candidates: "
        f"{total_remaining >= needed}"
    )
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()