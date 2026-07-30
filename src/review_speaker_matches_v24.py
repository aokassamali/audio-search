import json
from pathlib import Path

from src.annotate_register import (
    play_audio_slice,
    shorten_text,
)
from src.config import load_settings


MINIMUM_SIMILARITY = 0.97
REPRESENTATIVES_PER_GROUP = 2


def pair_key(
    left_group: str,
    right_group: str,
) -> str:
    return "||".join(
        sorted(
            [
                left_group,
                right_group,
            ]
        )
    )


def save_reviews(
    output_path: Path,
    artifact: dict,
) -> None:
    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            artifact,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temporary_path.replace(output_path)


def build_candidate_pairs(
    candidates: dict,
) -> list[dict]:
    pair_lookup = {}

    top_matches = candidates[
        "top_cross_source_matches"
    ]

    for left_group, matches in (
        top_matches.items()
    ):
        if left_group.endswith(":UNKNOWN"):
            continue

        for match in matches:
            right_group = match["group_id"]

            if right_group.endswith(":UNKNOWN"):
                continue

            similarity = float(
                match["similarity"]
            )

            if similarity < MINIMUM_SIMILARITY:
                continue

            key = pair_key(
                left_group,
                right_group,
            )

            existing = pair_lookup.get(key)

            if (
                existing is None
                or similarity
                > existing["similarity"]
            ):
                ordered_groups = sorted(
                    [
                        left_group,
                        right_group,
                    ]
                )

                pair_lookup[key] = {
                    "pair_key": key,
                    "left_group": (
                        ordered_groups[0]
                    ),
                    "right_group": (
                        ordered_groups[1]
                    ),
                    "similarity": similarity,
                    "selection_reason": (
                        "top-three similarity "
                        f">= {MINIMUM_SIMILARITY}"
                    ),
                }

    for pair in candidates[
        "mutual_top_pairs"
    ]:
        left_group = pair["left_group"]
        right_group = pair["right_group"]

        if (
            left_group.endswith(":UNKNOWN")
            or right_group.endswith(
                ":UNKNOWN"
            )
        ):
            continue

        key = pair_key(
            left_group,
            right_group,
        )

        ordered_groups = sorted(
            [
                left_group,
                right_group,
            ]
        )

        record = pair_lookup.get(
            key,
            {
                "pair_key": key,
                "left_group": (
                    ordered_groups[0]
                ),
                "right_group": (
                    ordered_groups[1]
                ),
                "similarity": float(
                    pair["similarity"]
                ),
                "selection_reason": (
                    "mutual top match"
                ),
            },
        )

        record["selection_reason"] = (
            "mutual top match"
        )

        record["similarity"] = max(
            record["similarity"],
            float(pair["similarity"]),
        )

        pair_lookup[key] = record

    return sorted(
        pair_lookup.values(),
        key=lambda pair: (
            -pair["similarity"],
            pair["pair_key"],
        ),
    )


def display_group(
    group_id: str,
    group_record: dict,
) -> None:
    print(
        f"\n{group_id}"
        f"\n  turns: "
        f"{group_record['turn_count']}"
        f"\n  labeled duration: "
        f"{group_record['total_duration']:.1f}s"
    )

    representatives = group_record[
        "representative_turns"
    ]

    for index, turn in enumerate(
        representatives[
            :REPRESENTATIVES_PER_GROUP
        ],
        start=1,
    ):
        print(
            f"  clip {index}: "
            f"{turn['start']:.1f}-"
            f"{turn['end']:.1f}s"
            f"\n    "
            f"{shorten_text(
                turn['text'],
                maximum_characters=180,
            )}"
        )


def play_group(
    group_id: str,
    group_record: dict,
    source_files: dict,
    settings,
    play_all: bool,
) -> None:
    source_key = group_record[
        "source_key"
    ]

    source = settings.get_source(
        source_key
    )

    representatives = group_record[
        "representative_turns"
    ]

    if play_all:
        selected = representatives[
            :REPRESENTATIVES_PER_GROUP
        ]
    else:
        selected = representatives[:1]

    print(f"\nPlaying {group_id}")

    for turn in selected:
        play_audio_slice(
            audio_path=(
                source.normalized_audio_path
            ),
            start=float(turn["start"]),
            end=float(turn["end"]),
            padding=0.25,
        )


def main() -> None:
    settings = load_settings()
    eval_dir = settings.paths.eval_dir

    candidate_path = (
        eval_dir
        / "speaker_match_candidates_v24.json"
    )

    output_path = (
        eval_dir
        / "speaker_match_reviews_v24.json"
    )

    with candidate_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        candidates = json.load(file)

    pairs = build_candidate_pairs(
        candidates
    )

    if output_path.exists():
        with output_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            review_artifact = json.load(file)
    else:
        review_artifact = {
            "schema_version": "1.0",
            "minimum_similarity": (
                MINIMUM_SIMILARITY
            ),
            "decisions": [],
        }

    existing_lookup = {
        decision["pair_key"]: decision
        for decision
        in review_artifact["decisions"]
    }

    pending_pairs = [
        pair
        for pair in pairs
        if pair["pair_key"]
        not in existing_lookup
    ]

    print(
        f"Candidate pairs: {len(pairs)}"
        f"\nAlready reviewed: "
        f"{len(pairs) - len(pending_pairs)}"
        f"\nRemaining: "
        f"{len(pending_pairs)}"
    )

    if not pending_pairs:
        print("All candidate pairs reviewed.")
        return

    groups = candidates["groups"]
    source_files = candidates[
        "source_files"
    ]

    for progress, pair in enumerate(
        pending_pairs,
        start=1,
    ):
        left_group = pair["left_group"]
        right_group = pair["right_group"]

        left_record = groups[left_group]
        right_record = groups[right_group]

        print("\n" + "=" * 80)

        print(
            f"Pair {progress}/"
            f"{len(pending_pairs)}"
            f"\nSimilarity: "
            f"{pair['similarity']:.4f}"
            f"\nReason: "
            f"{pair['selection_reason']}"
        )

        display_group(
            left_group,
            left_record,
        )

        display_group(
            right_group,
            right_record,
        )

        play_group(
            group_id=left_group,
            group_record=left_record,
            source_files=source_files,
            settings=settings,
            play_all=False,
        )

        play_group(
            group_id=right_group,
            group_record=right_record,
            source_files=source_files,
            settings=settings,
            play_all=False,
        )

        while True:
            action = input(
                "\n1 = same voice"
                "\n2 = different voices"
                "\n3 = unsure"
                "\nr = replay first clips"
                "\na = play all clips"
                "\nq = quit"
                "\nDecision: "
            ).strip().lower()

            if action == "r":
                play_group(
                    left_group,
                    left_record,
                    source_files,
                    settings,
                    play_all=False,
                )

                play_group(
                    right_group,
                    right_record,
                    source_files,
                    settings,
                    play_all=False,
                )

                continue

            if action == "a":
                play_group(
                    left_group,
                    left_record,
                    source_files,
                    settings,
                    play_all=True,
                )

                play_group(
                    right_group,
                    right_record,
                    source_files,
                    settings,
                    play_all=True,
                )

                continue

            if action == "q":
                print(
                    f"Progress saved in "
                    f"{output_path}"
                )
                return

            decision_lookup = {
                "1": "same",
                "2": "different",
                "3": "unsure",
            }

            if action not in decision_lookup:
                print(
                    "Enter 1, 2, 3, r, a, or q."
                )
                continue

            notes = input(
                "Notes, optional: "
            ).strip()

            decision_record = {
                **pair,
                "decision": (
                    decision_lookup[action]
                ),
                "notes": notes,
            }

            review_artifact[
                "decisions"
            ].append(decision_record)

            save_reviews(
                output_path,
                review_artifact,
            )

            print(
                f"Saved: "
                f"{len(
                    review_artifact[
                        'decisions'
                    ]
                )}/{len(pairs)}"
            )

            break

    print("\nSpeaker review complete")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()