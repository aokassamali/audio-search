import json
from collections import Counter
from pathlib import Path

from src.config import load_settings
from src.register_gold import RegisterGoldArtifact
from src.speech_act_gold import SpeechActGoldArtifact


EXPECTED_SOURCES = {
    "sripetch": 100,
    "case_two": 100,
    "case_three": 100,
    "case_four": 100,
}


def is_complete(item) -> bool:
    return (
        item.label_permissive is not None
        and item.label_strict is not None
        and item.difficulty is not None
    )


def main() -> None:
    settings = load_settings()
    eval_dir = settings.paths.eval_dir

    gold_path = (
        eval_dir
        / "speech_act_gold_v24.json"
    )

    frozen_path = (
        eval_dir
        / "register_gold.json"
    )

    with gold_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        artifact = (
            SpeechActGoldArtifact.model_validate(
                json.load(file)
            )
        )

    with frozen_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        frozen_artifact = (
            RegisterGoldArtifact.model_validate(
                json.load(file)
            )
        )

    problems = []

    if len(artifact.items) != 400:
        problems.append(
            f"Expected 400 items. "
            f"Found {len(artifact.items)}."
        )

    turn_ids = [
        item.turn_id
        for item in artifact.items
    ]

    duplicate_ids = [
        turn_id
        for turn_id, count
        in Counter(turn_ids).items()
        if count > 1
    ]

    if duplicate_ids:
        problems.append(
            "Duplicate turn IDs: "
            + ", ".join(duplicate_ids)
        )

    incomplete_items = [
        item.turn_id
        for item in artifact.items
        if not is_complete(item)
    ]

    if incomplete_items:
        problems.append(
            f"Incomplete items: "
            f"{len(incomplete_items)}"
        )

    source_counts = Counter(
        item.source_key
        for item in artifact.items
    )

    for source_key, expected_count in (
        EXPECTED_SOURCES.items()
    ):
        actual_count = source_counts[
            source_key
        ]

        if actual_count != expected_count:
            problems.append(
                f"{source_key}: expected "
                f"{expected_count}, found "
                f"{actual_count}."
            )

    frozen_items = [
        item
        for item in artifact.items
        if item.frozen
    ]

    if len(frozen_items) != 80:
        problems.append(
            f"Expected 80 frozen items. "
            f"Found {len(frozen_items)}."
        )

    original_lookup = {
        item.turn_id: item
        for item in frozen_artifact.items
    }

    changed_frozen_items = []

    for item in frozen_items:
        original = original_lookup.get(
            item.turn_id
        )

        if original is None:
            changed_frozen_items.append(
                item.turn_id
            )
            continue

        current_payload = item.model_dump(
            exclude={
                "source_key",
                "frozen",
            }
        )

        if current_payload != original.model_dump():
            changed_frozen_items.append(
                item.turn_id
            )

    if changed_frozen_items:
        problems.append(
            f"Changed frozen items: "
            f"{len(changed_frozen_items)}"
        )

    permissive_counts = Counter(
        item.label_permissive
        for item in artifact.items
    )

    strict_counts = Counter(
        item.label_strict
        for item in artifact.items
    )

    difficulty_counts = Counter(
        item.difficulty
        for item in artifact.items
    )

    disagreements = [
        item
        for item in artifact.items
        if item.label_permissive
        != item.label_strict
    ]

    transitions = Counter(
        (
            item.label_permissive,
            item.label_strict,
        )
        for item in disagreements
    )

    summary = {
        "total_items": len(artifact.items),
        "complete_items": sum(
            is_complete(item)
            for item in artifact.items
        ),
        "frozen_items": len(frozen_items),
        "source_counts": dict(source_counts),
        "permissive_counts": dict(
            permissive_counts
        ),
        "strict_counts": dict(
            strict_counts
        ),
        "difficulty_counts": dict(
            difficulty_counts
        ),
        "strict_permissive_agreement": (
            (
                len(artifact.items)
                - len(disagreements)
            )
            / len(artifact.items)
        ),
        "disagreement_count": (
            len(disagreements)
        ),
        "transitions": {
            f"{source}->{target}": count
            for (
                source,
                target,
            ), count in transitions.items()
        },
        "validation_problems": problems,
    }

    output_path = (
        eval_dir
        / "speech_act_gold_v24_summary.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    if problems:
        print("VALIDATION FAILED")

        for problem in problems:
            print(f"  {problem}")

        return

    print("VALIDATION PASSED")
    print(
        f"Items: "
        f"{summary['complete_items']}/"
        f"{summary['total_items']}"
    )
    print(
        f"Frozen: "
        f"{summary['frozen_items']}"
    )

    print("\nSources")

    for source_key in EXPECTED_SOURCES:
        print(
            f"  {source_key}: "
            f"{source_counts[source_key]}"
        )

    print("\nPermissive labels")

    for label, count in (
        permissive_counts.most_common()
    ):
        print(f"  {label}: {count}")

    print("\nStrict labels")

    for label, count in (
        strict_counts.most_common()
    ):
        print(f"  {label}: {count}")

    print("\nDifficulty")

    for difficulty, count in (
        difficulty_counts.most_common()
    ):
        print(
            f"  {difficulty}: {count}"
        )

    print(
        "\nStrict/permissive agreement: "
        f"{summary[
            'strict_permissive_agreement'
        ]:.1%}"
    )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()