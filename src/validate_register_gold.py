import json
from collections import Counter
from pathlib import Path

from src.config import load_settings
from src.register_gold import RegisterGoldArtifact


def validate_register_gold(
    gold_path: str | Path | None = None,
) -> None:
    settings = load_settings()

    if gold_path is None:
        gold_path = (
            settings.paths.eval_dir
            / "register_gold.json"
        )

    gold_path = Path(gold_path)

    with gold_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        artifact = RegisterGoldArtifact.model_validate(
            json.load(file)
        )

    items = artifact.items

    incomplete = [
        item
        for item in items
        if (
            item.label_permissive is None
            or item.label_strict is None
            or item.difficulty is None
        )
    ]

    permissive_counts = Counter(
        item.label_permissive
        for item in items
        if item.label_permissive is not None
    )

    strict_counts = Counter(
        item.label_strict
        for item in items
        if item.label_strict is not None
    )

    difficulty_counts = Counter(
        item.difficulty
        for item in items
        if item.difficulty is not None
    )

    disagreements = [
        item
        for item in items
        if (
            item.label_permissive is not None
            and item.label_strict is not None
            and item.label_permissive
            != item.label_strict
        )
    ]

    print(f"Gold path: {gold_path}")
    print(f"Total items: {len(items)}")
    print(f"Complete: {len(items) - len(incomplete)}")
    print(f"Incomplete: {len(incomplete)}")

    print("\nPermissive labels:")
    for label, count in sorted(
        permissive_counts.items()
    ):
        print(f"  {label}: {count}")

    print("\nStrict labels:")
    for label, count in sorted(
        strict_counts.items()
    ):
        print(f"  {label}: {count}")

    print("\nDifficulty:")
    for difficulty, count in sorted(
        difficulty_counts.items()
    ):
        print(f"  {difficulty}: {count}")

    agreement_count = (
        len(items) - len(disagreements)
    )

    agreement_rate = (
        agreement_count / len(items)
        if items
        else 0.0
    )

    print(
        "\nStrict/permissive agreement:"
        f"\n  same: {agreement_count}"
        f"\n  different: {len(disagreements)}"
        f"\n  agreement rate: "
        f"{agreement_rate:.1%}"
    )

    if incomplete:
        print("\nIncomplete turn IDs:")
        for item in incomplete:
            print(f"  {item.turn_id}")

    if disagreements:
        print("\nStrict/permissive disagreements:")

        for item in disagreements:
            print(
                f"\n{item.turn_id}"
                f"\n  permissive: "
                f"{item.label_permissive}"
                f"\n  strict: {item.label_strict}"
                f"\n  difficulty: {item.difficulty}"
                f"\n  text: {item.text}"
            )


if __name__ == "__main__":
    validate_register_gold()