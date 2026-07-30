import json
from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from src.config import load_settings
from src.speech_act_gold import SpeechActGoldArtifact


LABELS = [
    "assertion",
    "question",
    "characterization",
    "hypothetical",
]


def ordered_counts(values) -> dict[str, int]:
    counts = Counter(values)

    return {
        label: counts[label]
        for label in LABELS
    }


def main() -> None:
    settings = load_settings()

    input_path = (
        settings.paths.eval_dir
        / "speech_act_gold_v24.json"
    )

    output_path = (
        settings.paths.eval_dir
        / "speech_act_split_audit.json"
    )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        artifact = (
            SpeechActGoldArtifact.model_validate(
                json.load(file)
            )
        )

    items = artifact.items

    print("OVERALL")
    overall_counts = ordered_counts(
        item.label_permissive
        for item in items
    )

    for label, count in overall_counts.items():
        print(f"  {label}: {count}")

    majority_count = max(
        overall_counts.values()
    )

    print(
        "  majority baseline: "
        f"{majority_count / len(items):.3f}"
    )

    source_results = {}

    print("\nBY SOURCE")

    for source_key in sorted(
        {
            item.source_key
            for item in items
        }
    ):
        source_items = [
            item
            for item in items
            if item.source_key == source_key
        ]

        counts = ordered_counts(
            item.label_permissive
            for item in source_items
        )

        source_results[source_key] = counts

        print(f"\n{source_key}")

        for label, count in counts.items():
            print(f"  {label}: {count}")

    group_items = defaultdict(list)

    for item in items:
        provisional_group = (
            f"{item.source_key}:"
            f"{item.speaker}"
        )

        group_items[
            provisional_group
        ].append(item)

    minority_groups = {}

    print(
        "\nGROUPS CONTAINING "
        "HYPOTHETICAL OR CHARACTERIZATION"
    )

    for group, grouped_items in sorted(
        group_items.items()
    ):
        counts = ordered_counts(
            item.label_permissive
            for item in grouped_items
        )

        if (
            counts["hypothetical"] == 0
            and counts["characterization"] == 0
        ):
            continue

        minority_groups[group] = counts

        print(f"\n{group}")

        for label, count in counts.items():
            if count:
                print(f"  {label}: {count}")

    y = np.asarray(
        [
            item.label_permissive
            for item in items
        ]
    )

    groups = np.asarray(
        [
            f"{item.source_key}:"
            f"{item.speaker}"
            for item in items
        ]
    )

    X = np.zeros(
        (len(items), 1),
        dtype=float,
    )

    splitter = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=24,
    )

    fold_results = []

    print("\nPROVISIONAL 4-FOLD SPLIT")

    for fold, (
        train_indices,
        test_indices,
    ) in enumerate(
        splitter.split(
            X,
            y,
            groups=groups,
        ),
        start=1,
    ):
        train_counts = ordered_counts(
            y[train_indices]
        )

        test_counts = ordered_counts(
            y[test_indices]
        )

        train_groups = set(
            groups[train_indices]
        )

        test_groups = set(
            groups[test_indices]
        )

        overlap = (
            train_groups
            & test_groups
        )

        missing_train = [
            label
            for label, count
            in train_counts.items()
            if count == 0
        ]

        missing_test = [
            label
            for label, count
            in test_counts.items()
            if count == 0
        ]

        fold_result = {
            "fold": fold,
            "train_size": len(train_indices),
            "test_size": len(test_indices),
            "train_counts": train_counts,
            "test_counts": test_counts,
            "group_overlap": sorted(overlap),
            "missing_train_labels": (
                missing_train
            ),
            "missing_test_labels": (
                missing_test
            ),
        }

        fold_results.append(fold_result)

        print(
            f"\nFold {fold}"
            f"\n  train: {len(train_indices)} "
            f"{train_counts}"
            f"\n  test:  {len(test_indices)} "
            f"{test_counts}"
            f"\n  group overlap: "
            f"{len(overlap)}"
            f"\n  missing test labels: "
            f"{missing_test or 'none'}"
        )

    results = {
        "overall_counts": overall_counts,
        "majority_baseline_accuracy": (
            majority_count / len(items)
        ),
        "source_counts": source_results,
        "minority_class_groups": (
            minority_groups
        ),
        "provisional_group_definition": (
            "source_key:raw_speaker_id"
        ),
        "folds": fold_results,
        "warning": (
            "The grouping is provisional. "
            "Recurring justices must later share "
            "one canonical group across cases."
        ),
    }

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

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()