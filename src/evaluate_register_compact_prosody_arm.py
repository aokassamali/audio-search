import json
from collections import Counter

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

from src.config import load_settings
from src.run_register_prosody_arm import (
    ProsodyArmArtifact,
)


LABELS = [
    "assertion",
    "hypothetical",
    "question",
    "characterization",
]


def main() -> None:
    settings = load_settings()

    prediction_path = (
        settings.paths.eval_dir
        / "register_prosody_compact_predictions.json"
    )

    output_path = (
        settings.paths.eval_dir
        / "register_prosody_compact_metrics.json"
    )

    with prediction_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        artifact = ProsodyArmArtifact.model_validate(
            json.load(file)
        )

    gold = [
        item.gold_label
        for item in artifact.items
    ]

    predicted = [
        item.predicted_label
        for item in artifact.items
    ]

    report = classification_report(
        gold,
        predicted,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    predicted_counts = Counter(predicted)

    metrics = {
        "accuracy": accuracy_score(
            gold,
            predicted,
        ),
        "macro_f1": f1_score(
            gold,
            predicted,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "per_class": {
            label: report[label]
            for label in LABELS
        },
        "predicted_counts": dict(
            predicted_counts
        ),
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print(
        f"Accuracy: {metrics['accuracy']:.3f}"
    )
    print(
        f"Macro F1: {metrics['macro_f1']:.3f}"
    )

    for label in LABELS:
        row = metrics["per_class"][label]

        print(
            f"{label:16}"
            f" support={int(row['support']):2d}"
            f" predicted="
            f"{predicted_counts[label]:2d}"
            f" precision={row['precision']:.3f}"
            f" recall={row['recall']:.3f}"
            f" f1={row['f1-score']:.3f}"
        )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()