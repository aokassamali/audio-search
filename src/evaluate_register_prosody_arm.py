import json
from collections import Counter
from statistics import mean

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
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
        / "register_prosody_predictions.json"
    )

    output_path = (
        settings.paths.eval_dir
        / "register_prosody_metrics.json"
    )

    with prediction_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        artifact = (
            ProsodyArmArtifact.model_validate(
                json.load(file)
            )
        )

    true_labels = [
        item.gold_label
        for item in artifact.items
    ]

    predicted_labels = [
        item.predicted_label
        for item in artifact.items
    ]

    confidences = [
        item.confidence
        for item in artifact.items
    ]

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    gold_counts = Counter(true_labels)
    predicted_counts = Counter(
        predicted_labels
    )

    majority_count = max(
        gold_counts.values()
    )

    majority_accuracy = (
        majority_count / len(true_labels)
    )

    correct_confidences = [
        confidence
        for gold, predicted, confidence
        in zip(
            true_labels,
            predicted_labels,
            confidences,
        )
        if gold == predicted
    ]

    incorrect_confidences = [
        confidence
        for gold, predicted, confidence
        in zip(
            true_labels,
            predicted_labels,
            confidences,
        )
        if gold != predicted
    ]

    metrics = {
        "count": len(true_labels),
        "accuracy": accuracy_score(
            true_labels,
            predicted_labels,
        ),
        "macro_f1": f1_score(
            true_labels,
            predicted_labels,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "majority_baseline_accuracy": (
            majority_accuracy
        ),
        "per_class": {
            label: report[label]
            for label in LABELS
        },
        "gold_counts": {
            label: gold_counts[label]
            for label in LABELS
        },
        "predicted_counts": {
            label: predicted_counts[label]
            for label in LABELS
        },
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": (
            confusion_matrix(
                true_labels,
                predicted_labels,
                labels=LABELS,
            ).tolist()
        ),
        "mean_confidence_correct": (
            mean(correct_confidences)
            if correct_confidences
            else None
        ),
        "mean_confidence_incorrect": (
            mean(incorrect_confidences)
            if incorrect_confidences
            else None
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
        f"Accuracy: "
        f"{metrics['accuracy']:.3f}"
    )

    print(
        f"Macro F1: "
        f"{metrics['macro_f1']:.3f}"
    )

    print(
        "Majority baseline accuracy: "
        f"{metrics[
            'majority_baseline_accuracy'
        ]:.3f}"
    )

    print("\nPer class:")

    for label in LABELS:
        row = metrics["per_class"][label]

        print(
            f"  {label:16}"
            f" support={int(row['support']):2d}"
            f" predicted="
            f"{metrics['predicted_counts'][label]:2d}"
            f" precision={row['precision']:.3f}"
            f" recall={row['recall']:.3f}"
            f" f1={row['f1-score']:.3f}"
        )

    print(
        "\nMean confidence:"
        f"\n  correct: "
        f"{metrics['mean_confidence_correct']:.3f}"
        f"\n  incorrect: "
        f"{metrics['mean_confidence_incorrect']:.3f}"
    )

    print("\nConfusion matrix:")
    print("  labels:", LABELS)

    for row in metrics["confusion_matrix"]:
        print(" ", row)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()