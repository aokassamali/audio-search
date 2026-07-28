import json
from collections import Counter
from pathlib import Path
from statistics import mean

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.config import load_settings
from src.run_register_text_arm import (
    TextArmArtifact,
)


LABELS = [
    "assertion",
    "hypothetical",
    "question",
    "characterization",
    "hyperbole",
    "joke",
]


def summarize_variant(
    artifact: TextArmArtifact,
    variant: str,
) -> dict:
    true_labels = []
    predicted_labels = []
    confidences = []
    difficulties = []

    for item in artifact.items:
        if variant == "permissive":
            gold_label = item.gold_permissive
            prediction = (
                item.permissive_prediction
            )
        else:
            gold_label = item.gold_strict
            prediction = item.strict_prediction

        if prediction is None:
            continue

        true_labels.append(gold_label)
        predicted_labels.append(
            prediction.label
        )
        confidences.append(
            prediction.confidence
        )
        difficulties.append(
            item.difficulty
        )

    supported_labels = [
        label
        for label in LABELS
        if label in true_labels
    ]

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    predicted_counts = Counter(
        predicted_labels
    )

    by_difficulty = {}

    for difficulty in ["easy", "hard"]:
        indices = [
            index
            for index, value
            in enumerate(difficulties)
            if value == difficulty
        ]

        correct = [
            true_labels[index]
            == predicted_labels[index]
            for index in indices
        ]

        difficulty_confidences = [
            confidences[index]
            for index in indices
        ]

        by_difficulty[difficulty] = {
            "count": len(indices),
            "accuracy": (
                sum(correct) / len(correct)
                if correct
                else None
            ),
            "mean_confidence": (
                mean(
                    difficulty_confidences
                )
                if difficulty_confidences
                else None
            ),
        }

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

    return {
        "variant": variant,
        "count": len(true_labels),
        "accuracy": accuracy_score(
            true_labels,
            predicted_labels,
        ),
        "macro_f1_supported_classes": (
            f1_score(
                true_labels,
                predicted_labels,
                labels=supported_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "supported_labels": (
            supported_labels
        ),
        "predicted_counts": {
            label: predicted_counts[label]
            for label in LABELS
        },
        "per_class": {
            label: report[label]
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
        "by_difficulty": by_difficulty,
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


def main() -> None:
    settings = load_settings()

    prediction_path = (
        settings.paths.eval_dir
        / "register_text_predictions.json"
    )

    output_path = (
        settings.paths.eval_dir
        / "register_text_metrics.json"
    )

    with prediction_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        artifact = (
            TextArmArtifact.model_validate(
                json.load(file)
            )
        )

    results = {
        variant: summarize_variant(
            artifact,
            variant,
        )
        for variant in [
            "permissive",
            "strict",
        ]
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    for variant, metrics in (
        results.items()
    ):
        print(
            f"\n{variant.upper()}"
        )
        print(
            f"Accuracy: "
            f"{metrics['accuracy']:.3f}"
        )
        print(
            "Macro F1 "
            "(supported classes): "
            f"{metrics[
                'macro_f1_supported_classes'
            ]:.3f}"
        )

        print("\nPer class:")

        for label in LABELS:
            row = metrics[
                "per_class"
            ][label]

            predicted_count = metrics[
                "predicted_counts"
            ][label]

            print(
                f"  {label:16}"
                f" support={int(row['support']):2d}"
                f" predicted={predicted_count:2d}"
                f" precision={row['precision']:.3f}"
                f" recall={row['recall']:.3f}"
                f" f1={row['f1-score']:.3f}"
            )

        print("\nDifficulty:")

        for difficulty in [
            "easy",
            "hard",
        ]:
            row = metrics[
                "by_difficulty"
            ][difficulty]

            print(
                f"  {difficulty}: "
                f"n={row['count']}, "
                f"accuracy="
                f"{row['accuracy']:.3f}, "
                f"mean confidence="
                f"{row['mean_confidence']:.3f}"
            )

        print(
            "\nMean confidence:"
            f"\n  correct: "
            f"{metrics[
                'mean_confidence_correct'
            ]:.3f}"
            f"\n  incorrect: "
            f"{metrics[
                'mean_confidence_incorrect'
            ]:.3f}"
        )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()