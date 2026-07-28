import json
from collections import Counter
from statistics import mean

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

from src.config import load_settings
from src.run_register_multimodal_arm import (
    MultimodalArmArtifact,
)
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


def evaluate_variant(
    multimodal: MultimodalArmArtifact,
    text_only: TextArmArtifact,
    variant: str,
) -> dict:
    text_items = {
        item.turn_id: item
        for item in text_only.items
    }

    gold_labels = []
    multimodal_labels = []
    text_labels = []
    confidences = []
    difficulties = []

    for item in multimodal.items:
        text_item = text_items[item.turn_id]

        if variant == "permissive":
            gold = item.gold_permissive
            multimodal_prediction = (
                item.permissive_prediction
            )
            text_prediction = (
                text_item.permissive_prediction
            )
        else:
            gold = item.gold_strict
            multimodal_prediction = (
                item.strict_prediction
            )
            text_prediction = (
                text_item.strict_prediction
            )

        if (
            multimodal_prediction is None
            or text_prediction is None
        ):
            continue

        gold_labels.append(gold)
        multimodal_labels.append(
            multimodal_prediction.label
        )
        text_labels.append(
            text_prediction.label
        )
        confidences.append(
            multimodal_prediction.confidence
        )
        difficulties.append(
            item.difficulty
        )

    supported_labels = [
        label
        for label in LABELS
        if label in gold_labels
    ]

    report = classification_report(
        gold_labels,
        multimodal_labels,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    predicted_counts = Counter(
        multimodal_labels
    )

    fixes = 0
    regressions = 0
    both_correct = 0
    both_wrong = 0
    prediction_changes = 0

    for gold, text_label, multimodal_label in zip(
        gold_labels,
        text_labels,
        multimodal_labels,
    ):
        text_correct = text_label == gold
        multimodal_correct = (
            multimodal_label == gold
        )

        if text_label != multimodal_label:
            prediction_changes += 1

        if not text_correct and multimodal_correct:
            fixes += 1
        elif text_correct and not multimodal_correct:
            regressions += 1
        elif text_correct and multimodal_correct:
            both_correct += 1
        else:
            both_wrong += 1

    by_difficulty = {}

    for difficulty in ["easy", "hard"]:
        indices = [
            index
            for index, value
            in enumerate(difficulties)
            if value == difficulty
        ]

        correct = [
            gold_labels[index]
            == multimodal_labels[index]
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
                mean(difficulty_confidences)
                if difficulty_confidences
                else None
            ),
        }

    return {
        "variant": variant,
        "count": len(gold_labels),
        "accuracy": accuracy_score(
            gold_labels,
            multimodal_labels,
        ),
        "text_only_accuracy": accuracy_score(
            gold_labels,
            text_labels,
        ),
        "macro_f1_supported_classes": f1_score(
            gold_labels,
            multimodal_labels,
            labels=supported_labels,
            average="macro",
            zero_division=0,
        ),
        "per_class": {
            label: report[label]
            for label in LABELS
        },
        "predicted_counts": {
            label: predicted_counts[label]
            for label in LABELS
        },
        "comparison_to_text_only": {
            "fixes": fixes,
            "regressions": regressions,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "prediction_changes": (
                prediction_changes
            ),
        },
        "by_difficulty": by_difficulty,
    }


def main() -> None:
    settings = load_settings()

    multimodal_path = (
        settings.paths.eval_dir
        / "register_multimodal_predictions.json"
    )

    text_path = (
        settings.paths.eval_dir
        / "register_text_predictions.json"
    )

    output_path = (
        settings.paths.eval_dir
        / "register_multimodal_metrics.json"
    )

    with multimodal_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        multimodal = (
            MultimodalArmArtifact.model_validate(
                json.load(file)
            )
        )

    with text_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        text_only = (
            TextArmArtifact.model_validate(
                json.load(file)
            )
        )

    results = {
        variant: evaluate_variant(
            multimodal,
            text_only,
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

    for variant, metrics in results.items():
        comparison = metrics[
            "comparison_to_text_only"
        ]

        print(f"\n{variant.upper()}")
        print(
            "Multimodal accuracy: "
            f"{metrics['accuracy']:.3f}"
        )
        print(
            "Text-only accuracy: "
            f"{metrics['text_only_accuracy']:.3f}"
        )
        print(
            "Macro F1: "
            f"{metrics[
                'macro_f1_supported_classes'
            ]:.3f}"
        )

        print(
            "\nCompared with text-only:"
            f"\n  fixes: {comparison['fixes']}"
            f"\n  regressions: "
            f"{comparison['regressions']}"
            f"\n  prediction changes: "
            f"{comparison['prediction_changes']}"
        )

        print("\nPer class:")

        for label in LABELS:
            row = metrics["per_class"][label]

            print(
                f"  {label:16}"
                f" support={int(row['support']):2d}"
                f" predicted="
                f"{metrics[
                    'predicted_counts'
                ][label]:2d}"
                f" precision={row['precision']:.3f}"
                f" recall={row['recall']:.3f}"
                f" f1={row['f1-score']:.3f}"
            )

        print("\nDifficulty:")

        for difficulty in ["easy", "hard"]:
            row = metrics[
                "by_difficulty"
            ][difficulty]

            print(
                f"  {difficulty}: "
                f"n={row['count']}, "
                f"accuracy={row['accuracy']:.3f}, "
                f"mean confidence="
                f"{row['mean_confidence']:.3f}"
            )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()