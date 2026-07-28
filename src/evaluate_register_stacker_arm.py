import json

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

from src.config import load_settings
from src.run_register_stacker_arm import (
    StackerArtifact,
)


LABELS = [
    "assertion",
    "hypothetical",
    "question",
    "characterization",
]


def calculate_metrics(
    gold_labels: list[str],
    predicted_labels: list[str],
) -> dict:
    report = classification_report(
        gold_labels,
        predicted_labels,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(
            gold_labels,
            predicted_labels,
        ),
        "macro_f1": f1_score(
            gold_labels,
            predicted_labels,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "per_class": {
            label: report[label]
            for label in LABELS
        },
    }


def compare_predictions(
    gold_labels: list[str],
    baseline_labels: list[str],
    comparison_labels: list[str],
) -> dict:
    fixes = 0
    regressions = 0
    both_correct = 0
    both_wrong = 0
    prediction_changes = 0

    for gold, baseline, comparison in zip(
        gold_labels,
        baseline_labels,
        comparison_labels,
    ):
        baseline_correct = baseline == gold
        comparison_correct = comparison == gold

        if baseline != comparison:
            prediction_changes += 1

        if not baseline_correct and comparison_correct:
            fixes += 1
        elif baseline_correct and not comparison_correct:
            regressions += 1
        elif baseline_correct and comparison_correct:
            both_correct += 1
        else:
            both_wrong += 1

    return {
        "fixes": fixes,
        "regressions": regressions,
        "net_correct": fixes - regressions,
        "prediction_changes": prediction_changes,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
    }


def main() -> None:
    settings = load_settings()

    input_path = (
        settings.paths.eval_dir
        / "register_stacker_predictions.json"
    )

    output_path = (
        settings.paths.eval_dir
        / "register_stacker_metrics.json"
    )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        artifact = StackerArtifact.model_validate(
            json.load(file)
        )

    gold_labels = [
        item.gold_label
        for item in artifact.items
    ]

    direct_text_labels = [
        item.direct_text_label
        for item in artifact.items
    ]

    text_stack_labels = [
        item.text_stack_label
        for item in artifact.items
    ]

    text_prosody_labels = [
        item.text_prosody_stack_label
        for item in artifact.items
    ]

    results = {
        "direct_text": calculate_metrics(
            gold_labels,
            direct_text_labels,
        ),
        "text_stack": calculate_metrics(
            gold_labels,
            text_stack_labels,
        ),
        "text_prosody_stack": calculate_metrics(
            gold_labels,
            text_prosody_labels,
        ),
        "text_stack_vs_direct": (
            compare_predictions(
                gold_labels,
                direct_text_labels,
                text_stack_labels,
            )
        ),
        "prosody_vs_text_stack": (
            compare_predictions(
                gold_labels,
                text_stack_labels,
                text_prosody_labels,
            )
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
        )

    for model_name in [
        "direct_text",
        "text_stack",
        "text_prosody_stack",
    ]:
        metrics = results[model_name]

        print(f"\n{model_name.upper()}")
        print(
            f"Accuracy: "
            f"{metrics['accuracy']:.3f}"
        )
        print(
            f"Macro F1: "
            f"{metrics['macro_f1']:.3f}"
        )

        for label in LABELS:
            row = metrics["per_class"][label]

            print(
                f"  {label:16}"
                f" precision={row['precision']:.3f}"
                f" recall={row['recall']:.3f}"
                f" f1={row['f1-score']:.3f}"
            )

    comparison = results[
        "prosody_vs_text_stack"
    ]

    print(
        "\nTEXT + PROSODY VS TEXT STACK:"
        f"\n  fixes: {comparison['fixes']}"
        f"\n  regressions: "
        f"{comparison['regressions']}"
        f"\n  net correct: "
        f"{comparison['net_correct']}"
        f"\n  prediction changes: "
        f"{comparison['prediction_changes']}"
    )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()