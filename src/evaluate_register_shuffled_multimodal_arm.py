import json

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from src.config import load_settings
from src.run_register_multimodal_arm import (
    MultimodalArmArtifact,
)
from src.run_register_shuffled_multimodal_arm import (
    ShuffledMultimodalArtifact,
)
from src.run_register_text_arm import (
    TextArmArtifact,
)


LABELS = [
    "assertion",
    "hypothetical",
    "question",
    "characterization",
]


def evaluate_variant(
    shuffled: ShuffledMultimodalArtifact,
    real: MultimodalArmArtifact,
    text_only: TextArmArtifact,
    variant: str,
) -> dict:
    real_lookup = {
        item.turn_id: item
        for item in real.items
    }

    text_lookup = {
        item.turn_id: item
        for item in text_only.items
    }

    gold_labels = []
    shuffled_labels = []
    real_labels = []
    text_labels = []

    for shuffled_item in shuffled.items:
        real_item = real_lookup[
            shuffled_item.turn_id
        ]

        text_item = text_lookup[
            shuffled_item.turn_id
        ]

        if variant == "permissive":
            gold = (
                shuffled_item.gold_permissive
            )

            shuffled_prediction = (
                shuffled_item
                .permissive_prediction
            )

            real_prediction = (
                real_item
                .permissive_prediction
            )

            text_prediction = (
                text_item
                .permissive_prediction
            )

        else:
            gold = shuffled_item.gold_strict

            shuffled_prediction = (
                shuffled_item.strict_prediction
            )

            real_prediction = (
                real_item.strict_prediction
            )

            text_prediction = (
                text_item.strict_prediction
            )

        if (
            shuffled_prediction is None
            or real_prediction is None
            or text_prediction is None
        ):
            continue

        gold_labels.append(gold)
        shuffled_labels.append(
            shuffled_prediction.label
        )
        real_labels.append(
            real_prediction.label
        )
        text_labels.append(
            text_prediction.label
        )

    shuffled_fixes = 0
    shuffled_regressions = 0
    shuffled_changes = 0

    real_only_correct = 0
    shuffled_only_correct = 0
    both_correct = 0
    both_wrong = 0

    for (
        gold,
        shuffled_label,
        real_label,
        text_label,
    ) in zip(
        gold_labels,
        shuffled_labels,
        real_labels,
        text_labels,
    ):
        shuffled_correct = (
            shuffled_label == gold
        )

        real_correct = real_label == gold
        text_correct = text_label == gold

        if shuffled_label != text_label:
            shuffled_changes += 1

        if (
            not text_correct
            and shuffled_correct
        ):
            shuffled_fixes += 1

        elif (
            text_correct
            and not shuffled_correct
        ):
            shuffled_regressions += 1

        if real_correct and not shuffled_correct:
            real_only_correct += 1

        elif shuffled_correct and not real_correct:
            shuffled_only_correct += 1

        elif real_correct and shuffled_correct:
            both_correct += 1

        else:
            both_wrong += 1

    return {
        "count": len(gold_labels),
        "text_only_accuracy": accuracy_score(
            gold_labels,
            text_labels,
        ),
        "real_prosody_accuracy": accuracy_score(
            gold_labels,
            real_labels,
        ),
        "shuffled_prosody_accuracy": (
            accuracy_score(
                gold_labels,
                shuffled_labels,
            )
        ),
        "real_prosody_macro_f1": f1_score(
            gold_labels,
            real_labels,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "shuffled_prosody_macro_f1": (
            f1_score(
                gold_labels,
                shuffled_labels,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "shuffled_vs_text": {
            "fixes": shuffled_fixes,
            "regressions": (
                shuffled_regressions
            ),
            "prediction_changes": (
                shuffled_changes
            ),
        },
        "real_vs_shuffled": {
            "real_only_correct": (
                real_only_correct
            ),
            "shuffled_only_correct": (
                shuffled_only_correct
            ),
            "both_correct": both_correct,
            "both_wrong": both_wrong,
        },
    }


def main() -> None:
    settings = load_settings()
    eval_dir = settings.paths.eval_dir

    with (
        eval_dir
        / "register_multimodal_shuffled_predictions.json"
    ).open("r", encoding="utf-8") as file:
        shuffled = (
            ShuffledMultimodalArtifact
            .model_validate(
                json.load(file)
            )
        )

    with (
        eval_dir
        / "register_multimodal_predictions.json"
    ).open("r", encoding="utf-8") as file:
        real = (
            MultimodalArmArtifact
            .model_validate(
                json.load(file)
            )
        )

    with (
        eval_dir
        / "register_text_predictions.json"
    ).open("r", encoding="utf-8") as file:
        text_only = (
            TextArmArtifact.model_validate(
                json.load(file)
            )
        )

    results = {
        variant: evaluate_variant(
            shuffled=shuffled,
            real=real,
            text_only=text_only,
            variant=variant,
        )
        for variant in [
            "permissive",
            "strict",
        ]
    }

    output_path = (
        eval_dir
        / "register_multimodal_shuffled_metrics.json"
    )

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
            "real_vs_shuffled"
        ]

        placebo = metrics[
            "shuffled_vs_text"
        ]

        print(f"\n{variant.upper()}")

        print(
            "Accuracy:"
            f"\n  text-only: "
            f"{metrics['text_only_accuracy']:.3f}"
            f"\n  real prosody: "
            f"{metrics['real_prosody_accuracy']:.3f}"
            f"\n  shuffled prosody: "
            f"{metrics[
                'shuffled_prosody_accuracy'
            ]:.3f}"
        )

        print(
            "\nMacro F1:"
            f"\n  real prosody: "
            f"{metrics[
                'real_prosody_macro_f1'
            ]:.3f}"
            f"\n  shuffled prosody: "
            f"{metrics[
                'shuffled_prosody_macro_f1'
            ]:.3f}"
        )

        print(
            "\nShuffled vs text-only:"
            f"\n  fixes: {placebo['fixes']}"
            f"\n  regressions: "
            f"{placebo['regressions']}"
            f"\n  prediction changes: "
            f"{placebo['prediction_changes']}"
        )

        print(
            "\nReal vs shuffled:"
            f"\n  real only correct: "
            f"{comparison[
                'real_only_correct'
            ]}"
            f"\n  shuffled only correct: "
            f"{comparison[
                'shuffled_only_correct'
            ]}"
            f"\n  both correct: "
            f"{comparison['both_correct']}"
            f"\n  both wrong: "
            f"{comparison['both_wrong']}"
        )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()