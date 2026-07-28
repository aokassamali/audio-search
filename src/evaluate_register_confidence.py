import json
from statistics import mean, median

from sklearn.metrics import roc_auc_score

from src.config import load_settings
from src.run_register_multimodal_arm import (
    MultimodalArmArtifact,
)
from src.run_register_text_arm import (
    TextArmArtifact,
)


def summarize_confidence(
    items,
    prediction_field: str,
) -> dict:
    rows = []

    for item in items:
        prediction = getattr(
            item,
            prediction_field,
        )

        if prediction is None:
            continue

        rows.append(
            {
                "turn_id": item.turn_id,
                "difficulty": item.difficulty,
                "confidence": (
                    prediction.confidence
                ),
            }
        )

    easy_confidences = [
        row["confidence"]
        for row in rows
        if row["difficulty"] == "easy"
    ]

    hard_confidences = [
        row["confidence"]
        for row in rows
        if row["difficulty"] == "hard"
    ]

    hard_labels = [
        1 if row["difficulty"] == "hard" else 0
        for row in rows
    ]

    uncertainty_scores = [
        1.0 - row["confidence"]
        for row in rows
    ]

    difficulty_auc = roc_auc_score(
        hard_labels,
        uncertainty_scores,
    )

    ranked_rows = sorted(
        rows,
        key=lambda row: row["confidence"],
    )

    bottom_quartile_count = max(
        1,
        len(ranked_rows) // 4,
    )

    bottom_quartile = ranked_rows[
        :bottom_quartile_count
    ]

    hard_in_bottom_quartile = sum(
        row["difficulty"] == "hard"
        for row in bottom_quartile
    )

    return {
        "count": len(rows),
        "easy_count": len(easy_confidences),
        "hard_count": len(hard_confidences),
        "easy_mean_confidence": mean(
            easy_confidences
        ),
        "hard_mean_confidence": mean(
            hard_confidences
        ),
        "easy_median_confidence": median(
            easy_confidences
        ),
        "hard_median_confidence": median(
            hard_confidences
        ),
        "difficulty_auc_from_uncertainty": (
            difficulty_auc
        ),
        "bottom_quartile_count": (
            bottom_quartile_count
        ),
        "hard_in_bottom_quartile": (
            hard_in_bottom_quartile
        ),
        "bottom_quartile_hard_rate": (
            hard_in_bottom_quartile
            / bottom_quartile_count
        ),
        "overall_hard_rate": (
            len(hard_confidences)
            / len(rows)
        ),
    }


def main() -> None:
    settings = load_settings()
    eval_dir = settings.paths.eval_dir

    with (
        eval_dir
        / "register_text_predictions.json"
    ).open("r", encoding="utf-8") as file:
        text_artifact = (
            TextArmArtifact.model_validate(
                json.load(file)
            )
        )

    with (
        eval_dir
        / "register_multimodal_predictions.json"
    ).open("r", encoding="utf-8") as file:
        multimodal_artifact = (
            MultimodalArmArtifact.model_validate(
                json.load(file)
            )
        )

    results = {
        "text_permissive": summarize_confidence(
            text_artifact.items,
            "permissive_prediction",
        ),
        "text_strict": summarize_confidence(
            text_artifact.items,
            "strict_prediction",
        ),
        "multimodal_permissive": (
            summarize_confidence(
                multimodal_artifact.items,
                "permissive_prediction",
            )
        ),
        "multimodal_strict": (
            summarize_confidence(
                multimodal_artifact.items,
                "strict_prediction",
            )
        ),
    }

    output_path = (
        eval_dir
        / "register_confidence_metrics.json"
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

    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}")

        print(
            "Mean confidence:"
            f"\n  easy: "
            f"{metrics['easy_mean_confidence']:.3f}"
            f"\n  hard: "
            f"{metrics['hard_mean_confidence']:.3f}"
        )

        print(
            "Median confidence:"
            f"\n  easy: "
            f"{metrics['easy_median_confidence']:.3f}"
            f"\n  hard: "
            f"{metrics['hard_median_confidence']:.3f}"
        )

        print(
            "Difficulty AUROC from uncertainty: "
            f"{metrics[
                'difficulty_auc_from_uncertainty'
            ]:.3f}"
        )

        print(
            "Hard-turn rate:"
            f"\n  overall: "
            f"{metrics['overall_hard_rate']:.1%}"
            f"\n  lowest-confidence quartile: "
            f"{metrics[
                'bottom_quartile_hard_rate'
            ]:.1%}"
        )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()