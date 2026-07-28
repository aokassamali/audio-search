import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import load_settings
from src.extract_register_features import (
    RegisterFeaturesArtifact,
)
from src.register_gold import RegisterLabel


EXCLUDED_FEATURES = {
    "duration_seconds",
    "word_count",
}


class ProsodyArmItem(BaseModel):
    turn_id: str
    speaker: str
    gold_label: RegisterLabel

    predicted_label: RegisterLabel
    confidence: float
    probabilities: dict[str, float]

    fold: int


class ProsodyArmArtifact(BaseModel):
    schema_version: str = "1.0"
    taxonomy_variant: str = "permissive"
    model: str = (
        "StandardScaler + balanced "
        "L2 logistic regression"
    )

    feature_names: list[str] = Field(
        default_factory=list
    )

    items: list[ProsodyArmItem] = Field(
        default_factory=list
    )


def run_prosody_arm() -> None:
    settings = load_settings()

    feature_path = (
        settings.paths.eval_dir
        / "register_features.json"
    )

    output_path = (
        settings.paths.eval_dir
        / "register_prosody_predictions.json"
    )

    with feature_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        artifact = (
            RegisterFeaturesArtifact.model_validate(
                json.load(file)
            )
        )

    feature_names = sorted(
        name
        for name in artifact.items[0].features
        if name not in EXCLUDED_FEATURES
    )

    X = np.asarray(
        [
            [
                item.features[name]
                for name in feature_names
            ]
            for item in artifact.items
        ],
        dtype=float,
    )

    labels = np.asarray(
        [
            item.label_permissive
            for item in artifact.items
        ]
    )

    speakers = np.asarray(
        [
            item.speaker
            for item in artifact.items
        ]
    )

    splitter = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=23,
    )

    predictions: list[
        ProsodyArmItem | None
    ] = [None] * len(artifact.items)

    for fold, (
        train_indices,
        test_indices,
    ) in enumerate(
        splitter.split(
            X,
            labels,
            groups=speakers,
        ),
        start=1,
    ):
        print(
            f"Running fold {fold}: "
            f"train={len(train_indices)}, "
            f"test={len(test_indices)}"
        )

        model = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=5000,
                        random_state=23,
                    ),
                ),
            ]
        )

        model.fit(
            X[train_indices],
            labels[train_indices],
        )

        fold_probabilities = (
            model.predict_proba(
                X[test_indices]
            )
        )

        classifier = model.named_steps[
            "classifier"
        ]

        class_names = [
            str(label)
            for label in classifier.classes_
        ]

        for row_index, item_index in enumerate(
            test_indices
        ):
            probability_values = (
                fold_probabilities[row_index]
            )

            probability_map = {
                label: float(
                    probability_values[
                        class_index
                    ]
                )
                for class_index, label
                in enumerate(class_names)
            }

            predicted_label = max(
                probability_map,
                key=probability_map.get,
            )

            item = artifact.items[
                item_index
            ]

            predictions[item_index] = (
                ProsodyArmItem(
                    turn_id=item.turn_id,
                    speaker=item.speaker,
                    gold_label=(
                        item.label_permissive
                    ),
                    predicted_label=(
                        predicted_label
                    ),
                    confidence=(
                        probability_map[
                            predicted_label
                        ]
                    ),
                    probabilities=(
                        probability_map
                    ),
                    fold=fold,
                )
            )

    completed_predictions = [
        prediction
        for prediction in predictions
        if prediction is not None
    ]

    result = ProsodyArmArtifact(
        feature_names=feature_names,
        items=completed_predictions,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result.model_dump(mode="json"),
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved: {output_path}")
    print(
        f"Features used: "
        f"{len(feature_names)}"
    )
    print(
        f"Predictions: "
        f"{len(completed_predictions)}"
    )


if __name__ == "__main__":
    run_prosody_arm()