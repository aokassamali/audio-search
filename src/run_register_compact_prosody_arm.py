import json

import numpy as np
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
from src.run_register_prosody_arm import (
    ProsodyArmArtifact,
    ProsodyArmItem,
)


COMPACT_FEATURES = [
    "F0semitoneFrom27.5Hz_sma3nz_amean",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    "loudness_sma3_amean",
    "loudness_sma3_stddevNorm",
    "speaking_rate_wps",
    "silence_proportion",
    "pause_count",
]


def main() -> None:
    settings = load_settings()

    input_path = (
        settings.paths.eval_dir
        / "register_features.json"
    )

    output_path = (
        settings.paths.eval_dir
        / "register_prosody_compact_predictions.json"
    )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        artifact = (
            RegisterFeaturesArtifact.model_validate(
                json.load(file)
            )
        )

    X = np.asarray(
        [
            [
                item.features[name]
                for name in COMPACT_FEATURES
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

        probabilities = model.predict_proba(
            X[test_indices]
        )

        classes = [
            str(label)
            for label in model.named_steps[
                "classifier"
            ].classes_
        ]

        for row_index, item_index in enumerate(
            test_indices
        ):
            probability_map = {
                label: float(
                    probabilities[
                        row_index,
                        class_index,
                    ]
                )
                for class_index, label
                in enumerate(classes)
            }

            predicted_label = max(
                probability_map,
                key=probability_map.get,
            )

            item = artifact.items[item_index]

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

    result = ProsodyArmArtifact(
        model=(
            "StandardScaler + balanced "
            "L2 logistic regression; "
            "compact prosody features"
        ),
        feature_names=COMPACT_FEATURES,
        items=[
            prediction
            for prediction in predictions
            if prediction is not None
        ],
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
        f"{len(COMPACT_FEATURES)}"
    )
    print(
        f"Predictions: "
        f"{len(result.items)}"
    )


if __name__ == "__main__":
    main()