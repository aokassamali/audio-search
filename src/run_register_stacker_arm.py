import json

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
from src.run_register_compact_prosody_arm import (
    COMPACT_FEATURES,
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


class StackerItem(BaseModel):
    turn_id: str
    speaker: str
    gold_label: RegisterLabel

    direct_text_label: RegisterLabel
    direct_text_confidence: float

    text_stack_label: RegisterLabel
    text_stack_confidence: float

    text_prosody_stack_label: RegisterLabel
    text_prosody_stack_confidence: float

    fold: int


class StackerArtifact(BaseModel):
    schema_version: str = "1.0"
    taxonomy_variant: str = "permissive"

    prosody_feature_names: list[str] = Field(
        default_factory=list
    )

    items: list[StackerItem] = Field(
        default_factory=list
    )


def create_model() -> Pipeline:
    return Pipeline(
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


def predict_with_confidence(
    model: Pipeline,
    X: np.ndarray,
) -> tuple[list[str], list[float]]:
    probabilities = model.predict_proba(X)

    classifier = model.named_steps[
        "classifier"
    ]

    classes = [
        str(label)
        for label in classifier.classes_
    ]

    predicted_labels = []
    confidences = []

    for probability_row in probabilities:
        best_index = int(
            np.argmax(probability_row)
        )

        predicted_labels.append(
            classes[best_index]
        )

        confidences.append(
            float(probability_row[best_index])
        )

    return predicted_labels, confidences


def main() -> None:
    settings = load_settings()
    eval_dir = settings.paths.eval_dir

    with (
        eval_dir
        / "register_features.json"
    ).open("r", encoding="utf-8") as file:
        features = (
            RegisterFeaturesArtifact.model_validate(
                json.load(file)
            )
        )

    with (
        eval_dir
        / "register_text_predictions.json"
    ).open("r", encoding="utf-8") as file:
        text_predictions = (
            TextArmArtifact.model_validate(
                json.load(file)
            )
        )

    text_lookup = {
        item.turn_id: item
        for item in text_predictions.items
    }

    aligned_rows = []

    for feature_item in features.items:
        text_item = text_lookup.get(
            feature_item.turn_id
        )

        if text_item is None:
            continue

        prediction = (
            text_item.permissive_prediction
        )

        if prediction is None:
            continue

        aligned_rows.append(
            (
                feature_item,
                prediction,
            )
        )

    text_feature_rows = []
    prosody_feature_rows = []

    gold_labels = []
    speakers = []

    for feature_item, prediction in aligned_rows:
        label_one_hot = [
            1.0
            if prediction.label == label
            else 0.0
            for label in LABELS
        ]

        text_features = (
            label_one_hot
            + [prediction.confidence]
        )

        prosody_features = [
            feature_item.features[name]
            for name in COMPACT_FEATURES
        ]

        text_feature_rows.append(
            text_features
        )

        prosody_feature_rows.append(
            text_features
            + prosody_features
        )

        gold_labels.append(
            feature_item.label_permissive
        )

        speakers.append(
            feature_item.speaker
        )

    X_text = np.asarray(
        text_feature_rows,
        dtype=float,
    )

    X_text_prosody = np.asarray(
        prosody_feature_rows,
        dtype=float,
    )

    y = np.asarray(gold_labels)
    groups = np.asarray(speakers)

    splitter = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=23,
    )

    text_stack_labels = [None] * len(y)
    text_stack_confidences = [None] * len(y)

    multimodal_labels = [None] * len(y)
    multimodal_confidences = [None] * len(y)

    fold_assignments = [None] * len(y)

    for fold, (
        train_indices,
        test_indices,
    ) in enumerate(
        splitter.split(
            X_text,
            y,
            groups=groups,
        ),
        start=1,
    ):
        print(
            f"Running fold {fold}: "
            f"train={len(train_indices)}, "
            f"test={len(test_indices)}"
        )

        text_model = create_model()
        multimodal_model = create_model()

        text_model.fit(
            X_text[train_indices],
            y[train_indices],
        )

        multimodal_model.fit(
            X_text_prosody[train_indices],
            y[train_indices],
        )

        (
            fold_text_labels,
            fold_text_confidences,
        ) = predict_with_confidence(
            text_model,
            X_text[test_indices],
        )

        (
            fold_multimodal_labels,
            fold_multimodal_confidences,
        ) = predict_with_confidence(
            multimodal_model,
            X_text_prosody[test_indices],
        )

        for row_index, item_index in enumerate(
            test_indices
        ):
            text_stack_labels[item_index] = (
                fold_text_labels[row_index]
            )

            text_stack_confidences[item_index] = (
                fold_text_confidences[row_index]
            )

            multimodal_labels[item_index] = (
                fold_multimodal_labels[row_index]
            )

            multimodal_confidences[item_index] = (
                fold_multimodal_confidences[
                    row_index
                ]
            )

            fold_assignments[item_index] = fold

    result_items = []

    for index, (
        feature_item,
        direct_prediction,
    ) in enumerate(aligned_rows):
        result_items.append(
            StackerItem(
                turn_id=feature_item.turn_id,
                speaker=feature_item.speaker,
                gold_label=(
                    feature_item.label_permissive
                ),
                direct_text_label=(
                    direct_prediction.label
                ),
                direct_text_confidence=(
                    direct_prediction.confidence
                ),
                text_stack_label=(
                    text_stack_labels[index]
                ),
                text_stack_confidence=(
                    text_stack_confidences[index]
                ),
                text_prosody_stack_label=(
                    multimodal_labels[index]
                ),
                text_prosody_stack_confidence=(
                    multimodal_confidences[index]
                ),
                fold=fold_assignments[index],
            )
        )

    artifact = StackerArtifact(
        prosody_feature_names=COMPACT_FEATURES,
        items=result_items,
    )

    output_path = (
        eval_dir
        / "register_stacker_predictions.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            artifact.model_dump(mode="json"),
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved: {output_path}")
    print(f"Items: {len(result_items)}")
    print(
        "Prosody features: "
        f"{len(COMPACT_FEATURES)}"
    )


if __name__ == "__main__":
    main()