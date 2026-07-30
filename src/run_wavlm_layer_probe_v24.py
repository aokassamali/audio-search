import json
from collections import Counter
from statistics import mean, pstdev

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import load_settings
from src.speech_act_gold import (
    SpeechActGoldArtifact,
)


LABELS = [
    "assertion",
    "question",
    "characterization",
    "hypothetical",
]

C_VALUES = [
    0.001,
    0.01,
    0.1,
    1.0,
]


def create_pipeline() -> Pipeline:
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
                    random_state=24,
                ),
            ),
        ]
    )


def calculate_metrics(
    gold: np.ndarray,
    predicted: np.ndarray,
) -> dict:
    report = classification_report(
        gold,
        predicted,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(
            accuracy_score(
                gold,
                predicted,
            )
        ),
        "macro_f1": float(
            f1_score(
                gold,
                predicted,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class": {
            label: {
                "precision": float(
                    report[label]["precision"]
                ),
                "recall": float(
                    report[label]["recall"]
                ),
                "f1": float(
                    report[label]["f1-score"]
                ),
                "support": int(
                    report[label]["support"]
                ),
            }
            for label in LABELS
        },
    }


def main() -> None:
    settings = load_settings()
    eval_dir = settings.paths.eval_dir

    gold_path = (
        eval_dir
        / "speech_act_gold_v24.json"
    )

    embeddings_path = (
        eval_dir
        / "wavlm_base_plus_v24.npy"
    )

    manifest_path = (
        eval_dir
        / "wavlm_base_plus_v24_manifest.json"
    )

    output_path = (
        eval_dir
        / "wavlm_layer_probe_canonical_v24.json"
    )

    with gold_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        gold_artifact = (
            SpeechActGoldArtifact.model_validate(
                json.load(file)
            )
        )

    canonical_path = (
        eval_dir
        / "canonical_speaker_groups_v24.json"
    )

    with canonical_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        canonical_artifact = json.load(file)

    if not canonical_artifact[
        "validation_passed"
    ]:
        print(
            "Canonical speaker validation failed."
        )
        return

    canonical_mapping = (
        canonical_artifact["mapping"]
    )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    embeddings = np.load(
        embeddings_path,
        allow_pickle=False,
    )

    expected_shape = (
        len(gold_artifact.items),
        13,
        768,
    )

    if embeddings.shape != expected_shape:
        print(
            "Unexpected embedding shape."
            f"\nExpected: {expected_shape}"
            f"\nFound: {embeddings.shape}"
        )
        return

    manifest_turn_ids = [
        item["turn_id"]
        for item in manifest["items"]
    ]

    gold_turn_ids = [
        item.turn_id
        for item in gold_artifact.items
    ]

    if manifest_turn_ids != gold_turn_ids:
        print(
            "Manifest and gold-set order "
            "do not match."
        )
        return

    labels = np.asarray(
        [
            item.label_permissive
            for item in gold_artifact.items
        ]
    )

    provisional_groups = np.asarray(
        [
            f"{item.source_key}:"
            f"{item.speaker}"
            for item in gold_artifact.items
        ]
    )

    missing_groups = sorted(
        set(provisional_groups)
        - set(canonical_mapping)
    )

    if missing_groups:
        print(
            "Missing canonical mappings:"
        )

        for group in missing_groups:
            print(f"  {group}")

        return

    groups = np.asarray(
        [
            canonical_mapping[group]
            for group in provisional_groups
        ]
    )

    difficulties = np.asarray(
        [
            item.difficulty
            for item in gold_artifact.items
        ]
    )

    sources = np.asarray(
        [
            item.source_key
            for item in gold_artifact.items
        ]
    )

    outer_splitter = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=24,
    )

    outer_splits = list(
        outer_splitter.split(
            np.zeros(
                (len(labels), 1),
                dtype=float,
            ),
            labels,
            groups=groups,
        )
    )

    scorer = make_scorer(
        f1_score,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    print(
        "Canonical speaker groups: "
        f"{len(set(groups))}"
    )

    for fold, (
        train_indices,
        test_indices,
    ) in enumerate(
        outer_splits,
        start=1,
    ):
        test_counts = Counter(
            labels[test_indices]
        )

        missing_labels = [
            label
            for label in LABELS
            if test_counts[label] == 0
        ]

        print(
            f"Fold {fold}: "
            f"test={len(test_indices)} "
            f"{dict(test_counts)}"
        )

        if missing_labels:
            print(
                "Missing test labels: "
                + ", ".join(missing_labels)
            )
            return

    layer_results = []

    for layer_index in range(
        embeddings.shape[1]
    ):
        X = embeddings[
            :,
            layer_index,
            :,
        ]

        predictions = np.empty(
            len(labels),
            dtype=object,
        )

        confidences = np.zeros(
            len(labels),
            dtype=float,
        )

        fold_assignments = np.zeros(
            len(labels),
            dtype=int,
        )

        selected_c_values = []
        fold_results = []

        print(
            f"\nLayer {layer_index}/"
            f"{embeddings.shape[1] - 1}"
        )

        for fold, (
            train_indices,
            test_indices,
        ) in enumerate(
            outer_splits,
            start=1,
        ):
            inner_splitter = (
                StratifiedGroupKFold(
                    n_splits=3,
                    shuffle=True,
                    random_state=(
                        2400 + fold
                    ),
                )
            )

            search = GridSearchCV(
                estimator=create_pipeline(),
                param_grid={
                    "classifier__C": C_VALUES,
                },
                scoring=scorer,
                cv=inner_splitter,
                n_jobs=-1,
                refit=True,
            )

            search.fit(
                X[train_indices],
                labels[train_indices],
                groups=groups[
                    train_indices
                ],
            )

            fold_predictions = (
                search.predict(
                    X[test_indices]
                )
            )

            fold_probabilities = (
                search.predict_proba(
                    X[test_indices]
                )
            )

            fold_confidences = np.max(
                fold_probabilities,
                axis=1,
            )

            predictions[test_indices] = (
                fold_predictions
            )

            confidences[test_indices] = (
                fold_confidences
            )

            fold_assignments[
                test_indices
            ] = fold

            selected_c = float(
                search.best_params_[
                    "classifier__C"
                ]
            )

            selected_c_values.append(
                selected_c
            )

            fold_metrics = calculate_metrics(
                labels[test_indices],
                fold_predictions,
            )

            fold_results.append(
                {
                    "fold": fold,
                    "train_size": int(
                        len(train_indices)
                    ),
                    "test_size": int(
                        len(test_indices)
                    ),
                    "selected_c": selected_c,
                    "accuracy": (
                        fold_metrics[
                            "accuracy"
                        ]
                    ),
                    "macro_f1": (
                        fold_metrics[
                            "macro_f1"
                        ]
                    ),
                }
            )

            print(
                f"  fold {fold}: "
                f"C={selected_c:g} "
                f"accuracy="
                f"{fold_metrics['accuracy']:.3f} "
                f"macro_f1="
                f"{fold_metrics['macro_f1']:.3f}"
            )

        overall = calculate_metrics(
            labels,
            predictions,
        )

        easy_mask = (
            difficulties == "easy"
        )

        hard_mask = (
            difficulties == "hard"
        )

        easy_metrics = calculate_metrics(
            labels[easy_mask],
            predictions[easy_mask],
        )

        hard_metrics = calculate_metrics(
            labels[hard_mask],
            predictions[hard_mask],
        )

        fold_macro_f1 = [
            result["macro_f1"]
            for result in fold_results
        ]

        result = {
            "layer": layer_index,
            "overall": overall,
            "easy": easy_metrics,
            "hard": hard_metrics,
            "fold_macro_f1_mean": float(
                mean(fold_macro_f1)
            ),
            "fold_macro_f1_std": float(
                pstdev(fold_macro_f1)
            ),
            "selected_c_counts": dict(
                Counter(selected_c_values)
            ),
            "folds": fold_results,
            "predictions": [
                {
                    "turn_id": item.turn_id,
                    "source_key": (
                        item.source_key
                    ),
                    "speaker": item.speaker,
                    "provisional_group": (
                        groups[index]
                    ),
                    "gold_label": (
                        labels[index]
                    ),
                    "predicted_label": (
                        predictions[index]
                    ),
                    "confidence": float(
                        confidences[index]
                    ),
                    "difficulty": (
                        difficulties[index]
                    ),
                    "fold": int(
                        fold_assignments[index]
                    ),
                }
                for index, item
                in enumerate(
                    gold_artifact.items
                )
            ],
        }

        layer_results.append(result)

        print(
            f"  pooled accuracy: "
            f"{overall['accuracy']:.3f}"
        )

        print(
            f"  pooled macro F1: "
            f"{overall['macro_f1']:.3f}"
        )

    best_layer = max(
        layer_results,
        key=lambda result: (
            result["overall"]["macro_f1"]
        ),
    )

    majority_accuracy = (
        max(
            Counter(labels).values()
        )
        / len(labels)
    )

    output = {
        "schema_version": "1.0",
        "model": (
            "microsoft/wavlm-base-plus"
        ),
        "task": (
            "four-class permissive "
            "speech-act classification"
        ),
        "evaluation_status": (
            "canonical cross-case speaker groups"
        ),
        "group_definition": (
            "manually reviewed cross-case "
            "speaker identity components"
        ),
        "warning": None,
        "outer_split": (
            "4-fold StratifiedGroupKFold"
        ),
        "inner_selection": (
            "3-fold StratifiedGroupKFold "
            "over logistic-regression C"
        ),
        "c_values": C_VALUES,
        "majority_accuracy": float(
            majority_accuracy
        ),
        "best_layer_descriptive": int(
            best_layer["layer"]
        ),
        "best_macro_f1_descriptive": float(
            best_layer[
                "overall"
            ]["macro_f1"]
        ),
        "layers": layer_results,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nLAYER SWEEP COMPLETE")
    print(
        "Majority accuracy: "
        f"{majority_accuracy:.3f}"
    )
    print(
        "Best descriptive layer: "
        f"{best_layer['layer']}"
    )
    print(
        "Best descriptive macro F1: "
        f"{best_layer['overall']['macro_f1']:.3f}"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()