import json
from collections import Counter
from statistics import mean, pstdev

import numpy as np
from sentence_transformers import (
    SentenceTransformer,
)
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

    canonical_path = (
        eval_dir
        / "canonical_speaker_groups_v24.json"
    )

    embeddings_path = (
        eval_dir
        / "text_embeddings_v24.npy"
    )

    manifest_path = (
        eval_dir
        / "text_embeddings_v24_manifest.json"
    )

    output_path = (
        eval_dir
        / "text_probe_canonical_v24.json"
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

    texts = [
        item.text
        for item in gold_artifact.items
    ]

    if embeddings_path.exists():
        embeddings = np.load(
            embeddings_path,
            allow_pickle=False,
        )

        print(
            "Loaded cached text embeddings."
        )
    else:
        model_name = (
            settings.models.embedding_model
        )

        print(
            f"Encoding {len(texts)} turns"
            f"\nModel: {model_name}"
        )

        encoder = SentenceTransformer(
            model_name
        )

        embeddings = encoder.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        np.save(
            embeddings_path,
            embeddings,
            allow_pickle=False,
        )

        manifest = {
            "schema_version": "1.0",
            "model": model_name,
            "normalized": True,
            "embedding_shape": list(
                embeddings.shape
            ),
            "items": [
                {
                    "row_index": index,
                    "turn_id": item.turn_id,
                }
                for index, item
                in enumerate(
                    gold_artifact.items
                )
            ],
        }

        with manifest_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                manifest,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"Saved embeddings: "
            f"{embeddings_path}"
        )

    if embeddings.shape[0] != len(
        gold_artifact.items
    ):
        print(
            "Text embedding row count does "
            "not match the gold artifact."
        )
        return

    if not np.isfinite(
        embeddings
    ).all():
        print(
            "Text embeddings contain "
            "non-finite values."
        )
        return

    labels = np.asarray(
        [
            str(item.label_permissive)
            for item in gold_artifact.items
        ]
    )

    difficulties = np.asarray(
        [
            str(item.difficulty)
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

    outer_splitter = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=24,
    )

    outer_splits = list(
        outer_splitter.split(
            embeddings,
            labels,
            groups=groups,
        )
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

        print(
            f"Fold {fold}: "
            f"test={len(test_indices)} "
            f"{dict(test_counts)}"
        )

    scorer = make_scorer(
        f1_score,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

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
                random_state=2400 + fold,
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
            embeddings[train_indices],
            labels[train_indices],
            groups=groups[train_indices],
        )

        fold_predictions = search.predict(
            embeddings[test_indices]
        )

        fold_probabilities = (
            search.predict_proba(
                embeddings[test_indices]
            )
        )

        predictions[test_indices] = (
            fold_predictions
        )

        confidences[test_indices] = (
            np.max(
                fold_probabilities,
                axis=1,
            )
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
                    fold_metrics["accuracy"]
                ),
                "macro_f1": (
                    fold_metrics["macro_f1"]
                ),
            }
        )

        print(
            f"\nFold {fold}"
            f"\n  C: {selected_c:g}"
            f"\n  accuracy: "
            f"{fold_metrics['accuracy']:.3f}"
            f"\n  macro F1: "
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

    majority_accuracy = (
        max(
            Counter(labels).values()
        )
        / len(labels)
    )

    output = {
        "schema_version": "1.0",
        "model": (
            settings.models.embedding_model
        ),
        "task": (
            "four-class permissive "
            "speech-act classification"
        ),
        "modality": "text",
        "evaluation_status": (
            "canonical cross-case speaker groups"
        ),
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
                "source_key": item.source_key,
                "speaker": item.speaker,
                "canonical_group": (
                    str(groups[index])
                ),
                "gold_label": (
                    str(labels[index])
                ),
                "predicted_label": (
                    str(predictions[index])
                ),
                "confidence": float(
                    confidences[index]
                ),
                "difficulty": (
                    str(difficulties[index])
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

    print("\nTEXT PROBE COMPLETE")
    print(
        "Majority accuracy: "
        f"{majority_accuracy:.3f}"
    )
    print(
        "Accuracy: "
        f"{overall['accuracy']:.3f}"
    )
    print(
        "Macro F1: "
        f"{overall['macro_f1']:.3f}"
    )
    print(
        "Easy macro F1: "
        f"{easy_metrics['macro_f1']:.3f}"
    )
    print(
        "Hard macro F1: "
        f"{hard_metrics['macro_f1']:.3f}"
    )

    print("\nPer-class F1")

    for label in LABELS:
        class_result = (
            overall["per_class"][label]
        )

        print(
            f"  {label}: "
            f"{class_result['f1']:.3f} "
            f"(n={class_result['support']})"
        )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()