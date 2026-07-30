import json
from collections import Counter
from statistics import mean, pstdev

import numpy as np
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
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
from sklearn.pipeline import FeatureUnion, Pipeline

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
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]


def create_pipeline() -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    max_features=20_000,
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=20_000,
                ),
            ),
        ]
    )

    return Pipeline(
        [
            (
                "features",
                features,
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="lbfgs",
                    l1_ratio=0.0,
                    max_iter=5000,
                    tol=1e-4,
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

    output_path = (
        eval_dir
        / "tfidf_probe_canonical_v24.json"
    )

    with gold_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        gold = (
            SpeechActGoldArtifact.model_validate(
                json.load(file)
            )
        )

    with canonical_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        canonical = json.load(file)

    if not canonical["validation_passed"]:
        print(
            "Canonical speaker validation failed."
        )
        return

    canonical_mapping = canonical["mapping"]

    texts = np.asarray(
        [
            item.text
            for item in gold.items
        ],
        dtype=object,
    )

    labels = np.asarray(
        [
            str(item.label_permissive)
            for item in gold.items
        ]
    )

    difficulties = np.asarray(
        [
            str(item.difficulty)
            for item in gold.items
        ]
    )

    provisional_groups = np.asarray(
        [
            f"{item.source_key}:"
            f"{item.speaker}"
            for item in gold.items
        ]
    )

    missing_groups = sorted(
        set(provisional_groups)
        - set(canonical_mapping)
    )

    if missing_groups:
        print("Missing canonical groups.")

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
            texts,
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

    fold_results = []
    selected_c_values = []

    for fold, (
        train_indices,
        test_indices,
    ) in enumerate(
        outer_splits,
        start=1,
    ):
        inner_splitter = StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=2400 + fold,
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
            texts[train_indices],
            labels[train_indices],
            groups=groups[train_indices],
        )

        fold_predictions = search.predict(
            texts[test_indices]
        )

        fold_probabilities = (
            search.predict_proba(
                texts[test_indices]
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

        metrics = calculate_metrics(
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
                    metrics["accuracy"]
                ),
                "macro_f1": (
                    metrics["macro_f1"]
                ),
            }
        )

        print(
            f"\nFold {fold}"
            f"\n  C: {selected_c:g}"
            f"\n  accuracy: "
            f"{metrics['accuracy']:.3f}"
            f"\n  macro F1: "
            f"{metrics['macro_f1']:.3f}"
        )

    overall = calculate_metrics(
        labels,
        predictions,
    )

    easy_mask = difficulties == "easy"
    hard_mask = difficulties == "hard"

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
        max(Counter(labels).values())
        / len(labels)
    )

    output = {
        "schema_version": "1.0",
        "model": (
            "word 1-2 gram and character "
            "3-5 gram TF-IDF with balanced "
            "logistic regression"
        ),
        "modality": "text",
        "evaluation_status": (
            "canonical cross-case speaker groups"
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
                "canonical_group": (
                    str(groups[index])
                ),
                "fold": int(
                    fold_assignments[index]
                ),
            }
            for index, item
            in enumerate(gold.items)
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

    print("\nTF-IDF PROBE COMPLETE")
    print(
        f"Majority accuracy: "
        f"{majority_accuracy:.3f}"
    )
    print(
        f"Accuracy: "
        f"{overall['accuracy']:.3f}"
    )
    print(
        f"Macro F1: "
        f"{overall['macro_f1']:.3f}"
    )
    print(
        f"Easy macro F1: "
        f"{easy_metrics['macro_f1']:.3f}"
    )
    print(
        f"Hard macro F1: "
        f"{hard_metrics['macro_f1']:.3f}"
    )

    print("\nPer-class F1")

    for label in LABELS:
        result = overall[
            "per_class"
        ][label]

        print(
            f"  {label}: "
            f"{result['f1']:.3f} "
            f"(n={result['support']})"
        )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()