import json
from collections import Counter
from statistics import mean, pstdev

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
)
from sklearn.pipeline import FeatureUnion
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
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

AUDIO_WEIGHTS = [
    0.0,
    0.03,
    0.1,
    0.3,
    1.0,
]

WAVLM_LAYER = 10


def create_text_vectorizer() -> FeatureUnion:
    return FeatureUnion(
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


def create_classifier(
    c_value: float,
) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=5000,
        tol=1e-4,
        random_state=24,
    )


def combine_features(
    text_features,
    audio_features: np.ndarray,
    audio_weight: float,
):
    if audio_weight == 0.0:
        return text_features

    weighted_audio = (
        audio_features * audio_weight
    )

    return hstack(
        [
            text_features,
            csr_matrix(weighted_audio),
        ],
        format="csr",
    )


def calculate_macro_f1(
    gold: np.ndarray,
    predicted: np.ndarray,
) -> float:
    return float(
        f1_score(
            gold,
            predicted,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
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
        "macro_f1": (
            calculate_macro_f1(
                gold,
                predicted,
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


def select_hyperparameters(
    texts: np.ndarray,
    audio: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
    outer_fold: int,
) -> tuple[float, float, list[dict]]:
    inner_labels = labels[train_indices]
    inner_groups = groups[train_indices]

    splitter = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=2400 + outer_fold,
    )

    inner_splits = list(
        splitter.split(
            np.zeros(
                (len(train_indices), 1),
                dtype=float,
            ),
            inner_labels,
            groups=inner_groups,
        )
    )

    score_lookup = {
        (
            c_value,
            audio_weight,
        ): []
        for c_value in C_VALUES
        for audio_weight in AUDIO_WEIGHTS
    }

    for (
        inner_train_relative,
        inner_validation_relative,
    ) in inner_splits:
        inner_train = train_indices[
            inner_train_relative
        ]

        inner_validation = train_indices[
            inner_validation_relative
        ]

        vectorizer = create_text_vectorizer()

        train_text_features = (
            vectorizer.fit_transform(
                texts[inner_train]
            )
        )

        validation_text_features = (
            vectorizer.transform(
                texts[inner_validation]
            )
        )

        audio_scaler = StandardScaler()

        train_audio = (
            audio_scaler.fit_transform(
                audio[inner_train]
            )
        )

        validation_audio = (
            audio_scaler.transform(
                audio[inner_validation]
            )
        )

        for audio_weight in AUDIO_WEIGHTS:
            train_features = combine_features(
                text_features=(
                    train_text_features
                ),
                audio_features=train_audio,
                audio_weight=audio_weight,
            )

            validation_features = (
                combine_features(
                    text_features=(
                        validation_text_features
                    ),
                    audio_features=(
                        validation_audio
                    ),
                    audio_weight=(
                        audio_weight
                    ),
                )
            )

            for c_value in C_VALUES:
                classifier = create_classifier(
                    c_value
                )

                classifier.fit(
                    train_features,
                    labels[inner_train],
                )

                validation_predictions = (
                    classifier.predict(
                        validation_features
                    )
                )

                score = calculate_macro_f1(
                    labels[inner_validation],
                    validation_predictions,
                )

                score_lookup[
                    (
                        c_value,
                        audio_weight,
                    )
                ].append(score)

    candidate_results = []

    for (
        c_value,
        audio_weight,
    ), scores in score_lookup.items():
        candidate_results.append(
            {
                "c_value": c_value,
                "audio_weight": audio_weight,
                "mean_macro_f1": float(
                    mean(scores)
                ),
                "fold_macro_f1": [
                    float(score)
                    for score in scores
                ],
            }
        )

    best = max(
        candidate_results,
        key=lambda result: (
            result["mean_macro_f1"],
            -result["audio_weight"],
            -result["c_value"],
        ),
    )

    return (
        float(best["c_value"]),
        float(best["audio_weight"]),
        candidate_results,
    )


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

    wavlm_path = (
        eval_dir
        / "wavlm_base_plus_v24.npy"
    )

    text_baseline_path = (
        eval_dir
        / "tfidf_probe_canonical_v24.json"
    )

    output_path = (
        eval_dir
        / "tfidf_wavlm_fusion_canonical_v24.json"
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

    with text_baseline_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        text_baseline = json.load(file)

    if not canonical[
        "validation_passed"
    ]:
        print(
            "Canonical speaker validation failed."
        )
        return

    wavlm_embeddings = np.load(
        wavlm_path,
        allow_pickle=False,
    )

    if wavlm_embeddings.shape != (
        len(gold.items),
        13,
        768,
    ):
        print(
            "Unexpected WavLM embedding shape: "
            f"{wavlm_embeddings.shape}"
        )
        return

    audio = wavlm_embeddings[
        :,
        WAVLM_LAYER,
        :,
    ]

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

    canonical_mapping = canonical[
        "mapping"
    ]

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
            np.zeros(
                (len(labels), 1),
                dtype=float,
            ),
            labels,
            groups=groups,
        )
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

    for fold, (
        train_indices,
        test_indices,
    ) in enumerate(
        outer_splits,
        start=1,
    ):
        print(
            f"\nSelecting parameters "
            f"for fold {fold}"
        )

        (
            selected_c,
            selected_audio_weight,
            inner_results,
        ) = select_hyperparameters(
            texts=texts,
            audio=audio,
            labels=labels,
            groups=groups,
            train_indices=train_indices,
            outer_fold=fold,
        )

        print(
            f"  selected C: "
            f"{selected_c:g}"
        )

        print(
            f"  selected audio weight: "
            f"{selected_audio_weight:g}"
        )

        vectorizer = create_text_vectorizer()

        train_text_features = (
            vectorizer.fit_transform(
                texts[train_indices]
            )
        )

        test_text_features = (
            vectorizer.transform(
                texts[test_indices]
            )
        )

        audio_scaler = StandardScaler()

        train_audio = (
            audio_scaler.fit_transform(
                audio[train_indices]
            )
        )

        test_audio = (
            audio_scaler.transform(
                audio[test_indices]
            )
        )

        train_features = combine_features(
            text_features=(
                train_text_features
            ),
            audio_features=train_audio,
            audio_weight=(
                selected_audio_weight
            ),
        )

        test_features = combine_features(
            text_features=(
                test_text_features
            ),
            audio_features=test_audio,
            audio_weight=(
                selected_audio_weight
            ),
        )

        classifier = create_classifier(
            selected_c
        )

        classifier.fit(
            train_features,
            labels[train_indices],
        )

        fold_predictions = (
            classifier.predict(
                test_features
            )
        )

        fold_probabilities = (
            classifier.predict_proba(
                test_features
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
                "selected_audio_weight": (
                    selected_audio_weight
                ),
                "accuracy": (
                    fold_metrics["accuracy"]
                ),
                "macro_f1": (
                    fold_metrics["macro_f1"]
                ),
                "inner_search": inner_results,
            }
        )

        print(
            f"  accuracy: "
            f"{fold_metrics['accuracy']:.3f}"
        )

        print(
            f"  macro F1: "
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

    text_prediction_lookup = {
        item["turn_id"]: item[
            "predicted_label"
        ]
        for item
        in text_baseline["predictions"]
    }

    fixes = 0
    regressions = 0
    wrong_to_wrong = 0
    unchanged = 0

    for index, item in enumerate(
        gold.items
    ):
        text_prediction = (
            text_prediction_lookup[
                item.turn_id
            ]
        )

        fusion_prediction = str(
            predictions[index]
        )

        gold_label = str(
            labels[index]
        )

        if (
            text_prediction
            == fusion_prediction
        ):
            unchanged += 1
        elif (
            text_prediction != gold_label
            and fusion_prediction
            == gold_label
        ):
            fixes += 1
        elif (
            text_prediction == gold_label
            and fusion_prediction
            != gold_label
        ):
            regressions += 1
        else:
            wrong_to_wrong += 1

    text_macro_f1 = float(
        text_baseline[
            "overall"
        ]["macro_f1"]
    )

    output = {
        "schema_version": "1.0",
        "task": (
            "four-class permissive "
            "speech-act classification"
        ),
        "modality": (
            "TF-IDF text plus WavLM "
            f"layer {WAVLM_LAYER}"
        ),
        "evaluation_status": (
            "canonical cross-case "
            "speaker groups"
        ),
        "c_values": C_VALUES,
        "audio_weights": AUDIO_WEIGHTS,
        "wavlm_layer": WAVLM_LAYER,
        "overall": overall,
        "easy": easy_metrics,
        "hard": hard_metrics,
        "fold_macro_f1_mean": float(
            mean(fold_macro_f1)
        ),
        "fold_macro_f1_std": float(
            pstdev(fold_macro_f1)
        ),
        "folds": fold_results,
        "comparison_to_tfidf": {
            "text_macro_f1": (
                text_macro_f1
            ),
            "fusion_macro_f1": (
                overall["macro_f1"]
            ),
            "macro_f1_difference": (
                overall["macro_f1"]
                - text_macro_f1
            ),
            "unchanged": unchanged,
            "fixes": fixes,
            "regressions": regressions,
            "wrong_to_wrong": (
                wrong_to_wrong
            ),
        },
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

    comparison = output[
        "comparison_to_tfidf"
    ]

    print("\nFUSION COMPLETE")
    print(
        f"TF-IDF macro F1: "
        f"{text_macro_f1:.3f}"
    )
    print(
        f"Fusion accuracy: "
        f"{overall['accuracy']:.3f}"
    )
    print(
        f"Fusion macro F1: "
        f"{overall['macro_f1']:.3f}"
    )
    print(
        f"Macro F1 difference: "
        f"{comparison[
            'macro_f1_difference'
        ]:+.3f}"
    )
    print(
        f"Easy macro F1: "
        f"{easy_metrics['macro_f1']:.3f}"
    )
    print(
        f"Hard macro F1: "
        f"{hard_metrics['macro_f1']:.3f}"
    )

    print("\nPrediction changes")
    print(
        f"  fixes: {fixes}"
    )
    print(
        f"  regressions: {regressions}"
    )
    print(
        f"  wrong to different wrong: "
        f"{wrong_to_wrong}"
    )
    print(
        f"  unchanged: {unchanged}"
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