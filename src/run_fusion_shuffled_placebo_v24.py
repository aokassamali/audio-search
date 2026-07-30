import argparse
import json
from statistics import mean, pstdev

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    StratifiedGroupKFold,
)
from sklearn.preprocessing import StandardScaler

from src.config import load_settings
from src.run_tfidf_wavlm_fusion_canonical_v24 import (
    LABELS,
    WAVLM_LAYER,
    create_classifier,
    create_text_vectorizer,
)
from src.speech_act_gold import (
    SpeechActGoldArtifact,
)


def shuffle_within_groups(
    audio: np.ndarray,
    groups: np.ndarray,
    random_generator: np.random.Generator,
) -> np.ndarray:
    shuffled = np.empty_like(audio)

    for group in np.unique(groups):
        indices = np.flatnonzero(
            groups == group
        )

        if len(indices) == 1:
            shuffled[indices] = audio[indices]
            continue

        target_indices = (
            random_generator.permutation(
                indices
            )
        )

        source_indices = np.roll(
            target_indices,
            1,
        )

        shuffled[target_indices] = (
            audio[source_indices]
        )

    return shuffled


def combine_features(
    text_features,
    audio_features: np.ndarray,
    audio_weight: float,
):
    if audio_weight == 0.0:
        return text_features

    return hstack(
        [
            text_features,
            csr_matrix(
                audio_features
                * audio_weight
            ),
        ],
        format="csr",
    )


def macro_f1(
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


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--permutations",
        type=int,
        default=100,
    )

    arguments = parser.parse_args()

    settings = load_settings()
    eval_dir = settings.paths.eval_dir

    with (
        eval_dir
        / "speech_act_gold_v24.json"
    ).open("r", encoding="utf-8") as file:
        gold = (
            SpeechActGoldArtifact.model_validate(
                json.load(file)
            )
        )

    with (
        eval_dir
        / "canonical_speaker_groups_v24.json"
    ).open("r", encoding="utf-8") as file:
        canonical = json.load(file)

    with (
        eval_dir
        / "tfidf_wavlm_fusion_canonical_v24.json"
    ).open("r", encoding="utf-8") as file:
        real_result = json.load(file)

    embeddings = np.load(
        eval_dir
        / "wavlm_base_plus_v24.npy",
        allow_pickle=False,
    )

    audio = embeddings[
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

    provisional_groups = np.asarray(
        [
            f"{item.source_key}:"
            f"{item.speaker}"
            for item in gold.items
        ]
    )

    mapping = canonical["mapping"]

    groups = np.asarray(
        [
            mapping[group]
            for group in provisional_groups
        ]
    )

    splitter = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=24,
    )

    splits = list(
        splitter.split(
            np.zeros(
                (len(labels), 1)
            ),
            labels,
            groups=groups,
        )
    )

    fold_cache = []

    for fold_index, (
        train_indices,
        test_indices,
    ) in enumerate(splits):
        vectorizer = create_text_vectorizer()

        train_text = (
            vectorizer.fit_transform(
                texts[train_indices]
            )
        )

        test_text = vectorizer.transform(
            texts[test_indices]
        )

        fold_result = real_result[
            "folds"
        ][fold_index]

        fold_cache.append(
            {
                "train_indices": (
                    train_indices
                ),
                "test_indices": (
                    test_indices
                ),
                "train_text": train_text,
                "test_text": test_text,
                "c_value": float(
                    fold_result["selected_c"]
                ),
                "audio_weight": float(
                    fold_result[
                        "selected_audio_weight"
                    ]
                ),
            }
        )

    real_macro_f1 = float(
        real_result["overall"]["macro_f1"]
    )

    permutation_scores = []

    for permutation_index in range(
        arguments.permutations
    ):
        random_generator = (
            np.random.default_rng(
                25000 + permutation_index
            )
        )

        shuffled_audio = shuffle_within_groups(
            audio=audio,
            groups=groups,
            random_generator=random_generator,
        )

        predictions = np.empty(
            len(labels),
            dtype=object,
        )

        for fold in fold_cache:
            train_indices = fold[
                "train_indices"
            ]

            test_indices = fold[
                "test_indices"
            ]

            scaler = StandardScaler()

            train_audio = (
                scaler.fit_transform(
                    shuffled_audio[
                        train_indices
                    ]
                )
            )

            test_audio = scaler.transform(
                shuffled_audio[
                    test_indices
                ]
            )

            train_features = (
                combine_features(
                    fold["train_text"],
                    train_audio,
                    fold["audio_weight"],
                )
            )

            test_features = combine_features(
                fold["test_text"],
                test_audio,
                fold["audio_weight"],
            )

            classifier = create_classifier(
                fold["c_value"]
            )

            classifier.fit(
                train_features,
                labels[train_indices],
            )

            predictions[test_indices] = (
                classifier.predict(
                    test_features
                )
            )

        score = macro_f1(
            labels,
            predictions,
        )

        permutation_scores.append(score)

        print(
            f"[{permutation_index + 1}/"
            f"{arguments.permutations}] "
            f"macro F1={score:.3f}"
        )

    empirical_p = (
        1
        + sum(
            score >= real_macro_f1
            for score in permutation_scores
        )
    ) / (
        arguments.permutations + 1
    )

    output = {
        "schema_version": "1.0",
        "permutations": (
            arguments.permutations
        ),
        "real_fusion_macro_f1": (
            real_macro_f1
        ),
        "shuffled_mean_macro_f1": float(
            mean(permutation_scores)
        ),
        "shuffled_std_macro_f1": float(
            pstdev(permutation_scores)
        ),
        "empirical_p_value": float(
            empirical_p
        ),
        "scores": [
            float(score)
            for score in permutation_scores
        ],
    }

    output_path = (
        eval_dir
        / "fusion_shuffled_placebo_v24.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print("\nFUSION PLACEBO COMPLETE")
    print(
        "Real fusion macro F1: "
        f"{real_macro_f1:.3f}"
    )
    print(
        "Shuffled mean macro F1: "
        f"{mean(permutation_scores):.3f}"
    )
    print(
        "Shuffled standard deviation: "
        f"{pstdev(permutation_scores):.3f}"
    )
    print(
        "Empirical p-value: "
        f"{empirical_p:.3f}"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()