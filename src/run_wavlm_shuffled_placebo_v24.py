import argparse
import json
from statistics import mean, pstdev

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import (
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


def create_model(c_value: float) -> Pipeline:
    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=24,
                ),
            ),
        ]
    )


def shuffle_within_groups(
    embeddings: np.ndarray,
    groups: np.ndarray,
    random_generator: np.random.Generator,
) -> tuple[np.ndarray, int]:
    shuffled = np.empty_like(embeddings)

    self_matches = 0

    for group in np.unique(groups):
        indices = np.flatnonzero(
            groups == group
        )

        if len(indices) == 1:
            shuffled[indices] = embeddings[
                indices
            ]

            self_matches += 1
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
            embeddings[source_indices]
        )

    return shuffled, self_matches


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


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--permutations",
        type=int,
        default=25,
    )

    arguments = parser.parse_args()

    settings = load_settings()
    eval_dir = settings.paths.eval_dir

    with (
        eval_dir
        / "speech_act_gold_v24.json"
    ).open("r", encoding="utf-8") as file:
        gold_artifact = (
            SpeechActGoldArtifact.model_validate(
                json.load(file)
            )
        )

    with (
        eval_dir
        / "canonical_speaker_groups_v24.json"
    ).open("r", encoding="utf-8") as file:
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

    with (
        eval_dir
        / "wavlm_layer_probe_canonical_v24.json"
    ).open("r", encoding="utf-8") as file:
        real_results = json.load(file)

    embeddings = np.load(
        eval_dir
        / "wavlm_base_plus_v24.npy",
        allow_pickle=False,
    )

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
        print("Missing canonical mappings:")

        for group in missing_groups:
            print(f"  {group}")

        return

    groups = np.asarray(
        [
            canonical_mapping[group]
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
                (len(labels), 1),
                dtype=float,
            ),
            labels,
            groups=groups,
        )
    )

    real_best_layer = int(
        real_results[
            "best_layer_descriptive"
        ]
    )

    real_best_macro_f1 = float(
        real_results[
            "best_macro_f1_descriptive"
        ]
    )

    shuffled_runs = []

    for permutation_index in range(
        arguments.permutations
    ):
        seed = 24000 + permutation_index

        random_generator = (
            np.random.default_rng(seed)
        )

        (
            shuffled_embeddings,
            self_matches,
        ) = shuffle_within_groups(
            embeddings=embeddings,
            groups=groups,
            random_generator=random_generator,
        )

        layer_scores = []

        for layer_index in range(
            embeddings.shape[1]
        ):
            predictions = np.empty(
                len(labels),
                dtype=object,
            )

            real_layer_result = (
                real_results["layers"][
                    layer_index
                ]
            )

            for fold_index, (
                train_indices,
                test_indices,
            ) in enumerate(splits):
                c_value = float(
                    real_layer_result[
                        "folds"
                    ][fold_index][
                        "selected_c"
                    ]
                )

                model = create_model(
                    c_value
                )

                model.fit(
                    shuffled_embeddings[
                        train_indices,
                        layer_index,
                        :,
                    ],
                    labels[train_indices],
                )

                predictions[test_indices] = (
                    model.predict(
                        shuffled_embeddings[
                            test_indices,
                            layer_index,
                            :,
                        ]
                    )
                )

            layer_scores.append(
                calculate_macro_f1(
                    labels,
                    predictions,
                )
            )

        best_layer = int(
            np.argmax(layer_scores)
        )

        best_macro_f1 = float(
            layer_scores[best_layer]
        )

        selected_layer_macro_f1 = float(
            layer_scores[real_best_layer]
        )

        shuffled_runs.append(
            {
                "seed": seed,
                "self_matches": (
                    self_matches
                ),
                "best_layer": best_layer,
                "best_macro_f1": (
                    best_macro_f1
                ),
                "real_selected_layer": (
                    real_best_layer
                ),
                "selected_layer_macro_f1": (
                    selected_layer_macro_f1
                ),
                "layer_scores": (
                    layer_scores
                ),
            }
        )

        print(
            f"[{permutation_index + 1}/"
            f"{arguments.permutations}] "
            f"best layer={best_layer} "
            f"macro F1={best_macro_f1:.3f} "
            f"layer {real_best_layer}="
            f"{selected_layer_macro_f1:.3f}"
        )

    shuffled_best_scores = [
        run["best_macro_f1"]
        for run in shuffled_runs
    ]

    shuffled_selected_scores = [
        run["selected_layer_macro_f1"]
        for run in shuffled_runs
    ]

    best_layer_p_value = (
        1
        + sum(
            score >= real_best_macro_f1
            for score in shuffled_best_scores
        )
    ) / (
        arguments.permutations + 1
    )

    selected_layer_p_value = (
        1
        + sum(
            score >= real_best_macro_f1
            for score
            in shuffled_selected_scores
        )
    ) / (
        arguments.permutations + 1
    )

    output = {
        "schema_version": "1.0",
        "permutations": (
            arguments.permutations
        ),
        "shuffle": (
            "within canonical cross-case speaker "
            "identity, full-turn row permutation"
        ),
        "group_definition": (
            "manually reviewed cross-case"
            "speaker identity components"
        ),
        "real_best_layer": (
            real_best_layer
        ),
        "real_best_macro_f1": (
            real_best_macro_f1
        ),
        "shuffled_best_macro_f1_mean": (
            mean(shuffled_best_scores)
        ),
        "shuffled_best_macro_f1_std": (
            pstdev(shuffled_best_scores)
        ),
        "shuffled_selected_layer_mean": (
            mean(shuffled_selected_scores)
        ),
        "shuffled_selected_layer_std": (
            pstdev(
                shuffled_selected_scores
            )
        ),
        "best_layer_empirical_p_value": (
            best_layer_p_value
        ),
        "selected_layer_empirical_p_value": (
            selected_layer_p_value
        ),
        "runs": shuffled_runs,
    }

    output_path = (
        eval_dir
        / "wavlm_shuffled_placebo_canonical_v24.json"
    )

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

    print("\nPLACEBO COMPLETE")
    print(
        "Real best macro F1: "
        f"{real_best_macro_f1:.3f}"
    )
    print(
        "Shuffled best mean: "
        f"{mean(shuffled_best_scores):.3f}"
    )
    print(
        "Shuffled best std: "
        f"{pstdev(shuffled_best_scores):.3f}"
    )
    print(
        "Best-layer empirical p: "
        f"{best_layer_p_value:.3f}"
    )
    print(
        f"Layer {real_best_layer} "
        "shuffled mean: "
        f"{mean(
            shuffled_selected_scores
        ):.3f}"
    )
    print(
        "Selected-layer empirical p: "
        f"{selected_layer_p_value:.3f}"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()