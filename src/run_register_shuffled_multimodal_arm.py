import json
import random
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from src.config import load_settings
from src.extract_register_features import (
    RegisterFeaturesArtifact,
)
from src.llm_clients import LlamaCppClient
from src.register_classification import (
    RegisterPrediction,
)
from src.register_gold import (
    Difficulty,
    RegisterLabel,
)
from src.register_multimodal import (
    classify_text_plus_prosody,
    compute_compact_z_scores,
)


SHUFFLE_SEED = 23


class ShuffledMultimodalItem(BaseModel):
    turn_id: str
    prosody_source_turn_id: str
    text: str

    gold_permissive: RegisterLabel
    gold_strict: RegisterLabel
    difficulty: Difficulty

    permissive_prediction: (
        RegisterPrediction | None
    ) = None

    strict_prediction: (
        RegisterPrediction | None
    ) = None


class ShuffledMultimodalArtifact(BaseModel):
    schema_version: str = "1.0"
    model: str
    shuffle_seed: int = SHUFFLE_SEED
    shuffle_method: str = (
        "within-speaker derangement"
    )

    items: list[
        ShuffledMultimodalItem
    ] = Field(default_factory=list)


def build_within_speaker_shuffle(
    feature_items,
    seed: int,
) -> dict[int, int]:
    indices_by_speaker = defaultdict(list)

    for index, item in enumerate(
        feature_items
    ):
        indices_by_speaker[
            item.speaker
        ].append(index)

    random_generator = random.Random(seed)
    source_index_by_target = {}

    for speaker_indices in (
        indices_by_speaker.values()
    ):
        shuffled_indices = (
            speaker_indices.copy()
        )

        if len(speaker_indices) == 1:
            source_index_by_target[
                speaker_indices[0]
            ] = speaker_indices[0]
            continue

        while True:
            random_generator.shuffle(
                shuffled_indices
            )

            has_fixed_point = any(
                target_index == source_index
                for target_index, source_index
                in zip(
                    speaker_indices,
                    shuffled_indices,
                )
            )

            if not has_fixed_point:
                break

        for target_index, source_index in zip(
            speaker_indices,
            shuffled_indices,
        ):
            source_index_by_target[
                target_index
            ] = source_index

    return source_index_by_target


def save_artifact(
    artifact: ShuffledMultimodalArtifact,
    output_path: Path,
) -> None:
    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            artifact.model_dump(mode="json"),
            file,
            indent=2,
            ensure_ascii=False,
        )

    temporary_path.replace(output_path)


def main() -> None:
    settings = load_settings()

    feature_path = (
        settings.paths.eval_dir
        / "register_features.json"
    )

    output_path = (
        settings.paths.eval_dir
        / (
            "register_multimodal_"
            "shuffled_predictions.json"
        )
    )

    with feature_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        features = (
            RegisterFeaturesArtifact.model_validate(
                json.load(file)
            )
        )

    feature_lookup = {
        item.turn_id: item
        for item in features.items
    }

    if output_path.exists():
        with output_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            artifact = (
                ShuffledMultimodalArtifact
                .model_validate(
                    json.load(file)
                )
            )
    else:
        shuffled_indices = (
            build_within_speaker_shuffle(
                features.items,
                seed=SHUFFLE_SEED,
            )
        )

        artifact = (
            ShuffledMultimodalArtifact(
                model=settings.llm.model,
                items=[
                    ShuffledMultimodalItem(
                        turn_id=item.turn_id,
                        prosody_source_turn_id=(
                            features.items[
                                shuffled_indices[index]
                            ].turn_id
                        ),
                        text=item.text,
                        gold_permissive=(
                            item.label_permissive
                        ),
                        gold_strict=(
                            item.label_strict
                        ),
                        difficulty=(
                            item.difficulty
                        ),
                    )
                    for index, item
                    in enumerate(features.items)
                ],
            )
        )

        save_artifact(
            artifact,
            output_path,
        )

    client = LlamaCppClient(
        base_url=settings.llm.base_url,
        model=settings.llm.model,
        timeout=settings.llm.timeout_seconds,
    )

    for index, item in enumerate(
        artifact.items,
        start=1,
    ):
        print(
            f"\n{index}/{len(artifact.items)} "
            f"{item.turn_id}"
        )

        prosody_source = feature_lookup[
            item.prosody_source_turn_id
        ]

        z_scores = compute_compact_z_scores(
            features.items,
            prosody_source.turn_id,
        )

        if item.permissive_prediction is None:
            item.permissive_prediction = (
                classify_text_plus_prosody(
                    turn_id=item.turn_id,
                    text=item.text,
                    z_scores=z_scores,
                    taxonomy_variant="permissive",
                    llm_client=client,
                )
            )

            save_artifact(
                artifact,
                output_path,
            )

            print(
                "  permissive:",
                item.permissive_prediction.label,
                item.permissive_prediction.confidence,
            )

        if item.strict_prediction is None:
            item.strict_prediction = (
                classify_text_plus_prosody(
                    turn_id=item.turn_id,
                    text=item.text,
                    z_scores=z_scores,
                    taxonomy_variant="strict",
                    llm_client=client,
                )
            )

            save_artifact(
                artifact,
                output_path,
            )

            print(
                "  strict:",
                item.strict_prediction.label,
                item.strict_prediction.confidence,
            )

    self_matches = sum(
        item.turn_id
        == item.prosody_source_turn_id
        for item in artifact.items
    )

    print(f"\nSaved: {output_path}")
    print(f"Items: {len(artifact.items)}")
    print(f"Self-matches: {self_matches}")


if __name__ == "__main__":
    main()