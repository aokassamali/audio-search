import json
import math
from pathlib import Path

from pydantic import BaseModel, Field

from src.config import load_settings
from src.prosody import (
    create_egemaps_extractor,
    extract_turn_features,
)
from src.register_gold import (
    Difficulty,
    RegisterGoldArtifact,
    RegisterLabel,
)


class RegisterFeatureItem(BaseModel):
    turn_id: str
    source_id: str
    speaker: str
    speaker_label: str

    start: float
    end: float
    text: str

    label_permissive: RegisterLabel
    label_strict: RegisterLabel
    difficulty: Difficulty

    features: dict[str, float | None]


class RegisterFeaturesArtifact(BaseModel):
    schema_version: str = "1.0"
    feature_set: str = "eGeMAPSv02"
    feature_level: str = "Functionals"

    items: list[RegisterFeatureItem] = Field(
        default_factory=list
    )


def extract_register_features(
    source_key: str,
    gold_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> tuple[Path, int, int]:
    settings = load_settings()
    source = settings.get_source(source_key)

    if gold_path is None:
        gold_path = (
            settings.paths.eval_dir
            / "register_gold.json"
        )

    if output_path is None:
        output_path = (
            settings.paths.eval_dir
            / "register_features.json"
        )

    gold_path = Path(gold_path)
    output_path = Path(output_path)

    with gold_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        gold = RegisterGoldArtifact.model_validate(
            json.load(file)
        )

    extractor = create_egemaps_extractor()

    feature_items = []
    missing_value_count = 0

    for item_number, item in enumerate(
        gold.items,
        start=1,
    ):
        print(
            f"Extracting {item_number}/"
            f"{len(gold.items)}: "
            f"{item.turn_id}"
        )

        raw_features = extract_turn_features(
            audio_path=(
                source.normalized_audio_path
            ),
            start=item.start,
            end=item.end,
            word_count=item.word_count,
            extractor=extractor,
        )

        clean_features = {}

        for name, value in raw_features.items():
            if math.isfinite(value):
                clean_features[name] = value
            else:
                clean_features[name] = None
                missing_value_count += 1

        feature_items.append(
            RegisterFeatureItem(
                turn_id=item.turn_id,
                source_id=item.source_id,
                speaker=item.speaker,
                speaker_label=(
                    item.speaker_label
                ),
                start=item.start,
                end=item.end,
                text=item.text,
                label_permissive=(
                    item.label_permissive
                ),
                label_strict=(
                    item.label_strict
                ),
                difficulty=item.difficulty,
                features=clean_features,
            )
        )

    artifact = RegisterFeaturesArtifact(
        items=feature_items
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
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

    return (
        output_path,
        len(feature_items),
        missing_value_count,
    )


if __name__ == "__main__":
    path, item_count, missing_count = (
        extract_register_features(
            source_key="sripetch"
        )
    )

    print(f"\nSaved: {path}")
    print(f"Items: {item_count}")
    print(f"Missing feature values: {missing_count}")