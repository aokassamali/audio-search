import json
from pathlib import Path

from pydantic import BaseModel, Field

from src.config import load_settings
from src.llm_clients import LlamaCppClient
from src.register_classification import (
    RegisterPrediction,
    classify_text_only,
)
from src.register_gold import (
    Difficulty,
    RegisterGoldArtifact,
    RegisterLabel,
)


class TextArmItem(BaseModel):
    turn_id: str
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


class TextArmArtifact(BaseModel):
    schema_version: str = "1.0"
    model: str

    items: list[TextArmItem] = Field(
        default_factory=list
    )


def save_artifact(
    artifact: TextArmArtifact,
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


def run_text_arm() -> None:
    settings = load_settings()

    gold_path = (
        settings.paths.eval_dir
        / "register_gold.json"
    )

    output_path = (
        settings.paths.eval_dir
        / "register_text_predictions.json"
    )

    with gold_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        gold = RegisterGoldArtifact.model_validate(
            json.load(file)
        )

    if output_path.exists():
        with output_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            artifact = (
                TextArmArtifact.model_validate(
                    json.load(file)
                )
            )
    else:
        artifact = TextArmArtifact(
            model=settings.llm.model,
            items=[
                TextArmItem(
                    turn_id=item.turn_id,
                    text=item.text,
                    gold_permissive=(
                        item.label_permissive
                    ),
                    gold_strict=(
                        item.label_strict
                    ),
                    difficulty=item.difficulty,
                )
                for item in gold.items
            ],
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

        if item.permissive_prediction is None:
            item.permissive_prediction = (
                classify_text_only(
                    turn_id=item.turn_id,
                    text=item.text,
                    taxonomy_variant=(
                        "permissive"
                    ),
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
                classify_text_only(
                    turn_id=item.turn_id,
                    text=item.text,
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

    print(f"\nSaved: {output_path}")
    print(f"Items: {len(artifact.items)}")


if __name__ == "__main__":
    run_text_arm()