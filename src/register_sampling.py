import json
import random
from pathlib import Path

from src.chunk import (
    build_speaker_turns,
    load_speaker_labels,
)
from src.config import load_settings
from src.register_gold import (
    RegisterGoldArtifact,
    RegisterGoldItem,
    SamplingMetadata,
)


def create_register_gold_candidates(
    source_key: str,
    output_path: str | Path | None = None,
    requested_count: int = 80,
    seed: int = 23,
    minimum_words: int = 8,
) -> tuple[Path, int, int]:
    settings = load_settings()
    source = settings.get_source(source_key)

    with source.speaker_transcript_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        segments = json.load(file)

    speaker_labels = load_speaker_labels(
        source.speaker_roles_path
    )

    turns = build_speaker_turns(
        segments,
        speaker_labels=speaker_labels,
    )

    eligible_items = []

    for turn_index, turn in enumerate(turns):
        text = turn["text"].strip()
        word_count = len(text.split())

        if word_count < minimum_words:
            continue

        start = float(turn["start"])
        end = float(turn["end"])

        eligible_items.append(
            RegisterGoldItem(
                turn_id=(
                    f"{source.source_id}:"
                    f"turn_{turn_index:04d}"
                ),
                source_id=source.source_id,
                turn_index=turn_index,
                speaker=turn["speaker"],
                speaker_label=turn[
                    "speaker_label"
                ],
                start=start,
                end=end,
                duration=end - start,
                text=text,
                word_count=word_count,
            )
        )

    sample_count = min(
        requested_count,
        len(eligible_items),
    )

    random_generator = random.Random(seed)

    sampled_items = random_generator.sample(
        eligible_items,
        k=sample_count,
    )

    sampled_items.sort(
        key=lambda item: item.start
    )

    artifact = RegisterGoldArtifact(
        source_id=source.source_id,
        sampling=SamplingMetadata(
            method=(
                "uniform_random_over_eligible_"
                "speaker_turns"
            ),
            seed=seed,
            requested_count=requested_count,
            minimum_words=minimum_words,
        ),
        items=sampled_items,
    )

    if output_path is None:
        output_path = (
            settings.paths.eval_dir
            / "register_gold.json"
        )

    output_path = Path(output_path)

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
        sample_count,
        len(eligible_items),
    )