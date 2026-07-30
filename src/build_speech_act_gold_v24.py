import json
import random
from collections import Counter

from src.chunk import (
    build_speaker_turns,
    load_speaker_labels,
)
from src.config import load_settings
from src.register_gold import (
    RegisterGoldArtifact,
)
from src.speech_act_gold import (
    SourceSamplingRecord,
    SpeechActGoldArtifact,
    SpeechActGoldItem,
    SpeechActSamplingMetadata,
)


SEED = 24
MINIMUM_WORDS = 8
TARGET_COUNT = 400

SOURCE_QUOTAS = {
    "sripetch": 20,
    "case_two": 100,
    "case_three": 100,
    "case_four": 100,
}


def build_candidates(
    source_key: str,
    existing_turn_ids: set[str],
) -> tuple[
    list[SpeechActGoldItem],
    int,
]:
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

    candidates = []
    excluded_existing = 0

    for turn_index, turn in enumerate(turns):
        text = turn["text"].strip()
        word_count = len(text.split())

        if word_count < MINIMUM_WORDS:
            continue

        turn_id = (
            f"{source.source_id}:"
            f"turn_{turn_index:04d}"
        )

        if turn_id in existing_turn_ids:
            excluded_existing += 1
            continue

        start = float(turn["start"])
        end = float(turn["end"])

        candidates.append(
            SpeechActGoldItem(
                source_key=source_key,
                source_id=source.source_id,
                turn_id=turn_id,
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

    return candidates, excluded_existing


def main() -> None:
    settings = load_settings()
    eval_dir = settings.paths.eval_dir

    frozen_path = (
        eval_dir
        / "register_gold.json"
    )

    output_path = (
        eval_dir
        / "speech_act_gold_v24.json"
    )

    if not frozen_path.exists():
        print(
            "Missing frozen v2.3 gold file:"
            f"\n{frozen_path}"
        )
        return

    with frozen_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        frozen_artifact = (
            RegisterGoldArtifact.model_validate(
                json.load(file)
            )
        )

    if len(frozen_artifact.items) != 80:
        print(
            "Expected 80 frozen items. Found "
            f"{len(frozen_artifact.items)}."
        )
        return

    combined_items = [
        SpeechActGoldItem(
            **item.model_dump(),
            source_key="sripetch",
            frozen=True,
        )
        for item in frozen_artifact.items
    ]

    existing_turn_ids = {
        item.turn_id
        for item in combined_items
    }

    source_records = []

    for source_key, requested_count in (
        SOURCE_QUOTAS.items()
    ):
        candidates, excluded_existing = (
            build_candidates(
                source_key=source_key,
                existing_turn_ids=(
                    existing_turn_ids
                ),
            )
        )

        if len(candidates) < requested_count:
            print(
                f"{source_key} has only "
                f"{len(candidates)} candidates. "
                f"Requested {requested_count}."
            )
            return

        random_generator = random.Random(
            f"{SEED}:{source_key}"
        )

        sampled_items = random_generator.sample(
            candidates,
            k=requested_count,
        )

        sampled_items.sort(
            key=lambda item: item.start
        )

        combined_items.extend(sampled_items)

        existing_turn_ids.update(
            item.turn_id
            for item in sampled_items
        )

        source = settings.get_source(
            source_key
        )

        source_records.append(
            SourceSamplingRecord(
                source_key=source_key,
                source_id=source.source_id,
                requested_count=requested_count,
                eligible_count=(
                    len(candidates)
                    + excluded_existing
                ),
                excluded_existing_count=(
                    excluded_existing
                ),
            )
        )

    if len(combined_items) != TARGET_COUNT:
        print(
            "Expected "
            f"{TARGET_COUNT} total items. "
            f"Built {len(combined_items)}."
        )
        return

    artifact = SpeechActGoldArtifact(
        sampling=SpeechActSamplingMetadata(
            method=(
                "stratified_uniform_random_"
                "sampling_by_case"
            ),
            seed=SEED,
            minimum_words=MINIMUM_WORDS,
            target_count=TARGET_COUNT,
            source_quotas=SOURCE_QUOTAS,
            source_records=source_records,
        ),
        items=combined_items,
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

    counts = Counter(
        item.source_key
        for item in artifact.items
    )

    frozen_count = sum(
        item.frozen
        for item in artifact.items
    )

    print(f"Saved: {output_path}")
    print(f"Total items: {len(artifact.items)}")
    print(f"Frozen items: {frozen_count}")

    for source_key in SOURCE_QUOTAS:
        print(
            f"{source_key}: "
            f"{counts[source_key]}"
        )


if __name__ == "__main__":
    main()