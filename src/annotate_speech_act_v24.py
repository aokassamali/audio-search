import argparse
import json
from pathlib import Path

from src.annotate_register import (
    DIFFICULTY_OPTIONS,
    LABEL_OPTIONS,
    format_turn,
    play_audio_slice,
    prompt_choice,
)
from src.chunk import (
    build_speaker_turns,
    load_speaker_labels,
)
from src.config import load_settings
from src.speech_act_gold import (
    SpeechActGoldArtifact,
)


def save_artifact(
    artifact: SpeechActGoldArtifact,
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


def load_source_data(
    settings,
    source_key: str,
    cache: dict,
):
    if source_key in cache:
        return cache[source_key]

    source = settings.get_source(source_key)

    with source.speaker_transcript_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        segments = json.load(file)

    if source.speaker_roles_path.exists():
        speaker_labels = load_speaker_labels(
            source.speaker_roles_path
        )
    else:
        speaker_labels = {}

    turns = build_speaker_turns(
        segments,
        speaker_labels=speaker_labels,
    )

    cache[source_key] = (
        source,
        turns,
    )

    return source, turns


def validate_items(
    artifact: SpeechActGoldArtifact,
    settings,
    cache: dict,
) -> bool:
    problems = []

    for item in artifact.items:
        source, turns = load_source_data(
            settings=settings,
            source_key=item.source_key,
            cache=cache,
        )

        if source.source_id != item.source_id:
            problems.append(
                f"{item.turn_id}: source ID mismatch"
            )
            continue

        if item.turn_index >= len(turns):
            problems.append(
                f"{item.turn_id}: turn index missing"
            )
            continue

        current_text = turns[
            item.turn_index
        ]["text"].strip()

        if current_text != item.text.strip():
            problems.append(
                f"{item.turn_id}: transcript changed"
            )

    if problems:
        print(
            "Gold artifact validation failed."
        )

        for problem in problems[:10]:
            print(f"  {problem}")

        print(
            f"Total problems: {len(problems)}"
        )

        return False

    print(
        "Validated all "
        f"{len(artifact.items)} items."
    )

    return True


def is_complete(item) -> bool:
    return (
        item.label_permissive is not None
        and item.label_strict is not None
        and item.difficulty is not None
    )


def annotate(
    gold_path: Path,
    padding: float,
) -> None:
    settings = load_settings()

    with gold_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        artifact = (
            SpeechActGoldArtifact.model_validate(
                json.load(file)
            )
        )

    source_cache = {}

    if not validate_items(
        artifact=artifact,
        settings=settings,
        cache=source_cache,
    ):
        return

    incomplete_indices = [
        index
        for index, item
        in enumerate(artifact.items)
        if not item.frozen
        and not is_complete(item)
    ]

    total_new = sum(
        not item.frozen
        for item in artifact.items
    )

    completed_new = sum(
        not item.frozen
        and is_complete(item)
        for item in artifact.items
    )

    if not incomplete_indices:
        print("All new items are annotated.")
        return

    print(
        f"Frozen items: "
        f"{len(artifact.items) - total_new}"
    )
    print(
        f"New annotation progress: "
        f"{completed_new}/{total_new}"
    )

    print(
        "\nSpeech act labels"
        "\n1 = assertion"
        "\n2 = hypothetical"
        "\n3 = question"
        "\n4 = characterization"
        "\n5 = hyperbole"
        "\n6 = joke"
    )

    print(
        "\nStrict hypothetical requires an "
        "explicit marker such as suppose, "
        "imagine, what if, or assume."
    )

    for item_index in incomplete_indices:
        item = artifact.items[item_index]

        source, turns = load_source_data(
            settings=settings,
            source_key=item.source_key,
            cache=source_cache,
        )

        turn = turns[item.turn_index]

        source_total = sum(
            not gold_item.frozen
            and gold_item.source_key
            == item.source_key
            for gold_item in artifact.items
        )

        source_completed = sum(
            not gold_item.frozen
            and gold_item.source_key
            == item.source_key
            and is_complete(gold_item)
            for gold_item in artifact.items
        )

        print("\n" + "=" * 80)
        print(
            f"Source: {item.source_key}"
            f"\nSource progress: "
            f"{source_completed}/{source_total}"
            f"\nOverall progress: "
            f"{completed_new}/{total_new}"
            f"\nTurn: {item.turn_id}"
            f"\nTime: {item.start:.2f}"
            f"-{item.end:.2f}"
            f"\nDuration: "
            f"{item.duration:.2f}s"
        )

        if item.turn_index > 0:
            print(
                "\nPREVIOUS\n"
                + format_turn(
                    turns[item.turn_index - 1]
                )
            )

        print(
            "\nTARGET\n"
            + format_turn(turn)
        )

        if item.turn_index < len(turns) - 1:
            print(
                "\nNEXT\n"
                + format_turn(
                    turns[item.turn_index + 1]
                )
            )

        while True:
            play_audio_slice(
                audio_path=(
                    source.normalized_audio_path
                ),
                start=item.start,
                end=item.end,
                padding=padding,
            )

            action = input(
                "\nEnter = label | "
                "r = replay | "
                "s = skip | "
                "q = quit: "
            ).strip().lower()

            if action == "r":
                continue

            if action == "s":
                break

            if action == "q":
                print(
                    f"Progress saved in "
                    f"{gold_path}"
                )
                return

            permissive_label = prompt_choice(
                "\nPermissive label",
                LABEL_OPTIONS,
                default=item.label_permissive,
            )

            strict_label = prompt_choice(
                "Strict label",
                LABEL_OPTIONS,
                default=(
                    item.label_strict
                    or permissive_label
                ),
            )

            difficulty = prompt_choice(
                "Difficulty",
                DIFFICULTY_OPTIONS,
                default=item.difficulty,
            )

            notes_prompt = "Notes"

            if item.notes:
                notes_prompt += (
                    f" [Enter = {item.notes}]"
                )

            notes = input(
                f"{notes_prompt}: "
            ).strip()

            if not notes:
                notes = item.notes

            item.label_permissive = (
                permissive_label
            )
            item.label_strict = strict_label
            item.difficulty = difficulty
            item.notes = notes

            save_artifact(
                artifact=artifact,
                output_path=gold_path,
            )

            completed_new += 1

            print(
                f"\nSaved: "
                f"{completed_new}/{total_new}"
            )

            break


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gold",
        type=Path,
        default=(
            Path("data/eval")
            / "speech_act_gold_v24.json"
        ),
    )

    parser.add_argument(
        "--padding",
        type=float,
        default=1.0,
    )

    arguments = parser.parse_args()

    annotate(
        gold_path=arguments.gold,
        padding=arguments.padding,
    )


if __name__ == "__main__":
    main()