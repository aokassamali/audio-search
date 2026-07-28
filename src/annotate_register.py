import argparse
import json
import subprocess
from pathlib import Path

from src.chunk import (
    build_speaker_turns,
    load_speaker_labels,
)
from src.config import load_settings
from src.register_gold import (
    RegisterGoldArtifact,
)


LABEL_OPTIONS = {
    "1": "assertion",
    "2": "hypothetical",
    "3": "question",
    "4": "characterization",
    "5": "hyperbole",
    "6": "joke",
}

DIFFICULTY_OPTIONS = {
    "1": "easy",
    "2": "hard",
}


def save_artifact(
    artifact: RegisterGoldArtifact,
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


def play_audio_slice(
    audio_path: Path,
    start: float,
    end: float,
    padding: float,
) -> None:
    clip_start = max(
        0.0,
        start - padding,
    )

    clip_duration = max(
        0.1,
        end - start + (2 * padding),
    )

    try:
        subprocess.run(
            [
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "error",
                "-ss",
                str(clip_start),
                "-t",
                str(clip_duration),
                str(audio_path),
            ],
            check=False,
        )

    except FileNotFoundError:
        print(
            "\nffplay was not found. Confirm that "
            "your FFmpeg installation includes "
            "ffplay and that it is on PATH."
        )


def shorten_text(
    text: str,
    maximum_characters: int = 300,
) -> str:
    text = text.strip()

    if len(text) <= maximum_characters:
        return text

    return (
        text[:maximum_characters].rstrip()
        + " ..."
    )


def format_turn(turn: dict) -> str:
    speaker = turn["speaker"]

    speaker_label = turn.get(
        "speaker_label",
        speaker,
    )

    if speaker_label == speaker:
        display_name = speaker
    else:
        display_name = (
            f"{speaker_label} [{speaker}]"
        )

    return (
        f"{display_name}: "
        f"{shorten_text(turn['text'])}"
    )


def prompt_choice(
    prompt: str,
    options: dict[str, str],
    default: str | None = None,
) -> str:
    while True:
        default_display = ""

        if default is not None:
            default_display = (
                f" [Enter = {default}]"
            )

        answer = input(
            f"{prompt}{default_display}: "
        ).strip().lower()

        if not answer and default is not None:
            return default

        if answer in options:
            return options[answer]

        if answer in options.values():
            return answer

        print(
            "Enter one of: "
            + ", ".join(
                f"{number}={label}"
                for number, label
                in options.items()
            )
        )


def annotate_register_gold(
    source_key: str | None = None,
    gold_path: str | Path | None = None,
    padding: float = 1.0,
) -> None:
    settings = load_settings()

    if source_key is None:
        source_key = (
            settings.default_source_key
        )

    source = settings.get_source(
        source_key
    )

    if gold_path is None:
        gold_path = (
            settings.paths.eval_dir
            / "register_gold.json"
        )

    gold_path = Path(gold_path)

    with gold_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        artifact = (
            RegisterGoldArtifact.model_validate(
                json.load(file)
            )
        )

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

    incomplete_indices = [
        index
        for index, item
        in enumerate(artifact.items)
        if (
            item.label_permissive is None
            or item.label_strict is None
            or item.difficulty is None
        )
    ]

    if not incomplete_indices:
        print("All gold items are annotated.")
        return

    print(
        "\nRegister labels:"
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

    for progress, item_index in enumerate(
        incomplete_indices,
        start=1,
    ):
        item = artifact.items[item_index]

        if item.turn_index >= len(turns):
            print(
                f"\nCould not find "
                f"{item.turn_id}; skipping."
            )
            continue

        turn = turns[item.turn_index]

        print("\n" + "=" * 80)
        print(
            f"Remaining item "
            f"{progress}/{len(incomplete_indices)}"
        )
        print(
            f"Turn: {item.turn_id}"
            f"\nTime: {item.start:.2f}"
            f"–{item.end:.2f}"
            f"\nDuration: {item.duration:.2f}s"
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
                    f"Progress saved in {gold_path}"
                )
                return

            permissive_label = prompt_choice(
                "\nPermissive label",
                LABEL_OPTIONS,
                default=item.label_permissive,
            )

            strict_default = (
                item.label_strict
                or permissive_label
            )

            strict_label = prompt_choice(
                "Strict label",
                LABEL_OPTIONS,
                default=strict_default,
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
                artifact,
                gold_path,
            )

            completed_count = sum(
                gold_item.label_permissive
                is not None
                and gold_item.label_strict
                is not None
                and gold_item.difficulty
                is not None
                for gold_item in artifact.items
            )

            print(
                f"\nSaved: {completed_count}/"
                f"{len(artifact.items)}"
            )

            break


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-key",
        default=None,
    )

    parser.add_argument(
        "--gold",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--padding",
        type=float,
        default=1.0,
    )

    arguments = parser.parse_args()

    annotate_register_gold(
        source_key=arguments.source_key,
        gold_path=arguments.gold,
        padding=arguments.padding,
    )


if __name__ == "__main__":
    main()