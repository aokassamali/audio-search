import json
from collections import Counter, defaultdict

from src.config import load_settings
from src.extract_register_features import (
    RegisterFeaturesArtifact,
)


def summarize_variant(
    artifact: RegisterFeaturesArtifact,
    variant: str,
) -> None:
    label_field = f"label_{variant}"

    label_counts = Counter()
    speakers_by_label = defaultdict(set)
    counts_by_speaker = defaultdict(Counter)
    speaker_labels = {}

    for item in artifact.items:
        label = getattr(item, label_field)

        label_counts[label] += 1
        speakers_by_label[label].add(
            item.speaker
        )
        counts_by_speaker[
            item.speaker
        ][label] += 1

        speaker_labels[item.speaker] = (
            item.speaker_label
        )

    print(f"\n{variant.upper()}")

    print("\nClass coverage:")
    for label, count in sorted(
        label_counts.items()
    ):
        print(
            f"  {label:16}"
            f" turns={count:2d}"
            f" speakers="
            f"{len(speakers_by_label[label])}"
        )

    print("\nCounts by speaker:")
    for speaker in sorted(
        counts_by_speaker
    ):
        counts = counts_by_speaker[speaker]

        formatted_counts = ", ".join(
            f"{label}={count}"
            for label, count
            in sorted(counts.items())
        )

        print(
            f"  {speaker:10} "
            f"({speaker_labels[speaker]}): "
            f"{formatted_counts}"
        )


def main() -> None:
    settings = load_settings()

    feature_path = (
        settings.paths.eval_dir
        / "register_features.json"
    )

    with feature_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        artifact = (
            RegisterFeaturesArtifact.model_validate(
                json.load(file)
            )
        )

    print(
        "Items:",
        len(artifact.items),
    )

    print(
        "Features per item:",
        len(artifact.items[0].features),
    )

    summarize_variant(
        artifact,
        "permissive",
    )

    summarize_variant(
        artifact,
        "strict",
    )


if __name__ == "__main__":
    main()