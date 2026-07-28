import json
from collections import Counter

from src.config import load_settings
from src.register_gold import (
    RegisterGoldArtifact,
)


def main() -> None:
    settings = load_settings()

    input_path = (
        settings.paths.eval_dir
        / "register_gold.json"
    )

    output_path = (
        settings.paths.eval_dir
        / "register_construct_metrics.json"
    )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        artifact = (
            RegisterGoldArtifact.model_validate(
                json.load(file)
            )
        )

    items = artifact.items

    disagreements = [
        item
        for item in items
        if (
            item.label_permissive
            != item.label_strict
        )
    ]

    transitions = Counter(
        (
            item.label_permissive,
            item.label_strict,
        )
        for item in disagreements
    )

    difficulty_counts = Counter(
        item.difficulty
        for item in items
    )

    disagreement_difficulty_counts = Counter(
        item.difficulty
        for item in disagreements
    )

    permissive_hypotheticals = [
        item
        for item in items
        if item.label_permissive
        == "hypothetical"
    ]

    hypothetical_flips = [
        item
        for item in permissive_hypotheticals
        if item.label_strict
        != "hypothetical"
    ]

    metrics = {
        "total_items": len(items),
        "agreement_count": (
            len(items) - len(disagreements)
        ),
        "disagreement_count": (
            len(disagreements)
        ),
        "agreement_rate": (
            (
                len(items)
                - len(disagreements)
            )
            / len(items)
        ),
        "transitions": {
            f"{source}->{target}": count
            for (
                source,
                target,
            ), count in transitions.items()
        },
        "difficulty": {
            difficulty: {
                "total": difficulty_counts[
                    difficulty
                ],
                "disagreements": (
                    disagreement_difficulty_counts[
                        difficulty
                    ]
                ),
                "disagreement_rate": (
                    disagreement_difficulty_counts[
                        difficulty
                    ]
                    / difficulty_counts[
                        difficulty
                    ]
                    if difficulty_counts[
                        difficulty
                    ]
                    else 0.0
                ),
            }
            for difficulty in [
                "easy",
                "hard",
            ]
        },
        "permissive_hypothetical_count": (
            len(permissive_hypotheticals)
        ),
        "hypothetical_flip_count": (
            len(hypothetical_flips)
        ),
        "hypothetical_flip_rate": (
            len(hypothetical_flips)
            / len(permissive_hypotheticals)
            if permissive_hypotheticals
            else 0.0
        ),
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print(
        f"Overall agreement: "
        f"{metrics['agreement_rate']:.1%}"
    )

    print(
        f"Disagreements: "
        f"{metrics['disagreement_count']}/"
        f"{metrics['total_items']}"
    )

    print("\nTransitions:")

    for transition, count in (
        metrics["transitions"].items()
    ):
        print(
            f"  {transition}: {count}"
        )

    print("\nBy difficulty:")

    for difficulty in [
        "easy",
        "hard",
    ]:
        row = metrics[
            "difficulty"
        ][difficulty]

        print(
            f"  {difficulty}: "
            f"{row['disagreements']}/"
            f"{row['total']} "
            f"({row['disagreement_rate']:.1%})"
        )

    print(
        "\nPermissive hypothetical flips:"
        f"\n  {metrics[
            'hypothetical_flip_count'
        ]}/"
        f"{metrics[
            'permissive_hypothetical_count'
        ]} "
        f"({metrics[
            'hypothetical_flip_rate'
        ]:.1%})"
    )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()