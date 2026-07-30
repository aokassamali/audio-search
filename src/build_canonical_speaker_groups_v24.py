import json
from collections import Counter, defaultdict

from src.config import load_settings
from src.speech_act_gold import (
    SpeechActGoldArtifact,
)


class UnionFind:
    def __init__(
        self,
        values: list[str],
    ) -> None:
        self.parent = {
            value: value
            for value in values
        }

        self.rank = {
            value: 0
            for value in values
        }

    def find(
        self,
        value: str,
    ) -> str:
        parent = self.parent[value]

        if parent != value:
            self.parent[value] = self.find(
                parent
            )

        return self.parent[value]

    def union(
        self,
        left: str,
        right: str,
    ) -> None:
        left_root = self.find(left)
        right_root = self.find(right)

        if left_root == right_root:
            return

        left_rank = self.rank[left_root]
        right_rank = self.rank[right_root]

        if left_rank < right_rank:
            self.parent[left_root] = right_root
            return

        if left_rank > right_rank:
            self.parent[right_root] = left_root
            return

        self.parent[right_root] = left_root
        self.rank[left_root] += 1


def source_from_group(
    group_id: str,
) -> str:
    return group_id.split(
        ":",
        maxsplit=1,
    )[0]


def main() -> None:
    settings = load_settings()
    eval_dir = settings.paths.eval_dir

    gold_path = (
        eval_dir
        / "speech_act_gold_v24.json"
    )

    reviews_path = (
        eval_dir
        / "speaker_match_reviews_v24.json"
    )

    output_path = (
        eval_dir
        / "canonical_speaker_groups_v24.json"
    )

    with gold_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        gold = (
            SpeechActGoldArtifact.model_validate(
                json.load(file)
            )
        )

    with reviews_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        reviews = json.load(file)

    provisional_groups = sorted(
        {
            (
                f"{item.source_key}:"
                f"{item.speaker}"
            )
            for item in gold.items
        }
    )

    known_groups = set(
        provisional_groups
    )

    union_find = UnionFind(
        provisional_groups
    )

    decisions = reviews.get(
        "decisions",
        [],
    )

    decision_counts = Counter(
        decision["decision"]
        for decision in decisions
    )

    unknown_references = []

    for decision in decisions:
        left = decision["left_group"]
        right = decision["right_group"]

        if (
            left not in known_groups
            or right not in known_groups
        ):
            unknown_references.append(
                decision["pair_key"]
            )
            continue

        if decision["decision"] == "same":
            union_find.union(
                left,
                right,
            )

    members_by_root = defaultdict(list)

    for group_id in provisional_groups:
        root = union_find.find(
            group_id
        )

        members_by_root[root].append(
            group_id
        )

    sorted_components = sorted(
        (
            sorted(members)
            for members
            in members_by_root.values()
        ),
        key=lambda members: (
            members[0],
            len(members),
        ),
    )

    canonical_mapping = {}
    component_records = []

    turn_counts = Counter(
        (
            f"{item.source_key}:"
            f"{item.speaker}"
        )
        for item in gold.items
    )

    same_source_collisions = []

    for component_index, members in enumerate(
        sorted_components,
        start=1,
    ):
        canonical_id = (
            f"person_{component_index:03d}"
        )

        for member in members:
            canonical_mapping[member] = (
                canonical_id
            )

        source_counts = Counter(
            source_from_group(member)
            for member in members
        )

        duplicate_sources = {
            source: count
            for source, count
            in source_counts.items()
            if count > 1
        }

        record = {
            "canonical_id": canonical_id,
            "members": members,
            "member_count": len(members),
            "sources": sorted(
                source_counts
            ),
            "turn_count": sum(
                turn_counts[member]
                for member in members
            ),
            "same_source_duplicates": (
                duplicate_sources
            ),
        }

        component_records.append(record)

        if duplicate_sources:
            same_source_collisions.append(
                record
            )

    different_conflicts = []
    unsure_inside_components = []

    for decision in decisions:
        left = decision["left_group"]
        right = decision["right_group"]

        if (
            left not in canonical_mapping
            or right not in canonical_mapping
        ):
            continue

        same_component = (
            canonical_mapping[left]
            == canonical_mapping[right]
        )

        if (
            decision["decision"]
            == "different"
            and same_component
        ):
            different_conflicts.append(
                decision
            )

        if (
            decision["decision"]
            == "unsure"
            and same_component
        ):
            unsure_inside_components.append(
                decision
            )

    merged_components = [
        component
        for component in component_records
        if component["member_count"] > 1
    ]

    largest_component = max(
        (
            component["member_count"]
            for component in component_records
        ),
        default=0,
    )

    validation_passed = (
        not unknown_references
        and not different_conflicts
    )

    output = {
        "schema_version": "1.0",
        "validation_passed": (
            validation_passed
        ),
        "decision_counts": dict(
            decision_counts
        ),
        "provisional_group_count": len(
            provisional_groups
        ),
        "canonical_group_count": len(
            component_records
        ),
        "merged_component_count": len(
            merged_components
        ),
        "largest_component_size": (
            largest_component
        ),
        "mapping": canonical_mapping,
        "components": component_records,
        "unknown_review_references": (
            unknown_references
        ),
        "different_conflicts": (
            different_conflicts
        ),
        "unsure_inside_components": (
            unsure_inside_components
        ),
        "same_source_collisions": (
            same_source_collisions
        ),
    }

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

    print(
        "Decisions:"
        f"\n  same: "
        f"{decision_counts['same']}"
        f"\n  different: "
        f"{decision_counts['different']}"
        f"\n  unsure: "
        f"{decision_counts['unsure']}"
    )

    print(
        f"\nProvisional groups: "
        f"{len(provisional_groups)}"
    )

    print(
        f"Canonical groups: "
        f"{len(component_records)}"
    )

    print(
        f"Merged components: "
        f"{len(merged_components)}"
    )

    print(
        f"Largest component: "
        f"{largest_component}"
    )

    print(
        f"Different-decision conflicts: "
        f"{len(different_conflicts)}"
    )

    print(
        f"Same-source collisions: "
        f"{len(same_source_collisions)}"
    )

    print("\nMERGED COMPONENTS")

    for component in sorted(
        merged_components,
        key=lambda record: (
            -record["member_count"],
            record["canonical_id"],
        ),
    ):
        print(
            f"\n{component['canonical_id']} "
            f"({component['turn_count']} turns)"
        )

        for member in component["members"]:
            print(f"  {member}")

        if component[
            "same_source_duplicates"
        ]:
            print(
                "  REVIEW SAME-SOURCE "
                "DUPLICATES: "
                f"{component[
                    'same_source_duplicates'
                ]}"
            )

    print(
        "\nValidation: "
        + (
            "PASSED"
            if validation_passed
            else "FAILED"
        )
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()