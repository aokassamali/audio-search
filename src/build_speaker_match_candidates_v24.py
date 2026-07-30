import json
from collections import Counter, defaultdict

import numpy as np

from src.config import load_settings
from src.speech_act_gold import (
    SpeechActGoldArtifact,
)


SPEAKER_LAYERS = [0, 1, 2, 3]
TOP_MATCHES = 3


def normalize_rows(
    matrix: np.ndarray,
) -> np.ndarray:
    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(
        norms,
        1e-12,
    )

    return matrix / norms


def normalize_vector(
    vector: np.ndarray,
) -> np.ndarray:
    norm = float(
        np.linalg.norm(vector)
    )

    if norm < 1e-12:
        return vector

    return vector / norm


def build_group_embedding(
    turn_embeddings: np.ndarray,
) -> np.ndarray:
    layer_centroids = []

    for layer_index in SPEAKER_LAYERS:
        layer_vectors = turn_embeddings[
            :,
            layer_index,
            :,
        ]

        normalized_turns = normalize_rows(
            layer_vectors
        )

        centroid = normalized_turns.mean(
            axis=0
        )

        centroid = normalize_vector(
            centroid
        )

        layer_centroids.append(
            centroid
        )

    combined = np.concatenate(
        layer_centroids
    )

    return normalize_vector(combined)


def choose_representatives(
    items: list,
    maximum_count: int = 3,
) -> list:
    preferred = [
        item
        for item in items
        if 5.0 <= item.duration <= 30.0
    ]

    if len(preferred) < maximum_count:
        preferred = items

    selected = sorted(
        preferred,
        key=lambda item: (
            -item.duration,
            item.turn_id,
        ),
    )[:maximum_count]

    return [
        {
            "turn_id": item.turn_id,
            "start": item.start,
            "end": item.end,
            "duration": item.duration,
            "text": item.text,
        }
        for item in selected
    ]


def main() -> None:
    settings = load_settings()
    eval_dir = settings.paths.eval_dir

    gold_path = (
        eval_dir
        / "speech_act_gold_v24.json"
    )

    embeddings_path = (
        eval_dir
        / "wavlm_base_plus_v24.npy"
    )

    output_path = (
        eval_dir
        / "speaker_match_candidates_v24.json"
    )

    with gold_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        artifact = (
            SpeechActGoldArtifact.model_validate(
                json.load(file)
            )
        )

    embeddings = np.load(
        embeddings_path,
        allow_pickle=False,
    )

    if embeddings.shape != (
        len(artifact.items),
        13,
        768,
    ):
        print(
            "Embedding shape does not match "
            "the gold artifact."
        )
        return

    group_indices = defaultdict(list)

    for item_index, item in enumerate(
        artifact.items
    ):
        group_id = (
            f"{item.source_key}:"
            f"{item.speaker}"
        )

        group_indices[group_id].append(
            item_index
        )

    group_ids = sorted(
        group_indices
    )

    group_embeddings = {}
    group_records = {}

    for group_id in group_ids:
        indices = np.asarray(
            group_indices[group_id],
            dtype=int,
        )

        items = [
            artifact.items[index]
            for index in indices
        ]

        source_key = items[0].source_key
        speaker = items[0].speaker

        group_embeddings[group_id] = (
            build_group_embedding(
                embeddings[indices]
            )
        )

        label_counts = Counter(
            item.label_permissive
            for item in items
        )

        group_records[group_id] = {
            "group_id": group_id,
            "source_key": source_key,
            "speaker": speaker,
            "turn_count": len(items),
            "total_duration": float(
                sum(
                    item.duration
                    for item in items
                )
            ),
            "label_counts": dict(
                label_counts
            ),
            "question_share": float(
                label_counts["question"]
                / len(items)
            ),
            "representative_turns": (
                choose_representatives(
                    items
                )
            ),
        }

    similarity_lookup = {}

    for left_group in group_ids:
        left_source = group_records[
            left_group
        ]["source_key"]

        candidates = []

        for right_group in group_ids:
            if left_group == right_group:
                continue

            right_source = group_records[
                right_group
            ]["source_key"]

            if left_source == right_source:
                continue

            similarity = float(
                np.dot(
                    group_embeddings[left_group],
                    group_embeddings[right_group],
                )
            )

            candidates.append(
                {
                    "group_id": right_group,
                    "source_key": right_source,
                    "similarity": similarity,
                }
            )

        candidates.sort(
            key=lambda candidate: (
                -candidate["similarity"],
                candidate["group_id"],
            )
        )

        similarity_lookup[left_group] = (
            candidates[:TOP_MATCHES]
        )

    top_match = {
        group_id: matches[0]["group_id"]
        for group_id, matches
        in similarity_lookup.items()
        if matches
    }

    mutual_pairs = []
    seen_pairs = set()

    for left_group, right_group in (
        top_match.items()
    ):
        if (
            top_match.get(right_group)
            != left_group
        ):
            continue

        pair_key = tuple(
            sorted(
                [
                    left_group,
                    right_group,
                ]
            )
        )

        if pair_key in seen_pairs:
            continue

        seen_pairs.add(pair_key)

        similarity = next(
            candidate["similarity"]
            for candidate
            in similarity_lookup[left_group]
            if candidate["group_id"]
            == right_group
        )

        mutual_pairs.append(
            {
                "left_group": left_group,
                "right_group": right_group,
                "similarity": similarity,
            }
        )

    mutual_pairs.sort(
        key=lambda pair: (
            -pair["similarity"],
            pair["left_group"],
        )
    )

    source_files = {
        source_key: {
            "source_id": source.source_id,
            "audio_filename": (
                source.audio_filename
            ),
        }
        for source_key, source
        in settings.sources.items()
    }

    output = {
        "schema_version": "1.0",
        "purpose": (
            "Candidate generation only. "
            "Do not automatically merge "
            "speaker groups from similarity."
        ),
        "embedding_layers": SPEAKER_LAYERS,
        "source_files": source_files,
        "groups": group_records,
        "top_cross_source_matches": (
            similarity_lookup
        ),
        "mutual_top_pairs": mutual_pairs,
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

    print("SOURCE FILES")

    for source_key, record in (
        source_files.items()
    ):
        print(
            f"  {source_key}: "
            f"{record['audio_filename']}"
        )

    print(
        f"\nSpeaker groups: "
        f"{len(group_ids)}"
    )

    print(
        f"Mutual top pairs: "
        f"{len(mutual_pairs)}"
    )

    print("\nMUTUAL TOP PAIRS")

    for pair in mutual_pairs:
        left = pair["left_group"]
        right = pair["right_group"]

        left_count = group_records[
            left
        ]["turn_count"]

        right_count = group_records[
            right
        ]["turn_count"]

        print(
            f"  {pair['similarity']:.4f}  "
            f"{left} ({left_count})  <->  "
            f"{right} ({right_count})"
        )

    print("\nTOP MATCH PER GROUP")

    for group_id in group_ids:
        matches = similarity_lookup[
            group_id
        ]

        if not matches:
            continue

        first = matches[0]

        print(
            f"  {group_id} -> "
            f"{first['group_id']} "
            f"({first['similarity']:.4f})"
        )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()