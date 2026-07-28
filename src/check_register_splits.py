import json
from collections import Counter

import numpy as np
from sklearn.model_selection import (
    StratifiedGroupKFold,
)

from src.config import load_settings
from src.extract_register_features import (
    RegisterFeaturesArtifact,
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

    labels = np.array(
        [
            item.label_permissive
            for item in artifact.items
        ]
    )

    speakers = np.array(
        [
            item.speaker
            for item in artifact.items
        ]
    )

    placeholder_features = np.zeros(
        (len(artifact.items), 1)
    )

    splitter = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=23,
    )

    all_labels = set(labels)

    for fold, (
        train_indices,
        test_indices,
    ) in enumerate(
        splitter.split(
            placeholder_features,
            labels,
            groups=speakers,
        ),
        start=1,
    ):
        train_speakers = set(
            speakers[train_indices]
        )

        test_speakers = set(
            speakers[test_indices]
        )

        missing_train_labels = (
            all_labels
            - set(labels[train_indices])
        )

        print(f"\nFOLD {fold}")
        print(
            "Train counts:",
            dict(
                Counter(
                    labels[train_indices]
                )
            ),
        )
        print(
            "Test counts:",
            dict(
                Counter(
                    labels[test_indices]
                )
            ),
        )
        print(
            "Test speakers:",
            sorted(test_speakers),
        )
        print(
            "Speaker overlap:",
            train_speakers
            & test_speakers,
        )
        print(
            "Missing train labels:",
            sorted(missing_train_labels),
        )


if __name__ == "__main__":
    main()