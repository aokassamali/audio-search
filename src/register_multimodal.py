import json

import numpy as np

from src.rag import LLMClient
from src.register_classification import (
    REGISTER_SYSTEM_PROMPT,
    RegisterPrediction,
    TaxonomyVariant,
    build_register_schema,
)
from src.run_register_compact_prosody_arm import (
    COMPACT_FEATURES,
)


FEATURE_DISPLAY_NAMES = {
    "F0semitoneFrom27.5Hz_sma3nz_amean": (
        "pitch mean"
    ),
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": (
        "pitch variability"
    ),
    "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2": (
        "pitch range"
    ),
    "loudness_sma3_amean": (
        "loudness mean"
    ),
    "loudness_sma3_stddevNorm": (
        "loudness variability"
    ),
    "speaking_rate_wps": (
        "speaking rate"
    ),
    "silence_proportion": (
        "silence proportion"
    ),
    "pause_count": (
        "pause count"
    ),
}


MULTIMODAL_SYSTEM_PROMPT = (
    REGISTER_SYSTEM_PROMPT
    + """

You are also given prosodic measurements from the audio.

Each measurement is expressed as a z-score relative to the
other turns in this corpus:
- 0 means approximately average
- positive means above average
- negative means below average

Treat prosody as supporting evidence, not as a replacement for
the lexical meaning of the turn. Do not assume that high pitch,
loudness, or variability automatically implies any particular
label.
""".rstrip()
)


def compute_compact_z_scores(
    feature_items,
    turn_id: str,
) -> dict[str, float]:
    feature_matrix = np.asarray(
        [
            [
                item.features[name]
                for name in COMPACT_FEATURES
            ]
            for item in feature_items
        ],
        dtype=float,
    )

    feature_means = feature_matrix.mean(
        axis=0
    )

    feature_standard_deviations = (
        feature_matrix.std(axis=0)
    )

    feature_standard_deviations = (
        np.where(
            feature_standard_deviations == 0,
            1.0,
            feature_standard_deviations,
        )
    )

    target_index = next(
        index
        for index, item
        in enumerate(feature_items)
        if item.turn_id == turn_id
    )

    target_values = feature_matrix[
        target_index
    ]

    z_scores = (
        target_values - feature_means
    ) / feature_standard_deviations

    return {
        name: float(z_score)
        for name, z_score
        in zip(
            COMPACT_FEATURES,
            z_scores,
        )
    }


def format_prosody_summary(
    z_scores: dict[str, float],
) -> str:
    return "\n".join(
        (
            f"- {FEATURE_DISPLAY_NAMES[name]}: "
            f"{z_scores[name]:+.2f} SD"
        )
        for name in COMPACT_FEATURES
    )


def classify_text_plus_prosody(
    turn_id: str,
    text: str,
    z_scores: dict[str, float],
    taxonomy_variant: TaxonomyVariant,
    llm_client: LLMClient,
) -> RegisterPrediction:
    if taxonomy_variant == "strict":
        hypothetical_rule = (
            "Label hypothetical only when the "
            "turn contains an explicit marker "
            "such as suppose, assume, imagine, "
            "what if, or let's say. An explicit "
            "marker is necessary but not "
            "sufficient: if the hypothetical "
            "mainly sets up a substantive "
            "question, label the dominant speech "
            "act as question."
        )
    else:
        hypothetical_rule = (
            "Label hypothetical whenever the "
            "speaker entertains a proposition "
            "without asserting it, even when no "
            "explicit counterfactual marker appears."
        )

    user_prompt = (
        f"Taxonomy variant: {taxonomy_variant}\n"
        f"{hypothetical_rule}\n\n"
        f"Turn ID: {turn_id}\n"
        f"Turn text:\n{text}\n\n"
        f"Prosodic measurements:\n"
        f"{format_prosody_summary(z_scores)}"
    )

    raw_response = llm_client.generate(
        system_prompt=(
            MULTIMODAL_SYSTEM_PROMPT
        ),
        user_prompt=user_prompt,
        response_schema=(
            build_register_schema(turn_id)
        ),
        max_tokens=128,
    )

    return RegisterPrediction.model_validate(
        json.loads(raw_response)
    )