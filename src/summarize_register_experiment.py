import json
from pathlib import Path

from src.config import load_settings


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def format_score(value: float) -> str:
    return f"{value:.3f}"


def format_percent(value: float) -> str:
    return f"{value:.1%}"


def main() -> None:
    settings = load_settings()
    eval_dir = settings.paths.eval_dir

    text = load_json(
        eval_dir
        / "register_text_metrics.json"
    )

    full_prosody = load_json(
        eval_dir
        / "register_prosody_metrics.json"
    )

    compact_prosody = load_json(
        eval_dir
        / "register_prosody_compact_metrics.json"
    )

    multimodal = load_json(
        eval_dir
        / "register_multimodal_metrics.json"
    )

    shuffled = load_json(
        eval_dir
        / "register_multimodal_shuffled_metrics.json"
    )

    stacker = load_json(
        eval_dir
        / "register_stacker_metrics.json"
    )

    construct = load_json(
        eval_dir
        / "register_construct_metrics.json"
    )

    confidence = load_json(
        eval_dir
        / "register_confidence_metrics.json"
    )

    summary = {
        "research_question": (
            "Does prosodic information improve "
            "register classification beyond "
            "lexical cues alone in formal speech?"
        ),
        "null_hypothesis": (
            "Prosodic information does not "
            "improve register classification "
            "beyond text alone."
        ),
        "dataset": {
            "source": "SCOTUS oral argument",
            "gold_turns": (
                construct["total_items"]
            ),
            "annotators": 1,
            "observed_classes": [
                "assertion",
                "hypothetical",
                "question",
                "characterization",
            ],
            "unobserved_classes": [
                "hyperbole",
                "joke",
            ],
        },
        "experiment_a": {
            "text_only": {
                "permissive": {
                    "accuracy": (
                        text["permissive"][
                            "accuracy"
                        ]
                    ),
                    "macro_f1": (
                        text["permissive"][
                            "macro_f1_supported_classes"
                        ]
                    ),
                },
                "strict": {
                    "accuracy": (
                        text["strict"][
                            "accuracy"
                        ]
                    ),
                    "macro_f1": (
                        text["strict"][
                            "macro_f1_supported_classes"
                        ]
                    ),
                },
            },
            "prosody_only": {
                "full_egemaps": {
                    "accuracy": (
                        full_prosody[
                            "accuracy"
                        ]
                    ),
                    "macro_f1": (
                        full_prosody[
                            "macro_f1"
                        ]
                    ),
                },
                "compact_features": {
                    "accuracy": (
                        compact_prosody[
                            "accuracy"
                        ]
                    ),
                    "macro_f1": (
                        compact_prosody[
                            "macro_f1"
                        ]
                    ),
                },
                "majority_accuracy": (
                    full_prosody[
                        "majority_baseline_accuracy"
                    ]
                ),
            },
            "prompt_injection": {
                "permissive": {
                    "text_accuracy": (
                        multimodal[
                            "permissive"
                        ][
                            "text_only_accuracy"
                        ]
                    ),
                    "real_prosody_accuracy": (
                        shuffled[
                            "permissive"
                        ][
                            "real_prosody_accuracy"
                        ]
                    ),
                    "shuffled_prosody_accuracy": (
                        shuffled[
                            "permissive"
                        ][
                            "shuffled_prosody_accuracy"
                        ]
                    ),
                },
                "strict": {
                    "text_accuracy": (
                        multimodal[
                            "strict"
                        ][
                            "text_only_accuracy"
                        ]
                    ),
                    "real_prosody_accuracy": (
                        shuffled[
                            "strict"
                        ][
                            "real_prosody_accuracy"
                        ]
                    ),
                    "shuffled_prosody_accuracy": (
                        shuffled[
                            "strict"
                        ][
                            "shuffled_prosody_accuracy"
                        ]
                    ),
                },
            },
            "stacker": {
                "direct_text": (
                    stacker["direct_text"]
                ),
                "text_stack": (
                    stacker["text_stack"]
                ),
                "text_prosody_stack": (
                    stacker[
                        "text_prosody_stack"
                    ]
                ),
                "prosody_vs_text": (
                    stacker[
                        "prosody_vs_text_stack"
                    ]
                ),
            },
        },
        "experiment_b": {
            "overall_agreement_rate": (
                construct["agreement_rate"]
            ),
            "disagreements": (
                construct["disagreement_count"]
            ),
            "transitions": (
                construct["transitions"]
            ),
            "hypothetical_flip_rate": (
                construct[
                    "hypothetical_flip_rate"
                ]
            ),
            "easy_disagreement_rate": (
                construct["difficulty"][
                    "easy"
                ][
                    "disagreement_rate"
                ]
            ),
            "hard_disagreement_rate": (
                construct["difficulty"][
                    "hard"
                ][
                    "disagreement_rate"
                ]
            ),
        },
        "experiment_c": {
            model_name: {
                "difficulty_auc": (
                    metrics[
                        "difficulty_auc_from_uncertainty"
                    ]
                ),
                "easy_mean_confidence": (
                    metrics[
                        "easy_mean_confidence"
                    ]
                ),
                "hard_mean_confidence": (
                    metrics[
                        "hard_mean_confidence"
                    ]
                ),
            }
            for model_name, metrics
            in confidence.items()
        },
        "conclusion": (
            "The null hypothesis was not rejected. "
            "Within this SCOTUS corpus, matched "
            "prosodic features did not improve "
            "register classification beyond text. "
            "Prosody-only models were weak, real "
            "prosody did not meaningfully beat a "
            "shuffled-prosody placebo, and adding "
            "prosody to the text stacker caused "
            "six regressions and zero fixes."
        ),
        "scope_limit": (
            "This result supports the narrower "
            "claim that prosody did not help under "
            "this corpus, taxonomy, feature set, "
            "modeling approach, and sample size. "
            "It does not establish that prosody "
            "never helps register classification."
        ),
        "limitations": [
            (
                "Only 80 labeled turns from one "
                "formal SCOTUS argument."
            ),
            (
                "Single annotator with no "
                "inter-annotator agreement."
            ),
            (
                "No joke or hyperbole examples "
                "appeared in the random gold set."
            ),
            (
                "Speaker-grouped folds were small "
                "and had uneven class balance."
            ),
            (
                "Only hand-crafted eGeMAPS and "
                "compact prosodic features were "
                "tested."
            ),
        ],
    }

    json_path = (
        eval_dir
        / "register_experiment_summary.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    text_permissive = (
        summary["experiment_a"][
            "text_only"
        ]["permissive"]
    )

    text_strict = (
        summary["experiment_a"][
            "text_only"
        ]["strict"]
    )

    full = summary["experiment_a"][
        "prosody_only"
    ]["full_egemaps"]

    compact = summary["experiment_a"][
        "prosody_only"
    ]["compact_features"]

    text_stack = summary["experiment_a"][
        "stacker"
    ]["text_stack"]

    multimodal_stack = summary[
        "experiment_a"
    ]["stacker"]["text_prosody_stack"]

    stack_comparison = summary[
        "experiment_a"
    ]["stacker"]["prosody_vs_text"]

    markdown_lines = [
        "# v2.3 Register Experiment Summary",
        "",
        "## Research question",
        "",
        summary["research_question"],
        "",
        "## Null hypothesis",
        "",
        summary["null_hypothesis"],
        "",
        "## Experiment A — Ablation",
        "",
        "### Text-only LLM",
        "",
        (
            "- Permissive: accuracy "
            f"{format_score(text_permissive['accuracy'])}, "
            "macro F1 "
            f"{format_score(text_permissive['macro_f1'])}"
        ),
        (
            "- Strict: accuracy "
            f"{format_score(text_strict['accuracy'])}, "
            "macro F1 "
            f"{format_score(text_strict['macro_f1'])}"
        ),
        "",
        "### Prosody-only",
        "",
        (
            "- Full eGeMAPS: accuracy "
            f"{format_score(full['accuracy'])}, "
            "macro F1 "
            f"{format_score(full['macro_f1'])}"
        ),
        (
            "- Compact features: accuracy "
            f"{format_score(compact['accuracy'])}, "
            "macro F1 "
            f"{format_score(compact['macro_f1'])}"
        ),
        (
            "- Majority-class accuracy: "
            f"{format_score(
                summary['experiment_a'][
                    'prosody_only'
                ]['majority_accuracy']
            )}"
        ),
        "",
        "### Text plus prosody",
        "",
        (
            "- Permissive prompt arm: real and "
            "shuffled prosody both achieved "
            f"{format_score(
                summary['experiment_a'][
                    'prompt_injection'
                ]['permissive'][
                    'real_prosody_accuracy'
                ]
            )} accuracy."
        ),
        (
            "- Strict prompt arm: real prosody "
            "achieved "
            f"{format_score(
                summary['experiment_a'][
                    'prompt_injection'
                ]['strict'][
                    'real_prosody_accuracy'
                ]
            )} versus "
            f"{format_score(
                summary['experiment_a'][
                    'prompt_injection'
                ]['strict'][
                    'shuffled_prosody_accuracy'
                ]
            )} with shuffled prosody."
        ),
        (
            "- Text-only stacker: accuracy "
            f"{format_score(text_stack['accuracy'])}, "
            "macro F1 "
            f"{format_score(text_stack['macro_f1'])}"
        ),
        (
            "- Text plus prosody stacker: accuracy "
            f"{format_score(
                multimodal_stack['accuracy']
            )}, macro F1 "
            f"{format_score(
                multimodal_stack['macro_f1']
            )}"
        ),
        (
            "- Adding prosody produced "
            f"{stack_comparison['fixes']} fixes "
            "and "
            f"{stack_comparison['regressions']} "
            "regressions."
        ),
        "",
        "## Experiment B — Construct validity",
        "",
        (
            "- Overall strict/permissive agreement: "
            f"{format_percent(
                summary['experiment_b'][
                    'overall_agreement_rate'
                ]
            )}"
        ),
        (
            "- Permissive hypothetical flip rate: "
            f"{format_percent(
                summary['experiment_b'][
                    'hypothetical_flip_rate'
                ]
            )}"
        ),
        (
            "- Easy-turn disagreement rate: "
            f"{format_percent(
                summary['experiment_b'][
                    'easy_disagreement_rate'
                ]
            )}"
        ),
        (
            "- Hard-turn disagreement rate: "
            f"{format_percent(
                summary['experiment_b'][
                    'hard_disagreement_rate'
                ]
            )}"
        ),
        "",
        "The taxonomy was stable overall, but the "
        "hypothetical construct was highly "
        "definition-sensitive.",
        "",
        "## Experiment C — Confidence and difficulty",
        "",
        (
            "- Text permissive difficulty AUROC: "
            f"{format_score(
                summary['experiment_c'][
                    'text_permissive'
                ]['difficulty_auc']
            )}"
        ),
        (
            "- Text strict difficulty AUROC: "
            f"{format_score(
                summary['experiment_c'][
                    'text_strict'
                ]['difficulty_auc']
            )}"
        ),
        (
            "- Multimodal permissive difficulty "
            "AUROC: "
            f"{format_score(
                summary['experiment_c'][
                    'multimodal_permissive'
                ]['difficulty_auc']
            )}"
        ),
        (
            "- Multimodal strict difficulty AUROC: "
            f"{format_score(
                summary['experiment_c'][
                    'multimodal_strict'
                ]['difficulty_auc']
            )}"
        ),
        "",
        (
            "Confidence did not track human "
            "difficulty; all AUROCs were "
            "approximately chance."
        ),
        "",
        "## Conclusion",
        "",
        summary["conclusion"],
        "",
        summary["scope_limit"],
        "",
        "## Limitations",
        "",
    ]

    markdown_lines.extend(
        f"- {limitation}"
        for limitation
        in summary["limitations"]
    )

    markdown_path = (
        eval_dir
        / "register_experiment_summary.md"
    )

    markdown_path.write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )

    print(
        "Text-only permissive: "
        f"{format_score(
            text_permissive['accuracy']
        )} accuracy / "
        f"{format_score(
            text_permissive['macro_f1']
        )} macro F1"
    )

    print(
        "Compact prosody-only: "
        f"{format_score(
            compact['accuracy']
        )} accuracy / "
        f"{format_score(
            compact['macro_f1']
        )} macro F1"
    )

    print(
        "Text stacker: "
        f"{format_score(
            text_stack['accuracy']
        )} accuracy / "
        f"{format_score(
            text_stack['macro_f1']
        )} macro F1"
    )

    print(
        "Text + prosody stacker: "
        f"{format_score(
            multimodal_stack['accuracy']
        )} accuracy / "
        f"{format_score(
            multimodal_stack['macro_f1']
        )} macro F1"
    )

    print(
        "Construct agreement: "
        f"{format_percent(
            summary['experiment_b'][
                'overall_agreement_rate'
            ]
        )}"
    )

    print(
        "Hypothetical flip rate: "
        f"{format_percent(
            summary['experiment_b'][
                'hypothetical_flip_rate'
            ]
        )}"
    )

    print(f"\nSaved: {json_path}")
    print(f"Saved: {markdown_path}")


if __name__ == "__main__":
    main()