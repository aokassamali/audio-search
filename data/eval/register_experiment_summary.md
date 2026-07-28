# v2.3 Register Experiment Summary

## Research question

Does prosodic information improve register classification beyond lexical cues alone in formal speech?

## Null hypothesis

Prosodic information does not improve register classification beyond text alone.

## Experiment A — Ablation

### Text-only LLM

- Permissive: accuracy 0.662, macro F1 0.608
- Strict: accuracy 0.738, macro F1 0.609

### Prosody-only

- Full eGeMAPS: accuracy 0.225, macro F1 0.175
- Compact features: accuracy 0.325, macro F1 0.288
- Majority-class accuracy: 0.475

### Text plus prosody

- Permissive prompt arm: real and shuffled prosody both achieved 0.637 accuracy.
- Strict prompt arm: real prosody achieved 0.750 versus 0.738 with shuffled prosody.
- Text-only stacker: accuracy 0.688, macro F1 0.629
- Text plus prosody stacker: accuracy 0.613, macro F1 0.522
- Adding prosody produced 0 fixes and 6 regressions.

## Experiment B — Construct validity

- Overall strict/permissive agreement: 92.5%
- Permissive hypothetical flip rate: 75.0%
- Easy-turn disagreement rate: 0.0%
- Hard-turn disagreement rate: 20.7%

The taxonomy was stable overall, but the hypothetical construct was highly definition-sensitive.

## Experiment C — Confidence and difficulty

- Text permissive difficulty AUROC: 0.507
- Text strict difficulty AUROC: 0.498
- Multimodal permissive difficulty AUROC: 0.519
- Multimodal strict difficulty AUROC: 0.503

Confidence did not track human difficulty; all AUROCs were approximately chance.

## Conclusion

The null hypothesis was not rejected. Within this SCOTUS corpus, matched prosodic features did not improve register classification beyond text. Prosody-only models were weak, real prosody did not meaningfully beat a shuffled-prosody placebo, and adding prosody to the text stacker caused six regressions and zero fixes.

This result supports the narrower claim that prosody did not help under this corpus, taxonomy, feature set, modeling approach, and sample size. It does not establish that prosody never helps register classification.

## Limitations

- Only 80 labeled turns from one formal SCOTUS argument.
- Single annotator with no inter-annotator agreement.
- No joke or hyperbole examples appeared in the random gold set.
- Speaker-grouped folds were small and had uneven class balance.
- Only hand-crafted eGeMAPS and compact prosodic features were tested.
