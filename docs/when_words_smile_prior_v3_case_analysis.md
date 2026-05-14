# V3 Method Comparison Case Analysis

The case analysis compares the reproduced When Words Smile baseline, V2, V3 full, and the strongest V3 ablation variant `v3_no_emotion`.

The plotted files are saved locally in:

```text
outputs/when_words_smile_prior_v3/comparison_plots
```

These images are not committed because `outputs/` is ignored. They are intended for local inspection and screenshot-based reporting.

## Most Improved Cases

Improvement is computed as:

```text
baseline_case_mae - target_case_mae
```

| Case | Baseline MAE | Target MAE | Improvement | Text |
|---:|---:|---:|---:|---|
| 967 | 0.706772 | 0.620177 | 0.086595 | What about me? |
| 428 | 0.557187 | 0.471230 | 0.085957 | really? |
| 354 | 0.659555 | 0.578511 | 0.081044 | Look at that! |
| 196 | 0.539916 | 0.462956 | 0.076960 | This game is kinda fun. |
| 536 | 0.550435 | 0.476581 | 0.073854 | yeah, maybe. |

These examples are mostly short but expressive utterances with question marks, exclamation marks, or clear pragmatic cues. V3's residual fusion adapter can correct the baseline trajectory more effectively on this type of sequence.

## Weak Or Regressed Cases

| Case | Baseline MAE | Target MAE | Improvement | Text |
|---:|---:|---:|---:|---|
| 1076 | 0.199678 | 0.281490 | -0.081812 | drawing you. |
| 1105 | 0.249526 | 0.329520 | -0.079994 | Oh! |
| 966 | 0.325092 | 0.403452 | -0.078361 | No. |
| 1141 | 0.242164 | 0.298702 | -0.056538 | Love me. |
| 718 | 0.338737 | 0.388866 | -0.050129 | You really think so? |

The weak cases are often extremely short utterances. Their emotional meaning depends heavily on context, speaker identity, and scene information. Text-only modeling is therefore unstable on these cases.

## Takeaway

The case analysis supports two conclusions:

- V3 improves many expressive dialogue samples by learning a smoother residual correction over the original generated sequence.
- Text-only generation is still weak when the text is too short or context-dependent, which motivates adding uncertainty-aware priors or dialogue context in the next version.
