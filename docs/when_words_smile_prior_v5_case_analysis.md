# V5 Method Comparison Case Analysis

The case analysis uses `v5_full` as the target method because it tests the learned soft affect prior branch.

Local plots are saved in:

```text
outputs/when_words_smile_prior_v5/comparison_plots
```

These images compare baseline, V2, V3, V4 global, V5 full, V5 global, and V5 prior-only where available.

## Most Improved Cases

Improvement is computed as:

```text
baseline_case_mae - target_case_mae
```

| Case | Baseline MAE | Target MAE | Improvement | Text |
|---:|---:|---:|---:|---|
| 967 | 0.706772 | 0.615423 | 0.091349 | What about me? |
| 354 | 0.659555 | 0.584414 | 0.075141 | Look at that! |
| 428 | 0.557187 | 0.483399 | 0.073788 | really? |
| 420 | 0.574107 | 0.501164 | 0.072943 | and she won't tell me. |
| 131 | 0.569472 | 0.497544 | 0.071928 | I tell you what, if you get ready now, I'll let you play it at the wedding. |

V5 full improves many expressive dialogue samples, including question-like and exclamation-like utterances.

## Weak Or Regressed Cases

| Case | Baseline MAE | Target MAE | Improvement | Text |
|---:|---:|---:|---:|---|
| 966 | 0.325092 | 0.398720 | -0.073629 | No. |
| 718 | 0.338737 | 0.395081 | -0.056343 | You really think so? |
| 1105 | 0.249526 | 0.303604 | -0.054078 | Oh! |
| 1076 | 0.199678 | 0.249898 | -0.050220 | drawing you. |
| 1115 | 0.356102 | 0.405258 | -0.049157 | No. |

The weak cases remain short and context-dependent. This is consistent across V3, V4, and V5.

## Takeaway

The learned soft affect prior is stronger than the hand-crafted rule prior, but it still cannot solve extremely short utterances without dialogue context. This supports the next direction: combine learned affect prior with context-aware gating or a mixture-of-experts selector.
