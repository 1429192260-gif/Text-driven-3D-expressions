# V6 Method Comparison Case Analysis

The case analysis uses `v6_sample` as the target method because it is the current best overall method.

Local plots are saved in:

```text
outputs/when_words_smile_prior_v6/comparison_plots
```

These images compare baseline, V2, V3, V4 global, V5 learned prior, and V6 mixture gate outputs.

## Most Improved Cases

Improvement is computed as:

```text
baseline_case_mae - target_case_mae
```

| Case | Baseline MAE | Target MAE | Improvement | Text |
|---:|---:|---:|---:|---|
| 967 | 0.706772 | 0.614523 | 0.092249 | What about me? |
| 354 | 0.659555 | 0.582345 | 0.077210 | Look at that! |
| 295 | 0.569731 | 0.493885 | 0.075847 | The moment's over. |
| 428 | 0.557187 | 0.481764 | 0.075423 | really? |
| 84 | 0.611611 | 0.536233 | 0.075378 | Come on, Ross. |

V6 improves expressive dialogue samples where moving partially from the global correction output toward the learned-prior output is beneficial.

## Weak Or Regressed Cases

| Case | Baseline MAE | Target MAE | Improvement | Text |
|---:|---:|---:|---:|---|
| 966 | 0.325092 | 0.399895 | -0.074803 | No. |
| 718 | 0.338737 | 0.398155 | -0.059418 | You really think so? |
| 1245 | 0.496816 | 0.547754 | -0.050937 | It's okay, come on in. |
| 1115 | 0.356102 | 0.405734 | -0.049632 | No. |
| 1105 | 0.249526 | 0.297302 | -0.047777 | Oh! |

The weak cases remain very short or context-dependent. This suggests that the next major improvement likely requires dialogue context rather than another single-sentence gate.

## Takeaway

V6 provides the best quantitative result so far by selectively combining the robust global branch and learned affect-prior branch. It gives a stronger final method than either branch alone.
