# V4 Method Comparison Case Analysis

The case analysis uses `v4_global` as the target method because it is the strongest V4 variant on MAE/RMSE.

Local plots are saved in:

```text
outputs/when_words_smile_prior_v4/comparison_plots
```

These images compare baseline, V2, V3 full, V3 no emotion, V4 full, and V4 global.

## Most Improved Cases

Improvement is computed as:

```text
baseline_case_mae - target_case_mae
```

| Case | Baseline MAE | Target MAE | Improvement | Text |
|---:|---:|---:|---:|---|
| 967 | 0.706772 | 0.615200 | 0.091572 | What about me? |
| 84 | 0.611611 | 0.534305 | 0.077306 | Come on, Ross. |
| 1444 | 0.531132 | 0.453867 | 0.077265 | You got me. |
| 354 | 0.659555 | 0.582303 | 0.077252 | Look at that! |
| 809 | 0.623116 | 0.546440 | 0.076675 | No, not the groom. |

V4 global improves expressive short dialogue samples where the baseline trajectory has relatively large error.

## Weak Or Regressed Cases

| Case | Baseline MAE | Target MAE | Improvement | Text |
|---:|---:|---:|---:|---|
| 966 | 0.325092 | 0.401567 | -0.076476 | No. |
| 718 | 0.338737 | 0.400550 | -0.061813 | You really think so? |
| 1245 | 0.496816 | 0.555844 | -0.059028 | It's okay, come on in. |
| 463 | 0.377999 | 0.433328 | -0.055329 | What then? |
| 1115 | 0.356102 | 0.406920 | -0.050818 | No. |

The weak cases are still dominated by very short or context-dependent utterances. This is consistent with the V3 finding: text-only generation is unstable when the text lacks speaker and scene context.

## Takeaway

V4 global improves the average error, but the remaining failures are strongly related to missing context. This supports using V4 global as the current strongest correction backbone and using dialogue context or a stronger learned affect estimator as the next improvement direction.
