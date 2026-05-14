# When Words Smile Prior V4 Experiment Report

## 1. Motivation

V3 showed that decoder-side gated residual fusion is effective, but it also exposed a weakness: hard emotion labels and hard intensity values derived from keyword rules can be noisy on short dialogue text.

V4 was designed to test an uncertainty-aware soft prior:

```text
text -> emotion probability distribution + confidence + soft intensity
```

Instead of forcing each sentence into a single emotion category, V4 converts keyword evidence into:

- a 7-way soft emotion distribution,
- a confidence score,
- a confidence-weighted soft intensity.

The intended fusion form is:

```text
p'_t = p_t + g_global(t) * r_global(t) + c_text * g_prior(t) * r_prior(t)
```

The global branch keeps the stable residual correction discovered in V3. The prior branch is weakened when the text-derived prior is uncertain.

## 2. Main Results

| Method | MAE | RMSE | Smoothness | Acceleration | PPL |
|---|---:|---:|---:|---:|---:|
| When Words Smile baseline | 0.379475 | 0.563167 | 0.386786 | 0.663121 | 262.39 |
| V2 learned multiplicative adapter | 0.372688 | 0.549068 | 0.359706 | 0.616814 | 205.29 |
| V3 full prior-fusion | 0.358984 | 0.524042 | 0.293237 | 0.501486 | 152.39 |
| V3 best ablation: no emotion | 0.358055 | 0.522169 | 0.280578 | 0.479446 | 143.99 |
| V4 full uncertainty-aware prior | 0.359051 | 0.525381 | 0.326780 | 0.560040 | 160.77 |
| V4 global-only branch | 0.356114 | 0.519775 | 0.289948 | 0.495746 | 148.31 |

The best V4 result is the global-only branch. It achieves the lowest MAE so far:

```text
0.379475 -> 0.356114
```

This is about a 6.16% MAE reduction compared with the reproduced When Words Smile baseline.

## 3. V4 Ablation Results

| Variant | Description | MAE | RMSE | Smoothness | Acceleration | PPL |
|---|---|---:|---:|---:|---:|---:|
| V4 full | global branch + uncertainty-gated prior branch | 0.359051 | 0.525381 | 0.326780 | 0.560040 | 160.77 |
| V4 no prior branch | only global residual correction branch | 0.356114 | 0.519775 | 0.289948 | 0.495746 | 148.31 |
| V4 no uncertainty | prior branch always uses confidence 1.0 | 0.357158 | 0.520563 | 0.306734 | 0.525535 | 150.67 |
| V4 soft only | same as no uncertainty in this implementation | 0.357158 | 0.520563 | 0.306734 | 0.525535 | 150.67 |
| V4 no global branch | only uncertainty-gated prior branch | 0.367962 | 0.544393 | 0.359195 | 0.616008 | 207.23 |

## 4. Analysis

The result is mixed but useful.

First, the global-only V4 branch improves over all previous MAE results. This means the stronger two-input global residual correction structure is effective.

Second, the uncertainty-aware prior branch does not improve the final result. V4 full performs worse than V4 global-only and also worse than the best V3 ablations. This suggests that the current keyword-derived soft affect prior is still not reliable enough, even when confidence gating is added.

Third, the no-global-branch result is much worse. This confirms that the prior branch alone cannot generate useful corrections. It mainly works as a weak auxiliary cue, not as a strong standalone module.

Fourth, no-uncertainty and soft-only perform better than V4 full but worse than V4 global-only. This means the confidence formula did not solve the prior noise problem. The prior information itself is likely too weak or mismatched to the EmoAva dialogue distribution.

## 5. Current Best Version

The best practical model after V4 is:

```text
V4 global-only residual correction
```

It is not the most conceptually ambitious version, but it is the strongest measured version on MAE/RMSE:

- MAE: `0.356114`
- RMSE: `0.519775`
- PPL: `148.31`

Compared with V3 best ablation:

- MAE improves from `0.358055` to `0.356114`.
- RMSE improves from `0.522169` to `0.519775`.
- PPL is slightly worse than V3 best (`143.99` -> `148.31`), but still much better than baseline.

## 6. Research Interpretation

The most defensible interpretation is:

- The residual fusion framework is effective.
- Hard emotion labels are noisy.
- Current keyword-derived soft priors are also not strong enough.
- The next useful prior should come from a stronger affect estimator, not from more hand-crafted English keyword rules.

For the paper, V4 can be framed as an important negative/diagnostic experiment:

```text
We attempted uncertainty-aware soft prior injection. Ablation shows that the prior branch is still limited by weak text-prior quality, while the residual fusion backbone remains robust and achieves the best MAE.
```

This gives us a clear direction for the next stage:

```text
replace rule-based affect prior with learned emotion distribution / dialogue-context prior
```

## 7. Recommended Next Step

The next experiment should not keep tuning the keyword confidence formula. More useful options are:

- Use a learned affect classifier to provide probability distribution and confidence.
- Add dialogue context for short utterances such as "Oh!", "No.", and "Really?".
- Keep V4 global-only as the strong baseline correction module.
- Test whether learned soft affect priors can beat V4 global-only.
