# When Words Smile Prior V6 Experiment Report

## 1. Motivation

V4 showed that global residual correction is the strongest frame-wise backbone. V5 showed that learned affect prior is much stronger than rule-based prior, and slightly improves PPL, but does not beat the global branch on MAE.

V6 tests a direct mixture-of-experts idea:

```text
P_v6 = P_global + gamma * (P_prior - P_global)
```

where:

- `P_global` is the strong V4 global-only output.
- `P_prior` is the V5 learned-prior output.
- `gamma` is a learned gate trained on the dev split.

The goal is to let the model automatically decide when to trust the learned affect-prior output.

## 2. Main Results

| Method | MAE | RMSE | Smoothness | Acceleration | PPL |
|---|---:|---:|---:|---:|---:|
| When Words Smile baseline | 0.379475 | 0.563167 | 0.386786 | 0.663121 | 262.39 |
| V3 best: no emotion | 0.358055 | 0.522169 | 0.280578 | 0.479446 | 143.99 |
| V4 global-only | 0.356114 | 0.519775 | 0.289948 | 0.495746 | 148.31 |
| V5 full learned prior | 0.358160 | 0.522113 | 0.320896 | 0.550066 | 146.84 |
| V6 frame gate | 0.355470 | 0.519037 | 0.293978 | 0.502837 | 142.08 |
| V6 sample gate | 0.355414 | 0.519048 | 0.296180 | 0.506732 | 142.01 |

V6 sample gate is the current best overall result:

- best MAE: `0.355414`
- best PPL: `142.01`
- best RMSE is essentially tied with V6 frame gate.

Compared with the reproduced When Words Smile baseline:

- MAE improves from `0.379475` to `0.355414`.
- RMSE improves from `0.563167` to `0.519048`.
- PPL improves from `262.39` to `142.01`.

## 3. Why V6 Matters

Earlier experiments showed a tension:

- Global residual correction gives strong MAE/RMSE.
- Learned affect prior gives better distribution-level PPL but can hurt frame-wise metrics.

V6 addresses this tension with a learned gate. Instead of forcing one branch to dominate, it learns how much to move from the global output toward the learned-prior output.

This gives a more convincing innovation story:

```text
rule prior -> learned prior -> gated mixture between robust correction and affect-aware correction
```

## 4. Interpretation

The V6 result supports the usefulness of learned affect prior, but only when it is selectively fused.

The prior branch alone is not the final answer. The global branch alone is strong but misses some distribution-level benefit. The gate provides a controlled way to use learned prior information without fully trusting it.

This solves the main issue found in V4 and V5:

- V4 rule prior was too weak.
- V5 learned prior was stronger but unstable when directly fused.
- V6 selectively fuses learned prior and achieves the best combined result.

## 5. Current Best Method

The current recommended final method is:

```text
V6 sample-level mixture gate
```

It can be described as:

> A lightweight mixture-gated residual correction framework that combines a robust global correction branch with a learned text-affect prior branch for text-driven dynamic 3D expression generation.

## 6. Next Step

The next practical step is to generate visual demos using V6:

```text
baseline vs V4 global vs V5 learned prior vs V6 mixture gate
```

This will give us both strong quantitative results and a complete visual story for reporting.
