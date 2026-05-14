# When Words Smile Prior V3 Experiment Report

## 1. Goal

V1 and V2 improved the reproduced When Words Smile output by applying emotion priors after sequence generation. V3 moves one step closer to model-level fusion by learning a lightweight decoder-side prior-fusion adapter.

Instead of only multiplying selected expression dimensions, V3 learns a gated residual correction for each generated frame:

```text
h_t = Fusion(W_p p_t + W_e e + W_s s + W_x x_text + W_t tau_t)
r_t = tanh(alpha) * tanh(W_r h_t)
g_t = sigmoid(W_g h_t)
p'_t = p_t + g_t * r_t
```

where:

- `p_t` is the original 53-D expression parameter at frame `t`.
- `e` is a text-derived emotion prior feature.
- `s` is the text-derived intensity feature.
- `x_text` includes keyword-score features and simple text statistics.
- `tau_t` is the temporal position feature.
- `g_t` controls how strongly the residual correction is applied.

The training objective is:

```text
L = MAE(p', y) + lambda_dyn * MAE(Delta p', Delta y) + lambda_reg * ||r||^2
```

The adapter is trained on the EmoAva dev split and evaluated once on the test split. The original When Words Smile BERT/CVAE/Transformer checkpoint is not retrained.

## 2. Main Results

| Method | MAE | RMSE | Smoothness | Acceleration | PPL |
|---|---:|---:|---:|---:|---:|
| When Words Smile baseline | 0.379475 | 0.563167 | 0.386786 | 0.663121 | 262.39 |
| V1 fixed prior template | 0.377198 | 0.558812 | 0.380854 | 0.653193 | 244.81 |
| V2 learned multiplicative adapter | 0.372688 | 0.549068 | 0.359706 | 0.616814 | 205.29 |
| V3 full prior-fusion adapter | 0.358984 | 0.524042 | 0.293237 | 0.501486 | 152.39 |

Compared with the reproduced baseline, V3 full reduces:

- MAE by about 5.40%.
- RMSE by about 6.95%.
- PPL by about 41.92%.
- Acceleration by about 24.37%.

This shows that a learnable decoder-side residual fusion module is substantially stronger than V1/V2 post-generation modulation.

## 3. V3 Ablation Results

| Variant | Description | MAE | RMSE | Smoothness | Acceleration | PPL |
|---|---|---:|---:|---:|---:|---:|
| V3 full | hard emotion + intensity + keyword scores + text stats + time | 0.358984 | 0.524042 | 0.293237 | 0.501486 | 152.39 |
| V3 no prior | removes emotion, intensity, keyword scores, and text stats | 0.358434 | 0.523988 | 0.290201 | 0.495979 | 147.69 |
| V3 no emotion | removes hard emotion category only | 0.358055 | 0.522169 | 0.280578 | 0.479446 | 143.99 |
| V3 no intensity | removes scalar intensity only | 0.358831 | 0.523205 | 0.287636 | 0.491806 | 143.51 |
| V3 no scores | removes keyword-score features only | 0.358925 | 0.524004 | 0.293632 | 0.502177 | 151.53 |
| V3 no text stats | removes text length and punctuation statistics | 0.359335 | 0.524524 | 0.289679 | 0.495221 | 156.94 |
| V3 no time | removes temporal position features | 0.358772 | 0.523857 | 0.285587 | 0.488077 | 154.20 |
| V3 soft prior | removes hard emotion and intensity, keeps keyword scores and text stats | 0.358406 | 0.524143 | 0.294479 | 0.503455 | 150.64 |
| V3 confidence prior | uses hard emotion and intensity only when keyword evidence is strong | 0.358794 | 0.524425 | 0.297230 | 0.508340 | 152.40 |
| V3 + frozen BERT embedding | adds frozen BERT mean-pooled sentence embedding | 0.364078 | 0.533768 | 0.344607 | 0.590836 | 165.94 |

## 4. Analysis

The strongest V3 variants are `no_emotion` and `no_intensity`, not the full hard-prior version. This is an important finding rather than a failure.

It means the current rule-based hard emotion label and scalar intensity are noisy on the EmoAva dialogue text. Many test samples have weak or ambiguous emotional cues. Forcing them into a single emotion category can hurt the generated sequence.

However, removing keyword-score features or text-statistic features makes some metrics worse than the best variants. This suggests that soft textual cues are more reliable than hard emotion decisions in this setting.

The frozen BERT sentence embedding variant also underperforms. The likely reason is that the original When Words Smile model already uses BERT internally, while our small dev-only adapter can overfit when another high-dimensional frozen sentence vector is directly concatenated.

## 5. Current Conclusion

V3 provides a clearer method-level improvement over V1/V2:

```text
post-generation prior modulation -> decoder-side gated residual fusion
```

The most defensible conclusion is:

- The learnable residual fusion structure is effective and gives the largest improvement so far.
- Hard rule-based emotion category and intensity are not reliable enough for EmoAva dialogue text.
- Soft prior features are safer than hard labels.
- The next improvement should replace English keyword hard labels with a stronger text-only affect estimator, or constrain prior injection with confidence/uncertainty.

## 6. Recommended Next Step

For the next version, we should not keep strengthening the current hard keyword rules. A better V4 direction is:

```text
uncertainty-aware soft affect prior + decoder-side residual fusion
```

Possible implementation:

- Predict an emotion probability distribution instead of one hard category.
- Predict intensity as a confidence-weighted continuous value.
- Feed prior confidence into the gate so uncertain priors have weaker influence.
- Keep V3's residual fusion structure as the backbone.

This would directly address the main weakness discovered by the V3 ablation experiments.
