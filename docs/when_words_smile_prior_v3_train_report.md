# When Words Smile Prior V3 Training Report

## Method

V3 uses a lightweight decoder-side prior-fusion adapter. Compared with V2, it no longer learns only an emotion-time multiplicative factor. It combines base expression frames, text-derived emotion prior features, text statistics, and temporal position features to predict a gated residual sequence.

```text
h_t = Fusion(W_p p_t + W_e e + W_s s + W_x x_text + W_t tau_t)
r_t = tanh(alpha) * tanh(W_r h_t)
g_t = sigmoid(W_g h_t)
p'_t = p_t + g_t * r_t
```

Training loss:

```text
L = MAE(p', y) + lambda_dyn * MAE(Delta p', Delta y) + lambda_reg * ||r||^2
```

## Settings

- Dev prediction: `outputs/when_words_smile_prior_v1/dev_parallel_full.pt`
- Test prediction: `outputs/when_words_smile_repro/test_parallel_full.pt`
- Ablation: `full`
- Train size: `1200`
- Epochs: `160`
- Batch size: `96`
- Learning rate: `0.002`
- Hidden size: `64`
- Dynamic loss weight: `0.08`
- Residual regularization weight: `0.0005`
- Best val MAE: `0.361726`
- Output: `outputs/when_words_smile_prior_v3/test_prior_fusion.pt`
- Checkpoint: `outputs/when_words_smile_prior_v3/prior_fusion_v3.pt`

## Test Prior Counts

- angry: `44`
- calm: `180`
- concern: `234`
- happy: `102`
- neutral: `700`
- sad: `56`
- surprise: `184`

## History

| Epoch | Train Loss | Val MAE |
|---:|---:|---:|
| 0 | 0.417056 | 0.380086 |
| 10 | 0.398264 | 0.365923 |
| 20 | 0.390478 | 0.362992 |
| 30 | 0.388596 | 0.361726 |
| 40 | 0.386433 | 0.361989 |
| 50 | 0.384332 | 0.361972 |
| 60 | 0.383791 | 0.363128 |
| 70 | 0.383521 | 0.363683 |
| 80 | 0.381664 | 0.362928 |
| 90 | 0.381798 | 0.363259 |
| 100 | 0.381410 | 0.363387 |
| 110 | 0.380369 | 0.363204 |
| 120 | 0.381004 | 0.363586 |
| 130 | 0.379640 | 0.363505 |
| 140 | 0.380254 | 0.363589 |
| 150 | 0.378731 | 0.364603 |
| 159 | 0.378360 | 0.364893 |