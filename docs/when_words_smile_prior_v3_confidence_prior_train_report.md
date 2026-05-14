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
- Ablation: `confidence_prior`
- Train size: `1200`
- Epochs: `160`
- Batch size: `96`
- Learning rate: `0.002`
- Hidden size: `64`
- BERT model: `disabled`
- Dynamic loss weight: `0.08`
- Residual regularization weight: `0.0005`
- Best val MAE: `0.361196`
- Output: `outputs/when_words_smile_prior_v3/test_prior_fusion_confidence_prior.pt`
- Checkpoint: `outputs/when_words_smile_prior_v3/prior_fusion_v3_confidence_prior.pt`

## Test Prior Counts

- calm: `17`
- concern: `8`
- happy: `2`
- neutral: `1464`
- sad: `1`
- surprise: `8`

## History

| Epoch | Train Loss | Val MAE |
|---:|---:|---:|
| 0 | 0.417012 | 0.380037 |
| 10 | 0.398689 | 0.365968 |
| 20 | 0.390805 | 0.362585 |
| 30 | 0.388884 | 0.361196 |
| 40 | 0.386714 | 0.361372 |
| 50 | 0.384766 | 0.361419 |
| 60 | 0.384077 | 0.362534 |
| 70 | 0.383804 | 0.362428 |
| 80 | 0.382044 | 0.362551 |
| 90 | 0.382325 | 0.363888 |
| 100 | 0.381630 | 0.363453 |
| 110 | 0.380682 | 0.364663 |
| 120 | 0.381407 | 0.362881 |
| 130 | 0.380030 | 0.362988 |
| 140 | 0.380309 | 0.363610 |
| 150 | 0.379118 | 0.364065 |
| 159 | 0.378769 | 0.364764 |