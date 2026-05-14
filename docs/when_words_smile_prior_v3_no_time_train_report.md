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
- Ablation: `no_time`
- Train size: `1200`
- Epochs: `160`
- Batch size: `96`
- Learning rate: `0.002`
- Hidden size: `64`
- BERT model: `disabled`
- Dynamic loss weight: `0.08`
- Residual regularization weight: `0.0005`
- Best val MAE: `0.361544`
- Output: `outputs/when_words_smile_prior_v3/test_prior_fusion_no_time.pt`
- Checkpoint: `outputs/when_words_smile_prior_v3/prior_fusion_v3_no_time.pt`

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
| 0 | 0.417073 | 0.380128 |
| 10 | 0.397909 | 0.365487 |
| 20 | 0.390560 | 0.362838 |
| 30 | 0.388712 | 0.361544 |
| 40 | 0.386620 | 0.361578 |
| 50 | 0.384399 | 0.361657 |
| 60 | 0.383821 | 0.361937 |
| 70 | 0.383639 | 0.362759 |
| 80 | 0.381706 | 0.362682 |
| 90 | 0.381651 | 0.362321 |
| 100 | 0.381532 | 0.361767 |
| 110 | 0.380333 | 0.362405 |
| 120 | 0.380888 | 0.362827 |
| 130 | 0.379578 | 0.362607 |
| 140 | 0.379984 | 0.363164 |
| 150 | 0.378407 | 0.364274 |
| 159 | 0.378232 | 0.363700 |