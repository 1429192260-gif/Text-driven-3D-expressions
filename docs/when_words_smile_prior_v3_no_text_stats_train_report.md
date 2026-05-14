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
- Ablation: `no_text_stats`
- Train size: `1200`
- Epochs: `160`
- Batch size: `96`
- Learning rate: `0.002`
- Hidden size: `64`
- BERT model: `disabled`
- Dynamic loss weight: `0.08`
- Residual regularization weight: `0.0005`
- Best val MAE: `0.362205`
- Output: `outputs/when_words_smile_prior_v3/test_prior_fusion_no_text_stats.pt`
- Checkpoint: `outputs/when_words_smile_prior_v3/prior_fusion_v3_no_text_stats.pt`

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
| 0 | 0.417058 | 0.380101 |
| 10 | 0.398370 | 0.365880 |
| 20 | 0.390741 | 0.363108 |
| 30 | 0.388871 | 0.362205 |
| 40 | 0.386790 | 0.362239 |
| 50 | 0.384645 | 0.362248 |
| 60 | 0.384234 | 0.363604 |
| 70 | 0.384023 | 0.364008 |
| 80 | 0.382113 | 0.363651 |
| 90 | 0.382091 | 0.363329 |
| 100 | 0.381945 | 0.363424 |
| 110 | 0.381057 | 0.362687 |
| 120 | 0.381451 | 0.364342 |
| 130 | 0.380130 | 0.363360 |
| 140 | 0.380714 | 0.363486 |
| 150 | 0.379196 | 0.364274 |
| 159 | 0.378964 | 0.364544 |