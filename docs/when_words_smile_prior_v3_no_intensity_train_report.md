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
- Ablation: `no_intensity`
- Train size: `1200`
- Epochs: `160`
- Batch size: `96`
- Learning rate: `0.002`
- Hidden size: `64`
- BERT model: `disabled`
- Dynamic loss weight: `0.08`
- Residual regularization weight: `0.0005`
- Best val MAE: `0.361565`
- Output: `outputs/when_words_smile_prior_v3/test_prior_fusion_no_intensity.pt`
- Checkpoint: `outputs/when_words_smile_prior_v3/prior_fusion_v3_no_intensity.pt`

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
| 0 | 0.417060 | 0.380086 |
| 10 | 0.398258 | 0.365941 |
| 20 | 0.390618 | 0.362926 |
| 30 | 0.388722 | 0.361595 |
| 40 | 0.386556 | 0.361565 |
| 50 | 0.384376 | 0.361696 |
| 60 | 0.383771 | 0.362761 |
| 70 | 0.383453 | 0.363257 |
| 80 | 0.381639 | 0.362610 |
| 90 | 0.381702 | 0.362977 |
| 100 | 0.381274 | 0.363610 |
| 110 | 0.380241 | 0.363060 |
| 120 | 0.380900 | 0.363706 |
| 130 | 0.379487 | 0.362967 |
| 140 | 0.380088 | 0.362673 |
| 150 | 0.378596 | 0.364176 |
| 159 | 0.378312 | 0.363864 |