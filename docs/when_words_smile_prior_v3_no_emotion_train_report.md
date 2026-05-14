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
- Ablation: `no_emotion`
- Train size: `1200`
- Epochs: `160`
- Batch size: `96`
- Learning rate: `0.002`
- Hidden size: `64`
- BERT model: `disabled`
- Dynamic loss weight: `0.08`
- Residual regularization weight: `0.0005`
- Best val MAE: `0.361091`
- Output: `outputs/when_words_smile_prior_v3/test_prior_fusion_no_emotion.pt`
- Checkpoint: `outputs/when_words_smile_prior_v3/prior_fusion_v3_no_emotion.pt`

## Test Prior Counts

- neutral: `1500`

## History

| Epoch | Train Loss | Val MAE |
|---:|---:|---:|
| 0 | 0.417008 | 0.380020 |
| 10 | 0.398488 | 0.365688 |
| 20 | 0.390681 | 0.362350 |
| 30 | 0.388894 | 0.361236 |
| 40 | 0.386814 | 0.361393 |
| 50 | 0.384868 | 0.361091 |
| 60 | 0.384163 | 0.361953 |
| 70 | 0.383979 | 0.361531 |
| 80 | 0.382352 | 0.362038 |
| 90 | 0.382564 | 0.363098 |
| 100 | 0.381905 | 0.362628 |
| 110 | 0.380895 | 0.362935 |
| 120 | 0.381396 | 0.362443 |
| 130 | 0.380186 | 0.362006 |
| 140 | 0.380478 | 0.361902 |
| 150 | 0.379153 | 0.363646 |
| 159 | 0.378862 | 0.364338 |