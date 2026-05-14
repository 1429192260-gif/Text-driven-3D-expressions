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
- Ablation: `no_prior`
- Train size: `1200`
- Epochs: `160`
- Batch size: `96`
- Learning rate: `0.002`
- Hidden size: `64`
- Dynamic loss weight: `0.08`
- Residual regularization weight: `0.0005`
- Best val MAE: `0.361363`
- Output: `outputs/when_words_smile_prior_v3/test_prior_fusion_no_prior.pt`
- Checkpoint: `outputs/when_words_smile_prior_v3/prior_fusion_v3_no_prior.pt`

## Test Prior Counts

- neutral: `1500`

## History

| Epoch | Train Loss | Val MAE |
|---:|---:|---:|
| 0 | 0.417014 | 0.380030 |
| 10 | 0.398556 | 0.365806 |
| 20 | 0.390849 | 0.362506 |
| 30 | 0.389293 | 0.361363 |
| 40 | 0.387268 | 0.361676 |
| 50 | 0.385340 | 0.361405 |
| 60 | 0.384727 | 0.361902 |
| 70 | 0.384517 | 0.362404 |
| 80 | 0.382813 | 0.362445 |
| 90 | 0.383002 | 0.362233 |
| 100 | 0.382806 | 0.361789 |
| 110 | 0.381695 | 0.363325 |
| 120 | 0.382372 | 0.363077 |
| 130 | 0.381039 | 0.362745 |
| 140 | 0.381555 | 0.362528 |
| 150 | 0.380240 | 0.363510 |
| 159 | 0.380029 | 0.364372 |