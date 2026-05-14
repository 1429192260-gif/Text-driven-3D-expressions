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
- Ablation: `soft_prior`
- Train size: `1200`
- Epochs: `160`
- Batch size: `96`
- Learning rate: `0.002`
- Hidden size: `64`
- BERT model: `disabled`
- Dynamic loss weight: `0.08`
- Residual regularization weight: `0.0005`
- Best val MAE: `0.361105`
- Output: `outputs/when_words_smile_prior_v3/test_prior_fusion_soft_prior.pt`
- Checkpoint: `outputs/when_words_smile_prior_v3/prior_fusion_v3_soft_prior.pt`

## Test Prior Counts

- neutral: `1500`

## History

| Epoch | Train Loss | Val MAE |
|---:|---:|---:|
| 0 | 0.417011 | 0.380023 |
| 10 | 0.398475 | 0.365718 |
| 20 | 0.390680 | 0.362347 |
| 30 | 0.388884 | 0.361105 |
| 40 | 0.386729 | 0.361470 |
| 50 | 0.384852 | 0.361118 |
| 60 | 0.384092 | 0.361895 |
| 70 | 0.383895 | 0.361565 |
| 80 | 0.382303 | 0.362173 |
| 90 | 0.382502 | 0.363556 |
| 100 | 0.381936 | 0.363317 |
| 110 | 0.380955 | 0.363347 |
| 120 | 0.381509 | 0.362849 |
| 130 | 0.380214 | 0.362553 |
| 140 | 0.380669 | 0.362247 |
| 150 | 0.379174 | 0.365414 |
| 159 | 0.378855 | 0.364753 |