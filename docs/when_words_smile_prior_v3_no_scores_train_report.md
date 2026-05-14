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
- Ablation: `no_scores`
- Train size: `1200`
- Epochs: `160`
- Batch size: `96`
- Learning rate: `0.002`
- Hidden size: `64`
- BERT model: `disabled`
- Dynamic loss weight: `0.08`
- Residual regularization weight: `0.0005`
- Best val MAE: `0.361648`
- Output: `outputs/when_words_smile_prior_v3/test_prior_fusion_no_scores.pt`
- Checkpoint: `outputs/when_words_smile_prior_v3/prior_fusion_v3_no_scores.pt`

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
| 0 | 0.417058 | 0.380092 |
| 10 | 0.398279 | 0.365931 |
| 20 | 0.390529 | 0.362994 |
| 30 | 0.388701 | 0.361648 |
| 40 | 0.386554 | 0.361950 |
| 50 | 0.384549 | 0.361792 |
| 60 | 0.383971 | 0.362819 |
| 70 | 0.383754 | 0.363474 |
| 80 | 0.381847 | 0.362817 |
| 90 | 0.382022 | 0.362919 |
| 100 | 0.381670 | 0.362953 |
| 110 | 0.380624 | 0.362988 |
| 120 | 0.381327 | 0.363152 |
| 130 | 0.379956 | 0.362958 |
| 140 | 0.380520 | 0.362986 |
| 150 | 0.378997 | 0.363858 |
| 159 | 0.378749 | 0.364216 |