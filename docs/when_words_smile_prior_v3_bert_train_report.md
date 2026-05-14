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
- BERT model: `models/bert-base-cased`
- Dynamic loss weight: `0.08`
- Residual regularization weight: `0.0005`
- Best val MAE: `0.366296`
- Output: `outputs/when_words_smile_prior_v3/test_prior_fusion_bert.pt`
- Checkpoint: `outputs/when_words_smile_prior_v3/prior_fusion_v3_bert.pt`

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
| 0 | 0.415756 | 0.379689 |
| 10 | 0.393736 | 0.366296 |
| 20 | 0.384280 | 0.368260 |
| 30 | 0.378619 | 0.373203 |
| 40 | 0.373893 | 0.379759 |
| 50 | 0.370428 | 0.384615 |
| 60 | 0.368467 | 0.389847 |
| 70 | 0.367140 | 0.390609 |
| 80 | 0.364072 | 0.403576 |
| 90 | 0.363438 | 0.402500 |
| 100 | 0.362738 | 0.401763 |
| 110 | 0.362217 | 0.406422 |
| 120 | 0.360556 | 0.413402 |
| 130 | 0.358770 | 0.418639 |
| 140 | 0.359819 | 0.419533 |
| 150 | 0.358316 | 0.429761 |
| 159 | 0.358100 | 0.427168 |