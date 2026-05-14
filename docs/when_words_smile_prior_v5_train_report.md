# When Words Smile Prior V5 Training Report

## Method

V5 replaces the hand-crafted soft affect prior with a learned text-to-affect prior network. The prior network receives keyword-score features, text statistics, and optional frozen BERT sentence embeddings, then predicts a latent affect distribution, confidence, and intensity.

```text
z_text = PriorNet(x_text)
a = softmax(W_a z_text)
c = sigmoid(W_c z_text)
s = sigmoid(W_s z_text)
p'_t = p_t + g_global(t) * r_global(t) + c * g_prior(t) * r_prior(t)
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
- Affect dim: `8`
- BERT model: `disabled`
- Dynamic loss weight: `0.08`
- Residual regularization weight: `0.0005`
- Prior gate regularization weight: `0.0002`
- Entropy weight: `0.0`
- Confidence weight: `0.0001`
- Best val MAE: `0.361135`
- Output: `outputs/when_words_smile_prior_v5/test_learned_affect_prior_fusion.pt`
- Checkpoint: `outputs/when_words_smile_prior_v5/learned_affect_prior_fusion_v5.pt`

## Test Learned Prior Statistics

- Mean confidence: `0.863518`
- Median confidence: `0.882384`

## Hard Rule Counts For Reference Only

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
| 0 | 0.415521 | 0.379342 |
| 10 | 0.393812 | 0.364469 |
| 20 | 0.390111 | 0.362600 |
| 30 | 0.388004 | 0.361437 |
| 40 | 0.385643 | 0.361135 |
| 50 | 0.383384 | 0.361864 |
| 60 | 0.382487 | 0.361376 |
| 70 | 0.382465 | 0.362705 |
| 80 | 0.381623 | 0.362061 |
| 90 | 0.380024 | 0.363961 |
| 100 | 0.379618 | 0.364621 |
| 110 | 0.378792 | 0.363762 |
| 120 | 0.378568 | 0.363950 |
| 130 | 0.378020 | 0.365449 |
| 140 | 0.377852 | 0.366345 |
| 150 | 0.376429 | 0.365043 |
| 159 | 0.376476 | 0.366574 |