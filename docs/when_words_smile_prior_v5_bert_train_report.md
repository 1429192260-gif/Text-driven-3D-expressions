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
- Epochs: `80`
- Batch size: `96`
- Learning rate: `0.002`
- Hidden size: `64`
- Affect dim: `8`
- BERT model: `models/bert-base-cased`
- Dynamic loss weight: `0.08`
- Residual regularization weight: `0.0005`
- Prior gate regularization weight: `0.0002`
- Entropy weight: `0.0`
- Confidence weight: `0.0001`
- Best val MAE: `0.363219`
- Output: `outputs/when_words_smile_prior_v5/test_learned_affect_prior_fusion_bert.pt`
- Checkpoint: `outputs/when_words_smile_prior_v5/learned_affect_prior_fusion_v5_bert.pt`

## Test Learned Prior Statistics

- Mean confidence: `0.943911`
- Median confidence: `0.985704`

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
| 0 | 0.416188 | 0.379209 |
| 10 | 0.393792 | 0.363862 |
| 20 | 0.388544 | 0.363219 |
| 30 | 0.384047 | 0.366597 |
| 40 | 0.381860 | 0.369314 |
| 50 | 0.378696 | 0.372889 |
| 60 | 0.375855 | 0.375837 |
| 70 | 0.373125 | 0.381048 |
| 79 | 0.373228 | 0.383086 |