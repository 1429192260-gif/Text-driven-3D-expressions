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
- Ablation: `no_global_branch`
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
- Best val MAE: `0.360682`
- Output: `outputs/when_words_smile_prior_v5/test_learned_affect_prior_fusion_no_global_branch.pt`
- Checkpoint: `outputs/when_words_smile_prior_v5/learned_affect_prior_fusion_v5_no_global_branch.pt`

## Test Learned Prior Statistics

- Mean confidence: `0.957069`
- Median confidence: `0.983695`

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
| 0 | 0.415874 | 0.381038 |
| 10 | 0.397024 | 0.365921 |
| 20 | 0.392034 | 0.363138 |
| 30 | 0.390225 | 0.362112 |
| 40 | 0.388476 | 0.361593 |
| 50 | 0.386343 | 0.361040 |
| 60 | 0.385789 | 0.361238 |
| 70 | 0.385980 | 0.360743 |
| 80 | 0.385323 | 0.360682 |
| 90 | 0.383840 | 0.361401 |
| 100 | 0.383687 | 0.362169 |
| 110 | 0.383221 | 0.361040 |
| 120 | 0.382896 | 0.361142 |
| 130 | 0.382473 | 0.362686 |
| 140 | 0.382674 | 0.362415 |
| 150 | 0.381095 | 0.361150 |
| 159 | 0.381332 | 0.362184 |