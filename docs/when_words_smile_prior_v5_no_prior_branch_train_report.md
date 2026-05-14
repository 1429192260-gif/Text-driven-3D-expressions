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
- Ablation: `no_prior_branch`
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
- Best val MAE: `0.359402`
- Output: `outputs/when_words_smile_prior_v5/test_learned_affect_prior_fusion_no_prior_branch.pt`
- Checkpoint: `outputs/when_words_smile_prior_v5/learned_affect_prior_fusion_v5_no_prior_branch.pt`

## Test Learned Prior Statistics

- Mean confidence: `0.000001`
- Median confidence: `0.000001`

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
| 0 | 0.415525 | 0.379361 |
| 10 | 0.394774 | 0.364366 |
| 20 | 0.391414 | 0.363028 |
| 30 | 0.389750 | 0.361731 |
| 40 | 0.388325 | 0.360854 |
| 50 | 0.386447 | 0.360890 |
| 60 | 0.386085 | 0.359700 |
| 70 | 0.386642 | 0.359772 |
| 80 | 0.385915 | 0.359975 |
| 90 | 0.384784 | 0.359705 |
| 100 | 0.384506 | 0.359807 |
| 110 | 0.384337 | 0.359851 |
| 120 | 0.384319 | 0.360329 |
| 130 | 0.383990 | 0.359981 |
| 140 | 0.384200 | 0.359998 |
| 150 | 0.382922 | 0.359402 |
| 159 | 0.383095 | 0.359432 |