# When Words Smile Prior V4 Training Report

## Method

V4 uses uncertainty-aware soft affect priors. Instead of injecting one hard emotion label and one hard intensity score, it converts keyword evidence into an emotion probability distribution, a confidence score, and a confidence-weighted soft intensity.

```text
p'_t = p_t + g_global(t) * r_global(t) + c_text * g_prior(t) * r_prior(t)
```

The global branch keeps the stable residual correction found in V3, while the prior branch is weakened when the text prior is uncertain.

## Settings

- Dev prediction: `outputs/when_words_smile_prior_v1/dev_parallel_full.pt`
- Test prediction: `outputs/when_words_smile_repro/test_parallel_full.pt`
- Ablation: `full`
- Train size: `1200`
- Epochs: `160`
- Batch size: `96`
- Learning rate: `0.002`
- Hidden size: `64`
- Temperature: `0.85`
- Evidence scale: `3.0`
- Dynamic loss weight: `0.08`
- Residual regularization weight: `0.0005`
- Prior gate regularization weight: `0.0002`
- Best val MAE: `0.361583`
- Output: `outputs/when_words_smile_prior_v4/test_uncertainty_prior_fusion.pt`
- Checkpoint: `outputs/when_words_smile_prior_v4/uncertainty_prior_fusion_v4.pt`

## Test Prior Statistics

- Mean confidence: `0.143529`
- Median confidence: `0.070648`
- Mean soft intensity: `0.077552`
- Median soft intensity: `0.029437`

## Hard Rule Counts For Reference

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
| 0 | 0.415353 | 0.379118 |
| 10 | 0.394395 | 0.364947 |
| 20 | 0.390287 | 0.362407 |
| 30 | 0.389111 | 0.361583 |
| 40 | 0.386887 | 0.361803 |
| 50 | 0.384879 | 0.362560 |
| 60 | 0.383561 | 0.362581 |
| 70 | 0.382140 | 0.363062 |
| 80 | 0.380814 | 0.361951 |
| 90 | 0.380912 | 0.362686 |
| 100 | 0.380258 | 0.363291 |
| 110 | 0.379616 | 0.363590 |
| 120 | 0.378916 | 0.363439 |
| 130 | 0.378793 | 0.364039 |
| 140 | 0.378790 | 0.363910 |
| 150 | 0.376915 | 0.363646 |
| 159 | 0.376961 | 0.364932 |