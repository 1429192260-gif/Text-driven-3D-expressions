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
- Ablation: `no_global_branch`
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
- Best val MAE: `0.370924`
- Output: `outputs/when_words_smile_prior_v4/test_uncertainty_prior_fusion_no_global_branch.pt`
- Checkpoint: `outputs/when_words_smile_prior_v4/uncertainty_prior_fusion_v4_no_global_branch.pt`

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
| 0 | 0.415748 | 0.380864 |
| 10 | 0.408984 | 0.375644 |
| 20 | 0.404918 | 0.372963 |
| 30 | 0.403834 | 0.371909 |
| 40 | 0.402263 | 0.371293 |
| 50 | 0.400250 | 0.371259 |
| 60 | 0.400154 | 0.370924 |
| 70 | 0.398873 | 0.371062 |
| 80 | 0.397669 | 0.371091 |
| 90 | 0.398357 | 0.371055 |
| 100 | 0.398137 | 0.371022 |
| 110 | 0.397597 | 0.371033 |
| 120 | 0.397166 | 0.371085 |
| 130 | 0.397377 | 0.371201 |
| 140 | 0.397700 | 0.371438 |
| 150 | 0.395756 | 0.371418 |
| 159 | 0.395967 | 0.371426 |