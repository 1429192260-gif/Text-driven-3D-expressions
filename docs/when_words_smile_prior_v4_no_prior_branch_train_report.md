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
- Ablation: `no_prior_branch`
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
- Best val MAE: `0.359551`
- Output: `outputs/when_words_smile_prior_v4/test_uncertainty_prior_fusion_no_prior_branch.pt`
- Checkpoint: `outputs/when_words_smile_prior_v4/uncertainty_prior_fusion_v4_no_prior_branch.pt`

## Test Prior Statistics

- Mean confidence: `0.000000`
- Median confidence: `0.000000`
- Mean soft intensity: `0.000000`
- Median soft intensity: `0.000000`

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
| 0 | 0.415382 | 0.379266 |
| 10 | 0.394778 | 0.364897 |
| 20 | 0.391116 | 0.362542 |
| 30 | 0.390417 | 0.361381 |
| 40 | 0.389182 | 0.361170 |
| 50 | 0.387654 | 0.361436 |
| 60 | 0.387039 | 0.360411 |
| 70 | 0.385922 | 0.360071 |
| 80 | 0.385214 | 0.359995 |
| 90 | 0.385676 | 0.359837 |
| 100 | 0.385550 | 0.360010 |
| 110 | 0.385072 | 0.359913 |
| 120 | 0.384756 | 0.359758 |
| 130 | 0.384642 | 0.359615 |
| 140 | 0.384837 | 0.359728 |
| 150 | 0.383388 | 0.359551 |
| 159 | 0.383346 | 0.360131 |