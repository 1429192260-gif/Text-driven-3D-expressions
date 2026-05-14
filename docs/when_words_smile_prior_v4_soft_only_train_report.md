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
- Ablation: `soft_only`
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
- Best val MAE: `0.360516`
- Output: `outputs/when_words_smile_prior_v4/test_uncertainty_prior_fusion_soft_only.pt`
- Checkpoint: `outputs/when_words_smile_prior_v4/uncertainty_prior_fusion_v4_soft_only.pt`

## Test Prior Statistics

- Mean confidence: `1.000000`
- Median confidence: `1.000000`
- Mean soft intensity: `0.450611`
- Median soft intensity: `0.416667`

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
| 0 | 0.415151 | 0.378115 |
| 10 | 0.393352 | 0.364903 |
| 20 | 0.389384 | 0.362142 |
| 30 | 0.388526 | 0.361088 |
| 40 | 0.386972 | 0.361000 |
| 50 | 0.385205 | 0.360516 |
| 60 | 0.384227 | 0.360929 |
| 70 | 0.382512 | 0.360628 |
| 80 | 0.381433 | 0.360996 |
| 90 | 0.381542 | 0.360971 |
| 100 | 0.381007 | 0.362783 |
| 110 | 0.380298 | 0.362378 |
| 120 | 0.379466 | 0.362131 |
| 130 | 0.379608 | 0.363839 |
| 140 | 0.379421 | 0.363580 |
| 150 | 0.377492 | 0.363122 |
| 159 | 0.377372 | 0.364703 |