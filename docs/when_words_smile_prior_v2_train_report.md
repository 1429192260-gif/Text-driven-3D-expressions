# Prior Adapter V2 Training Report

- Dev prediction: `outputs\when_words_smile_prior_v1\dev_parallel_full.pt`
- Train size: `1200`
- Val size: `300`
- Epochs: `200`
- Best val MAE: `0.378464`
- Output: `outputs\when_words_smile_prior_v2\test_prior_adapter.pt`
- Checkpoint: `outputs\when_words_smile_prior_v2\adapter_v2.pt`

## Emotion Counts On Test

- angry: `44`
- calm: `180`
- concern: `234`
- happy: `102`
- neutral: `700`
- sad: `56`
- surprise: `184`

## History

| Epoch | Train MAE | Val MAE |
|---:|---:|---:|
| 0 | 0.382745 | 0.384290 |
| 10 | 0.376120 | 0.379544 |
| 20 | 0.375163 | 0.378757 |
| 30 | 0.374934 | 0.378589 |
| 40 | 0.374868 | 0.378505 |
| 50 | 0.374833 | 0.378467 |
| 60 | 0.374818 | 0.378464 |
| 70 | 0.374805 | 0.378481 |
| 80 | 0.374796 | 0.378487 |
| 90 | 0.374788 | 0.378492 |
| 100 | 0.374781 | 0.378505 |
| 110 | 0.374774 | 0.378514 |
| 120 | 0.374768 | 0.378523 |
| 130 | 0.374762 | 0.378531 |
| 140 | 0.374757 | 0.378541 |
| 150 | 0.374754 | 0.378550 |
| 160 | 0.374750 | 0.378562 |
| 170 | 0.374748 | 0.378568 |
| 180 | 0.374745 | 0.378576 |
| 190 | 0.374743 | 0.378582 |
| 199 | 0.374742 | 0.378583 |