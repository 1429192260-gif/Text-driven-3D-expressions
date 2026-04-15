# Per-Emotion Error Analysis for Text-Only Front-Ends

- setting: manual-only source-holdout test set

## Parameter MAE by Emotion

| Emotion | learned | hf_raw | hf_rules_v4 |
| --- | ---: | ---: | ---: |
| happy | 0.073741 | 0.028583 | 0.005929 |
| sad | 0.102878 | 0.042756 | 0.015070 |
| angry | 0.119988 | 0.108771 | 0.032428 |
| surprise | 0.085003 | 0.112874 | 0.052957 |
| disgust | 0.098146 | 0.077289 | 0.019594 |
| concern | 0.081933 | 0.062709 | 0.010472 |
| bored | 0.041373 | 0.063873 | 0.004988 |
| calm | 0.081792 | 0.025098 | 0.001514 |

## Emotion Accuracy by Emotion

| Emotion | learned | hf_raw | hf_rules_v4 |
| --- | ---: | ---: | ---: |
| happy | 0.400000 | 0.600000 | 1.000000 |
| sad | 0.200000 | 0.800000 | 1.000000 |
| angry | 0.200000 | 0.000000 | 1.000000 |
| surprise | 0.600000 | 0.200000 | 0.800000 |
| disgust | 0.400000 | 0.200000 | 1.000000 |
| concern | 0.200000 | 0.000000 | 1.000000 |
| bored | 0.800000 | 0.000000 | 1.000000 |
| calm | 0.200000 | 0.600000 | 1.000000 |

## Notes

- `param_mae` measures final expression parameter error after passing through the fixed backend.
- `emotion_acc` measures whether the front-end predicted the correct emotion category on that emotion subset.
- This analysis helps identify which fine-grained emotions benefit most from Chinese rule enhancement.
