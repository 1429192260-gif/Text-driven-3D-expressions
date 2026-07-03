# Experiment Summary

- epochs: 60
- batch_size: 8
- seed: 42

| Run | Setting | Test MAE | Test RMSE | Active MAE | Test Loss |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline MLP | text embedding + emotion + intensity | 0.035630 | 0.088019 | 0.143153 | 0.007747 |
| Semantic MLP | baseline + semantic cue features | 0.035277 | 0.088445 | 0.142557 | 0.007823 |
| Prior Only | expression prior + residual refinement without semantic cues | 0.004571 | 0.012246 | 0.019477 | 0.000150 |
| Prior-Fusion | semantic features + expression prior + residual refinement | 0.004881 | 0.012749 | 0.020648 | 0.000163 |

## Per-Emotion Test MAE

| Group | Baseline MLP | Semantic MLP | Prior Only | Prior-Fusion |
| --- | ---: | ---: | ---: | ---: |
| angry | 0.067115 | 0.062228 | 0.004324 | 0.004562 |
| bored | 0.027482 | 0.026154 | 0.004076 | 0.005482 |
| calm | 0.005629 | 0.006201 | 0.001141 | 0.001197 |
| concern | 0.032003 | 0.029918 | 0.003648 | 0.003441 |
| disgust | 0.034822 | 0.032785 | 0.006938 | 0.007123 |
| happy | 0.018521 | 0.022069 | 0.006149 | 0.006387 |
| sad | 0.042695 | 0.047101 | 0.004722 | 0.005340 |
| surprise | 0.056777 | 0.055759 | 0.005566 | 0.005516 |

## Per-Intensity Test MAE

| Group | Baseline MLP | Semantic MLP | Prior Only | Prior-Fusion |
| --- | ---: | ---: | ---: | ---: |
| medium | 0.027016 | 0.028890 | 0.004280 | 0.004751 |
| strong | 0.043540 | 0.043734 | 0.004883 | 0.004938 |
| weak | 0.046942 | 0.040783 | 0.004901 | 0.005121 |

## Writing Notes

- `Semantic MLP` vs `Baseline MLP` isolates the value of handcrafted semantic cue features.
- `Prior Only` vs `Baseline MLP` isolates the effect of expression priors without semantic enhancement.
- `Prior-Fusion` vs `Prior Only` shows whether semantic cues still help once prior guidance is introduced.
- Focus the paper on weak/medium intensity samples and boundary emotions such as `concern`, `calm`, and `surprise` when discussing the innovation point.
