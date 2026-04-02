# Experiment Summary

- epochs: 60
- batch_size: 8
- seed: 42

| Run | Setting | Test MAE | Test RMSE | Active MAE | Test Loss |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline MLP | text embedding + emotion + intensity | 0.015648 | 0.042022 | 0.067150 | 0.001732 |
| Semantic MLP | baseline + semantic cue features | 0.015197 | 0.039368 | 0.063664 | 0.001579 |
| Prior Only | expression prior + residual refinement without semantic cues | 0.003813 | 0.010813 | 0.017590 | 0.000115 |
| Prior-Fusion | semantic features + expression prior + residual refinement | 0.003941 | 0.010990 | 0.018206 | 0.000120 |

## Per-Emotion Test MAE

| Group | Baseline MLP | Semantic MLP | Prior Only | Prior-Fusion |
| --- | ---: | ---: | ---: | ---: |
| angry | 0.016339 | 0.015377 | 0.004406 | 0.004172 |
| bored | 0.014144 | 0.014557 | 0.004261 | 0.004856 |
| calm | 0.004048 | 0.004729 | 0.001565 | 0.001487 |
| concern | 0.024501 | 0.027310 | 0.002723 | 0.002800 |
| disgust | 0.022592 | 0.015173 | 0.006364 | 0.008038 |
| happy | 0.014311 | 0.014109 | 0.004365 | 0.004484 |
| sad | 0.011957 | 0.012680 | 0.003912 | 0.004162 |
| surprise | 0.028014 | 0.026228 | 0.004343 | 0.004433 |

## Per-Intensity Test MAE

| Group | Baseline MLP | Semantic MLP | Prior Only | Prior-Fusion |
| --- | ---: | ---: | ---: | ---: |
| medium | 0.013688 | 0.012969 | 0.003503 | 0.003604 |
| strong | 0.020719 | 0.021687 | 0.005017 | 0.005431 |
| weak | 0.024592 | 0.024369 | 0.004683 | 0.004635 |

## Writing Notes

- `Semantic MLP` vs `Baseline MLP` isolates the value of handcrafted semantic cue features.
- `Prior Only` vs `Baseline MLP` isolates the effect of expression priors without semantic enhancement.
- `Prior-Fusion` vs `Prior Only` shows whether semantic cues still help once prior guidance is introduced.
- Focus the paper on weak/medium intensity samples and boundary emotions such as `concern`, `calm`, and `surprise` when discussing the innovation point.
