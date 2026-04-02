# Experiment Summary

- epochs: 60
- batch_size: 8
- seed: 42

| Run | Setting | Test MAE | Test RMSE | Active MAE | Test Loss |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline MLP | text embedding + emotion + intensity | 0.022371 | 0.058152 | 0.089541 | 0.003206 |
| Semantic MLP | baseline + semantic cue features | 0.022879 | 0.058589 | 0.090810 | 0.003178 |
| Prior Only | expression prior + residual refinement without semantic cues | 0.004618 | 0.012288 | 0.020058 | 0.000148 |
| Prior-Fusion | semantic features + expression prior + residual refinement | 0.004476 | 0.012292 | 0.019439 | 0.000143 |

## Per-Emotion Test MAE

| Group | Baseline MLP | Semantic MLP | Prior Only | Prior-Fusion |
| --- | ---: | ---: | ---: | ---: |
| angry | 0.029820 | 0.029196 | 0.004704 | 0.004361 |
| bored | 0.019219 | 0.019006 | 0.003229 | 0.002898 |
| calm | 0.006513 | 0.005176 | 0.001244 | 0.000694 |
| concern | 0.038111 | 0.040822 | 0.005511 | 0.005305 |
| disgust | 0.031644 | 0.028146 | 0.006843 | 0.008122 |
| happy | 0.015301 | 0.017958 | 0.004605 | 0.004763 |
| sad | 0.016522 | 0.016938 | 0.004899 | 0.004692 |
| surprise | 0.028407 | 0.031241 | 0.005480 | 0.004510 |

## Per-Intensity Test MAE

| Group | Baseline MLP | Semantic MLP | Prior Only | Prior-Fusion |
| --- | ---: | ---: | ---: | ---: |
| medium | 0.020204 | 0.020617 | 0.004261 | 0.003940 |
| strong | 0.026343 | 0.025111 | 0.005523 | 0.004847 |
| weak | 0.024029 | 0.026682 | 0.004618 | 0.005550 |

## Writing Notes

- `Semantic MLP` vs `Baseline MLP` isolates the value of handcrafted semantic cue features.
- `Prior Only` vs `Baseline MLP` isolates the effect of expression priors without semantic enhancement.
- `Prior-Fusion` vs `Prior Only` shows whether semantic cues still help once prior guidance is introduced.
- Focus the paper on weak/medium intensity samples and boundary emotions such as `concern`, `calm`, and `surprise` when discussing the innovation point.
