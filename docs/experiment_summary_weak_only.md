# Experiment Summary

- epochs: 80
- batch_size: 8
- seed: 42

| Run | Setting | Test MAE | Test RMSE | Active MAE | Test Loss |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline MLP | text embedding + emotion + intensity | 0.039733 | 0.056648 | 0.091679 | 0.003209 |
| Semantic MLP | baseline + semantic cue features | 0.037874 | 0.059055 | 0.108714 | 0.003487 |
| Prior Only | expression prior + residual refinement without semantic cues | 0.001352 | 0.003816 | 0.006572 | 0.000015 |
| Prior-Fusion | semantic features + expression prior + residual refinement | 0.001601 | 0.004360 | 0.007808 | 0.000019 |

## Per-Emotion Test MAE

| Group | Baseline MLP | Semantic MLP | Prior Only | Prior-Fusion |
| --- | ---: | ---: | ---: | ---: |
| bored | 0.030245 | 0.030866 | 0.001400 | 0.002256 |
| calm | 0.018849 | 0.014535 | 0.000528 | 0.000346 |
| concern | 0.045908 | 0.043489 | 0.001321 | 0.001567 |
| happy | 0.054386 | 0.045319 | 0.002663 | 0.003110 |
| sad | 0.043300 | 0.042128 | 0.001230 | 0.001674 |
| surprise | 0.045712 | 0.050909 | 0.000968 | 0.000652 |

## Per-Intensity Test MAE

| Group | Baseline MLP | Semantic MLP | Prior Only | Prior-Fusion |
| --- | ---: | ---: | ---: | ---: |
| medium | 0.038442 | 0.037765 | 0.001012 | 0.001060 |
| weak | 0.042316 | 0.038092 | 0.002031 | 0.002683 |

## Writing Notes

- `Semantic MLP` vs `Baseline MLP` isolates the value of handcrafted semantic cue features.
- `Prior Only` vs `Baseline MLP` isolates the effect of expression priors without semantic enhancement.
- `Prior-Fusion` vs `Prior Only` shows whether semantic cues still help once prior guidance is introduced.
- Focus the paper on weak/medium intensity samples and boundary emotions such as `concern`, `calm`, and `surprise` when discussing the innovation point.
