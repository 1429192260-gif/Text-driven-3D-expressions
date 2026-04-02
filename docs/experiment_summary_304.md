# Experiment Summary

- epochs: 60
- batch_size: 8
- seed: 42

| Run | Setting | Test MAE | Test RMSE | Active MAE | Test Loss |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline MLP | text embedding + emotion + intensity | 0.028781 | 0.071406 | 0.111220 | 0.005099 |
| Semantic MLP | baseline + semantic cue features | 0.027164 | 0.067392 | 0.107105 | 0.004542 |
| Prior Only | expression prior + residual refinement without semantic cues | 0.004215 | 0.011278 | 0.018629 | 0.000127 |
| Prior-Fusion | semantic features + expression prior + residual refinement | 0.004182 | 0.011400 | 0.018566 | 0.000130 |

## Per-Emotion Test MAE

| Group | Baseline MLP | Semantic MLP | Prior Only | Prior-Fusion |
| --- | ---: | ---: | ---: | ---: |
| angry | 0.039290 | 0.032449 | 0.005270 | 0.005507 |
| bored | 0.016826 | 0.016970 | 0.002894 | 0.002327 |
| calm | 0.006424 | 0.004499 | 0.001042 | 0.001222 |
| concern | 0.027047 | 0.028355 | 0.004170 | 0.004072 |
| disgust | 0.039903 | 0.038757 | 0.006257 | 0.005696 |
| happy | 0.027878 | 0.026584 | 0.004377 | 0.004533 |
| sad | 0.021629 | 0.019587 | 0.004157 | 0.004641 |
| surprise | 0.051254 | 0.050111 | 0.005554 | 0.005461 |

## Per-Intensity Test MAE

| Group | Baseline MLP | Semantic MLP | Prior Only | Prior-Fusion |
| --- | ---: | ---: | ---: | ---: |
| medium | 0.026517 | 0.025515 | 0.004154 | 0.004060 |
| strong | 0.030678 | 0.030287 | 0.006132 | 0.006741 |
| weak | 0.038147 | 0.032532 | 0.002911 | 0.002644 |

## Writing Notes

- `Semantic MLP` vs `Baseline MLP` isolates the value of handcrafted semantic cue features.
- `Prior Only` vs `Baseline MLP` isolates the effect of expression priors without semantic enhancement.
- `Prior-Fusion` vs `Prior Only` shows whether semantic cues still help once prior guidance is introduced.
- Focus the paper on weak/medium intensity samples and boundary emotions such as `concern`, `calm`, and `surprise` when discussing the innovation point.
