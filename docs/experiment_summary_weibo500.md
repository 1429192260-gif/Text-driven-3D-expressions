# Experiment Summary

- epochs: 60
- batch_size: 8
- seed: 42
- split_mode: random

| Run | Setting | Test MAE | Test RMSE | Active MAE | Test Loss |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline MLP | text embedding + emotion + intensity | 0.010654 | 0.027340 | 0.047040 | 0.000900 |
| Semantic MLP | baseline + semantic cue features | 0.010794 | 0.027207 | 0.045822 | 0.000887 |
| Prior Only | expression prior + residual refinement without semantic cues | 0.003281 | 0.008988 | 0.015535 | 0.000085 |
| Prior-Fusion | semantic features + expression prior + residual refinement | 0.003142 | 0.008548 | 0.015424 | 0.000079 |

## Split Diagnostics

- split_mode: random
- train_manual: 0
- train_augmented: 350
- val_manual: 0
- val_augmented: 75
- test_manual: 0
- test_augmented: 75

## Per-Emotion Test MAE

| Group | Baseline MLP | Semantic MLP | Prior Only | Prior-Fusion |
| --- | ---: | ---: | ---: | ---: |
| angry | 0.011807 | 0.011635 | 0.003725 | 0.003531 |
| calm | 0.003822 | 0.004195 | 0.001578 | 0.001149 |
| happy | 0.009259 | 0.011948 | 0.003797 | 0.003729 |
| sad | 0.013231 | 0.011571 | 0.003473 | 0.003376 |
| surprise | 0.015153 | 0.014622 | 0.003831 | 0.003925 |

## Per-Intensity Test MAE

| Group | Baseline MLP | Semantic MLP | Prior Only | Prior-Fusion |
| --- | ---: | ---: | ---: | ---: |
| medium | 0.009006 | 0.009142 | 0.003209 | 0.003043 |
| strong | 0.018516 | 0.018675 | 0.003622 | 0.003612 |

## Per-Source Test MAE

| Group | Baseline MLP | Semantic MLP | Prior Only | Prior-Fusion |
| --- | ---: | ---: | ---: | ---: |
| weibo_mined_large | 0.010654 | 0.010794 | 0.003281 | 0.003142 |

## Writing Notes

- `Semantic MLP` vs `Baseline MLP` isolates the value of handcrafted semantic cue features.
- `Prior Only` vs `Baseline MLP` isolates the effect of expression priors without semantic enhancement.
- `Prior-Fusion` vs `Prior Only` shows whether semantic cues still help once prior guidance is introduced.
- When `split_mode=source_holdout`, the test set is intended to be dominated by manual samples, which is more conservative than random mixing.
