# Experiment Summary

- epochs: 60
- batch_size: 8
- seed: 42
- split_mode: source_holdout

| Run | Setting | Test MAE | Test RMSE | Active MAE | Test Loss |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline MLP | text embedding + emotion + intensity | 0.016927 | 0.044947 | 0.066844 | 0.002020 |
| Semantic MLP | baseline + semantic cue features | 0.016244 | 0.043251 | 0.065581 | 0.001871 |
| Prior Only | expression prior + residual refinement without semantic cues | 0.004722 | 0.012838 | 0.020666 | 0.000165 |
| Prior-Fusion | semantic features + expression prior + residual refinement | 0.004559 | 0.012615 | 0.020124 | 0.000159 |

## Split Diagnostics

- split_mode: source_holdout
- train_manual: 216
- train_augmented: 410
- val_manual: 48
- val_augmented: 90
- test_manual: 40
- test_augmented: 0

## Per-Emotion Test MAE

| Group | Baseline MLP | Semantic MLP | Prior Only | Prior-Fusion |
| --- | ---: | ---: | ---: | ---: |
| angry | 0.020799 | 0.027600 | 0.006708 | 0.006807 |
| bored | 0.009923 | 0.009707 | 0.004657 | 0.004578 |
| calm | 0.002917 | 0.002890 | 0.001735 | 0.001406 |
| concern | 0.030714 | 0.022569 | 0.003491 | 0.002785 |
| disgust | 0.017039 | 0.017906 | 0.005182 | 0.005497 |
| happy | 0.011683 | 0.011993 | 0.006741 | 0.006652 |
| sad | 0.013826 | 0.014470 | 0.006156 | 0.005836 |
| surprise | 0.028516 | 0.022820 | 0.003109 | 0.002911 |

## Per-Intensity Test MAE

| Group | Baseline MLP | Semantic MLP | Prior Only | Prior-Fusion |
| --- | ---: | ---: | ---: | ---: |
| medium | 0.014921 | 0.014371 | 0.004528 | 0.004205 |
| strong | 0.023749 | 0.021827 | 0.005672 | 0.005525 |
| weak | 0.017226 | 0.017214 | 0.004500 | 0.004820 |

## Per-Source Test MAE

| Group | Baseline MLP | Semantic MLP | Prior Only | Prior-Fusion |
| --- | ---: | ---: | ---: | ---: |
| manual | 0.016927 | 0.016244 | 0.004722 | 0.004559 |

## Writing Notes

- `Semantic MLP` vs `Baseline MLP` isolates the value of handcrafted semantic cue features.
- `Prior Only` vs `Baseline MLP` isolates the effect of expression priors without semantic enhancement.
- `Prior-Fusion` vs `Prior Only` shows whether semantic cues still help once prior guidance is introduced.
- When `split_mode=source_holdout`, the test set is intended to be dominated by manual samples, which is more conservative than random mixing.
