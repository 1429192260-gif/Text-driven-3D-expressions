# Experiment Summary

- epochs: 60
- batch_size: 8
- seed: 42

| Run | Setting | Test MAE | Test RMSE | Active MAE | Test Loss |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline MLP | text embedding + emotion + intensity | 0.035630 | 0.088019 | 0.143153 | 0.007747 |
| Semantic MLP | baseline + semantic cue features | 0.035277 | 0.088445 | 0.142557 | 0.007823 |
| Prior-Fusion | semantic features + expression prior + residual refinement | 0.004881 | 0.012749 | 0.020648 | 0.000163 |

## Writing Notes

- `Baseline MLP` can serve as the core baseline without semantic enhancement or explicit prior guidance.
- `Semantic MLP` is the first ablation to verify whether handcrafted semantic cues improve parameter prediction.
- `Prior-Fusion` is the main method and tests whether expression priors plus residual correction stabilize learning in small-sample settings.
- If `Prior-Fusion` is much better than the other models, the paper should state clearly that the current labels are strongly aligned with rule-based priors.
