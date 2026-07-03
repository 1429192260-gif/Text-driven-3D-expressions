# Cross-Domain Summary: Weibo Train -> Manual Test

- train_data: `data/weibo_expression_samples_500.json`
- test_data: `data/manual_samples_5class_190.json`
- train_domain: weak-labeled real weibo text
- test_domain: manual 5-class subset from `full_samples_804.json`

| Run | Setting | Test MAE | Test RMSE | Active MAE | Test Loss |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline MLP | text embedding + emotion + intensity | 0.024909 | 0.064123 | 0.120104 | 0.004070 |
| Semantic MLP | baseline + semantic cue features | 0.025172 | 0.063969 | 0.120942 | 0.004050 |
| Prior Only | expression prior + residual refinement | 0.005401 | 0.015737 | 0.027724 | 0.000245 |
| Prior-Fusion | semantic features + expression prior + residual refinement | 0.005644 | 0.016086 | 0.028046 | 0.000256 |

## Notes

- This evaluation is more realistic than random weibo-only split because training and testing come from different domains.
- The five shared emotions are: `happy`, `sad`, `angry`, `surprise`, `calm`.
- Prior-guided models remain clearly better than pure MLP under cross-domain transfer.
