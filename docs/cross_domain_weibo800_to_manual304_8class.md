# Cross-Domain Summary: Weibo 8-Class Train -> Manual 8-Class Test

- train_data: `data/weibo_expression_samples_8class_800.json`
- test_data: `data/manual_samples_8class_304.json`
- train_domain: weak-labeled real weibo text, 8 emotions, 800 samples
- test_domain: manual 8-class dataset, 304 samples

| Run | Setting | Test MAE | Test RMSE | Active MAE | Test Loss |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline MLP | text embedding + emotion + intensity | 0.022876 | 0.057425 | 0.095499 | 0.003298 |
| Semantic MLP | baseline + semantic cue features | 0.021603 | 0.053932 | 0.088888 | 0.002909 |
| Prior Only | expression prior + residual refinement | 0.007138 | 0.018870 | 0.031355 | 0.000356 |
| Prior-Fusion | semantic features + expression prior + residual refinement | 0.006719 | 0.018040 | 0.030069 | 0.000325 |

## Notes

- This is the 8-class cross-domain setting: train on mined weibo samples, test on manual balanced set.
- The prior-guided methods remain substantially better than pure MLP baselines under domain shift.
- `Semantic MLP` improves over `Baseline MLP`, while `Prior-Fusion` remains competitive with `Prior Only`.
