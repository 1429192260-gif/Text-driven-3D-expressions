# Strict Text-Only Front-End Ablation Summary

- evaluation setting: `source_holdout`, manual-only test set
- fixed backend checkpoint: `outputs/prior_fusion_804_source_holdout_strict_mapper.pt`
- upper bound MAE: 0.004520
- upper bound RMSE: 0.012210
- upper bound active MAE: 0.019938

| Front-End | Setting | Text-Only MAE | RMSE | Active MAE |
| --- | --- | ---: | ---: | ---: |
| Learned Front-End (8-class) | self-trained 8-class classifier + simple intensity rules | 0.085059 | 0.179165 | 0.189854 |
| Chinese-Emotion-Small (raw) | original HF output + minimal label mapping, no task-specific correction | 0.065244 | 0.140318 | 0.192397 |
| Chinese-Emotion-Small + Rules (v4) | HF output + Chinese fine-grained correction rules + intensity calibration | 0.017869 | 0.056679 | 0.074145 |

## Key Findings

- `Chinese-Emotion-Small (raw)` is better than the learned front-end, which shows the value of external Chinese emotion knowledge.
- `Chinese-Emotion-Small + Rules (v4)` is dramatically better than the raw version, which shows the gain does not come from model replacement alone.
- The main improvement comes from task-specific Chinese emotion-boundary correction and intensity calibration.
