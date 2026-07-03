# Text-Only Front-End Ablation Summary

- evaluation setting: manual-only test set under `source_holdout`
- backend expression model: `outputs/prior_fusion_full_mapper.pt`
- upper bound MAE: 0.004559
- upper bound RMSE: 0.012615
- upper bound active MAE: 0.020124

| Front-End | Setting | Text-Only MAE | RMSE | Active MAE |
| --- | --- | ---: | ---: | ---: |
| Learned Front-End (8-class classifier) | learned emotion classifier + rule intensity | 0.085774 | 0.180892 | 0.190206 |
| Chinese-Emotion-Small (initial) | local HF model + minimal label/intensity mapping | 0.067002 | 0.150095 | 0.176635 |
| Chinese-Emotion-Small + Rules (v2) | HF model + keyword correction rules | 0.034238 | 0.098984 | 0.136327 |
| Chinese-Emotion-Small + Rules (v3) | HF model + intensity calibration | 0.026412 | 0.078465 | 0.098631 |
| Chinese-Emotion-Small + Rules (v4) | HF model + boundary/intensity refinements | 0.018212 | 0.057428 | 0.075437 |

## Key Findings

- Replacing the learned front-end with `Chinese-Emotion-Small` gives a clear improvement.
- Adding Chinese task-specific rules further improves both overall MAE and active-parameter MAE.
- The best current text-only front-end is `Chinese-Emotion-Small + Rules (v4)`.
- The main gain path is: stronger emotion recognition first, then intensity calibration and emotion-boundary refinement.
