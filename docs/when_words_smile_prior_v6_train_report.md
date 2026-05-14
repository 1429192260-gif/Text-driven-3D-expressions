# When Words Smile Prior V6 Training Report

## Method

V6 trains a mixture gate between the strongest global residual correction output and the learned-prior output. The gate learns when to trust the learned affect-prior branch.

```text
P_v6 = P_global + gamma * (P_prior - P_global)
```

The gate is trained only on the dev split and then evaluated on the test split.

## Settings

- Gate mode: `frame`
- Epochs: `120`
- Batch size: `96`
- Learning rate: `0.002`
- Hidden size: `64`
- Dynamic loss weight: `0.05`
- Gate regularization weight: `0.0002`
- Best val MAE: `0.358793`
- Output: `outputs\when_words_smile_prior_v6\test_mixture_gate.pt`
- Checkpoint: `outputs\when_words_smile_prior_v6\mixture_gate_v6.pt`

## Test Gate Statistics

- Mean gate: `0.232662`
- Median gate: `0.220512`
- Min gate: `0.146457`
- Max gate: `0.707542`

## History

| Epoch | Train Loss | Val MAE |
|---:|---:|---:|
| 0 | 0.370638 | 0.358793 |
| 10 | 0.371210 | 0.359008 |
| 20 | 0.369585 | 0.359296 |
| 30 | 0.369591 | 0.359591 |
| 40 | 0.369328 | 0.359746 |
| 50 | 0.368955 | 0.359757 |
| 60 | 0.369679 | 0.359953 |
| 70 | 0.369072 | 0.359861 |
| 80 | 0.368055 | 0.359887 |
| 90 | 0.367830 | 0.359898 |
| 100 | 0.368018 | 0.360038 |
| 110 | 0.368019 | 0.360020 |
| 119 | 0.368514 | 0.359985 |