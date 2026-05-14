# When Words Smile Prior V6 Training Report

## Method

V6 trains a mixture gate between the strongest global residual correction output and the learned-prior output. The gate learns when to trust the learned affect-prior branch.

```text
P_v6 = P_global + gamma * (P_prior - P_global)
```

The gate is trained only on the dev split and then evaluated on the test split.

## Settings

- Gate mode: `sample`
- Epochs: `120`
- Batch size: `96`
- Learning rate: `0.002`
- Hidden size: `64`
- Dynamic loss weight: `0.05`
- Gate regularization weight: `0.0002`
- Best val MAE: `0.358716`
- Output: `outputs\when_words_smile_prior_v6\test_mixture_gate_sample.pt`
- Checkpoint: `outputs\when_words_smile_prior_v6\mixture_gate_v6_sample.pt`

## Test Gate Statistics

- Mean gate: `0.289096`
- Median gate: `0.278742`
- Min gate: `0.249150`
- Max gate: `0.471743`

## History

| Epoch | Train Loss | Val MAE |
|---:|---:|---:|
| 0 | 0.370668 | 0.358716 |
| 10 | 0.371428 | 0.358804 |
| 20 | 0.369984 | 0.358829 |
| 30 | 0.370324 | 0.358821 |
| 40 | 0.370328 | 0.358847 |
| 50 | 0.370199 | 0.358775 |
| 60 | 0.371184 | 0.358921 |
| 70 | 0.370726 | 0.358900 |
| 80 | 0.369849 | 0.358815 |
| 90 | 0.369757 | 0.358818 |
| 100 | 0.370042 | 0.358875 |
| 110 | 0.370164 | 0.358893 |
| 119 | 0.370699 | 0.358888 |