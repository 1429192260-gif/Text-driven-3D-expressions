# When Words Smile Prior V5 Experiment Report

## 1. Motivation

V4 showed that a hand-crafted uncertainty-aware prior still fails to improve over the global residual correction branch. V5 tests whether a learned text-to-affect prior can provide a stronger prior signal.

Instead of converting keyword evidence into a fixed probability distribution, V5 learns:

```text
text features -> latent affect distribution + confidence + intensity
```

The text features include:

- keyword score features,
- text length and punctuation statistics,
- optional frozen BERT sentence embeddings.

The learned prior is then injected into the same two-branch residual fusion structure:

```text
p'_t = p_t + g_global(t) * r_global(t) + c_text * g_prior(t) * r_prior(t)
```

## 2. Main Results

| Method | MAE | RMSE | Smoothness | Acceleration | PPL |
|---|---:|---:|---:|---:|---:|
| When Words Smile baseline | 0.379475 | 0.563167 | 0.386786 | 0.663121 | 262.39 |
| V3 best: no emotion | 0.358055 | 0.522169 | 0.280578 | 0.479446 | 143.99 |
| V4 global-only | 0.356114 | 0.519775 | 0.289948 | 0.495746 | 148.31 |
| V4 prior-only | 0.367962 | 0.544393 | 0.359195 | 0.616008 | 207.23 |
| V5 full learned prior | 0.358160 | 0.522113 | 0.320896 | 0.550066 | 146.84 |
| V5 global-only | 0.356194 | 0.519915 | 0.304176 | 0.520397 | 147.51 |
| V5 learned prior-only | 0.358501 | 0.522493 | 0.331309 | 0.568695 | 151.53 |
| V5 + frozen BERT | 0.359538 | 0.524921 | 0.325744 | 0.558203 | 168.32 |

## 3. Key Findings

The learned prior is clearly stronger than the rule-based prior:

```text
V4 prior-only MAE: 0.367962
V5 prior-only MAE: 0.358501
```

This means that learning the affect prior from data is much better than relying on fixed keyword confidence rules.

However, V5 full still does not beat V4 global-only on MAE:

```text
V4 global-only MAE: 0.356114
V5 full MAE: 0.358160
```

V5 full does slightly improve PPL over V4 global-only:

```text
V4 global-only PPL: 148.31
V5 full PPL: 146.84
```

This suggests that the learned prior may improve distribution-level likelihood but still introduces frame-wise error or smoothness instability.

The frozen BERT feature variant performs worse. This is consistent with V3: direct concatenation of high-dimensional frozen BERT sentence embeddings tends to overfit on the small dev training split.

## 4. Interpretation

V5 gives a more nuanced conclusion than V4.

The earlier rule-based affect prior was too weak. The learned prior branch is much stronger and nearly reaches the global correction branch. This supports the general idea that text-derived affect prior can be useful.

But the current fusion strategy is still not ideal. When the learned prior branch is combined with the global branch, it does not surpass the global branch on MAE. The prior branch likely needs better supervision, better gating, or more data/context to become consistently beneficial.

## 5. Current Best Method

The best MAE/RMSE method remains:

```text
V4 global-only / V5 global-only residual correction
```

The best PPL among the latest experiments is:

```text
V5 full learned prior
```

So the current conclusion is:

- For frame-wise accuracy, use global residual correction.
- For likelihood-style metric, learned affect prior begins to help.
- Learned prior is much better than rule prior, but not yet strong enough to replace or improve the global correction backbone.

## 6. Research Value

V5 improves the research story:

1. Rule-based prior fails.
2. Learned prior is much better than rule prior.
3. Residual correction remains the strongest backbone.
4. The remaining challenge is how to fuse learned affect prior without hurting frame-wise accuracy.

This is a stronger and more honest innovation path than simply claiming that hand-crafted emotion rules work.

## 7. Recommended Next Step

The next useful experiment should focus on fusion, not on more text features.

Recommended V6 options:

- Use V4 global-only as a frozen base and train only a small learned-prior residual on top.
- Add a constraint so the learned prior branch only activates on samples where it reduces validation error.
- Train a mixture-of-experts gate between global-only and learned-prior outputs.
- Add dialogue context for short utterances before feeding the learned prior.

Among these, the easiest and most promising is:

```text
V6: mixture gate between V4 global-only and V5 learned-prior output
```

This would directly test whether the model can choose when to trust the learned affect prior.
