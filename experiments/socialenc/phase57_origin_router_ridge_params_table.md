# Phase57 Main OL-Origin Ridge Parameters

Main-table setting: threshold 0.55, origin window first10, target `log_future_reach`, ridge alpha 3.0.

Validation metrics reproduced: NDCG@3 0.7551, Hit@1 0.4934, Mass@3 0.9012, JS 0.2626.

The fitted model uses standardized features:

```text
score = intercept + sum_j beta_j * standardized(feature_j)
```

Standardized intercept: `2.1691`

| Feature | Standardized coef | Raw-space coef | Train median | Train mean | Train sd |
|---|---:|---:|---:|---:|---:|
| `origin_ol` | -5.0147 | -2.6655 | 0.2772 | 0.5018 | 1.8814 |
| `origin_logfoll` | +0.4057 | +0.4204 | 12.8926 | 12.8281 | 0.9651 |
| `origin_verified` | -0.0286 | -0.1225 | 1.0000 | 0.9422 | 0.2333 |
| `log_origin_rank` | -0.0477 | -0.0681 | 1.6094 | 1.4108 | 0.7003 |
| `elapsed_hours` | -0.3146 | -0.0508 | 9.8636 | 9.1037 | 6.1877 |
| `prior_frame_count` | -0.5733 | -0.2354 | 3.0000 | 3.6117 | 2.4358 |
| `origin_stance` | -0.0754 | -0.1394 | 0.0000 | 0.0772 | 0.5408 |
| `origin_stance_abs` | +0.0868 | +0.1897 | 0.0000 | 0.2984 | 0.4576 |
| `novelty_global` | -0.8810 | -6.7827 | 0.3073 | 0.2967 | 0.1299 |
| `novelty_event` | -0.3483 | -3.9276 | 0.5732 | 0.5868 | 0.0887 |
| `hist_log_origin_count` | -0.3633 | -0.3680 | 2.5649 | 2.3897 | 0.9873 |
| `hist_mean_log_adopt` | -0.1105 | -0.4674 | 0.0885 | 0.1547 | 0.2365 |
| `hist_success_rate` | +0.4637 | +2.3028 | 0.1111 | 0.1568 | 0.2014 |
| `ol_x_visibility` | +4.9235 | +0.2015 | 3.5196 | 6.3368 | 24.4332 |
| `ol_x_novelty` | +0.3500 | +0.6074 | 0.0568 | 0.0969 | 0.5762 |

Raw-space expression for the OL channel, holding other variables fixed:

```text
d score / d origin_ol ~= -2.6655 + 0.2015 * origin_logfoll + 0.6074 * novelty_global
```

This is why `origin_ol` should not be read as a standalone KOL ranking. The positive effect is context-dependent and mainly appears through `O_k × visibility` and `O_k × novelty` interactions.
