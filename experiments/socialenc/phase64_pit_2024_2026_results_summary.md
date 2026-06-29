# Phase64 PIT 2024-2026 Results Summary

This file consolidates the current point-in-time experiment results for the
recent two-year final test. It supersedes the earlier frozen 2020-2021 to
2022-2026 diagnostic tables for paper-facing reporting.

## Protocol

| Role | Period | Use |
|---|---|---|
| Train | 2022-06-01 to 2023-06-01 | fit ridge/readout parameters |
| Validation | 2023-06-01 to 2024-06-01 | select ridge alpha and reporting settings |
| Final test | 2024-06-01 to 2026-06-01 | final held-out evaluation |

Point-in-time history:

| Evaluation block | O_k / history cutoff |
|---|---|
| Train rows | before 2022-06-01 |
| Validation rows | before 2023-06-01 |
| Test 2024-2025 | before 2024-06-01 |
| Test 2025-2026 | before 2025-06-01 |

Panel size:

```text
origin candidates: 59,802
train rows:        14,559
validation rows:   15,128
test rows:         30,115
test events:        4,055
symbols:               17
```

## Main Experiment Combined Table

Sources:

- `phase61_pit_lightweight_2024_2026_table.md`
- `phase63_pit_surface_2024_2026_table.md`
- `phase62_pit_qwen3_4b_2024_2026_table.md`

| Family | Method | Events | Symbols | NDCG@3 | Hit@1 | Mass@3 | JS ↓ | ΔNDCG vs No-OL | ΔHit vs No-OL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Scale | Follower | 4055 | 17 | 0.516 | 0.273 | 0.619 | 0.354 | -0.148 | -0.101 |
| Scale | Visibility | 4055 | 17 | 0.519 | 0.275 | 0.624 | 0.354 | -0.145 | -0.098 |
| Context | Rank/Time | 4055 | 17 | 0.571 | 0.248 | 0.713 | 0.343 | -0.093 | -0.125 |
| Context | Sentiment | 4055 | 17 | 0.567 | 0.290 | 0.669 | 0.360 | -0.097 | -0.083 |
| Context | Novelty | 4055 | 17 | 0.603 | 0.283 | 0.736 | 0.335 | -0.061 | -0.091 |
| Context | History | 4055 | 17 | 0.616 | 0.344 | 0.724 | 0.334 | -0.048 | -0.030 |
| Context | No-OL Strong | 4055 | 17 | 0.664 | 0.373 | 0.773 | 0.313 | +0.000 | +0.000 |
| Origin Role | OL Only | 4055 | 17 | 0.475 | 0.193 | 0.606 | 0.360 | -0.189 | -0.180 |
| Origin Role | OL-Origin | 4055 | 17 | 0.671 | 0.379 | 0.776 | 0.308 | +0.007 | +0.006 |
| Surface | Symbol one-hot | 4055 | 17 | 0.587 | 0.283 | 0.714 | 0.360 | -0.077 | -0.090 |
| Surface | Text surface | 4055 | 17 | 0.576 | 0.328 | 0.677 | 0.344 | -0.088 | -0.045 |
| Surface | Symbol + surface | 4055 | 17 | 0.577 | 0.327 | 0.677 | 0.342 | -0.087 | -0.046 |
| Text Encoder | **Qwen3-4B-origin text** | 4055 | 17 | **0.688** | **0.421** | **0.781** | **0.290** | +0.024 | +0.048 |

Main-table bootstrap support:

| Comparison | ΔNDCG@3 90% CI | ΔHit@1 90% CI | JS improvement 90% CI |
|---|---:|---:|---:|
| OL-Origin vs Follower | +0.155 [+0.106, +0.214] | +0.106 [+0.056, +0.155] | +0.047 [+0.029, +0.064] |
| OL-Origin vs Visibility | +0.151 [+0.097, +0.207] | +0.104 [+0.055, +0.154] | +0.046 [+0.028, +0.065] |
| OL-Origin vs Rank/Time | +0.099 [+0.076, +0.122] | +0.131 [+0.097, +0.171] | +0.035 [+0.022, +0.047] |
| OL-Origin vs History | +0.054 [+0.041, +0.068] | +0.036 [+0.018, +0.057] | +0.026 [+0.020, +0.032] |
| OL-Origin vs No-OL Strong | +0.007 [+0.002, +0.011] | +0.006 [-0.004, +0.017] | +0.006 [+0.004, +0.007] |
| OL-Origin vs Symbol one-hot | +0.083 [+0.059, +0.106] | +0.096 [+0.063, +0.136] | +0.053 [+0.036, +0.070] |
| OL-Origin vs Text surface | +0.094 [+0.039, +0.166] | +0.051 [-0.008, +0.110] | +0.036 [+0.008, +0.067] |
| OL-Origin vs Symbol + surface | +0.094 [+0.042, +0.151] | +0.052 [-0.006, +0.106] | +0.034 [+0.007, +0.060] |
| Qwen3-4B vs No-OL Strong | +0.024 [-0.005, +0.052] | +0.048 [-0.001, +0.087] | +0.023 [+0.007, +0.040] |
| Qwen3-4B vs OL-Origin | +0.018 [-0.008, +0.043] | +0.042 [-0.000, +0.085] | +0.017 [-0.001, +0.036] |

## Ablation Rows

Source: `phase61_pit_lightweight_2024_2026_table.md`

| Method | Events | Symbols | NDCG@3 | Hit@1 | Mass@3 | JS ↓ | ΔNDCG vs No-OL | ΔHit vs No-OL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OL only | 4055 | 17 | 0.475 | 0.193 | 0.606 | 0.360 | -0.189 | -0.180 |
| No-OL strong | 4055 | 17 | 0.664 | 0.373 | 0.773 | 0.313 | +0.000 | +0.000 |
| Shuffled OL | 4055 | 17 | 0.659 | 0.380 | 0.765 | 0.313 | -0.005 | +0.007 |
| Follower replacement | 4055 | 17 | 0.664 | 0.373 | 0.774 | 0.313 | +0.000 | +0.000 |
| Raw OL | 4055 | 17 | 0.664 | **0.382** | 0.771 | 0.309 | +0.000 | +0.008 |
| **Full OL-Origin** | 4055 | 17 | **0.671** | 0.379 | **0.776** | **0.308** | +0.007 | +0.006 |

Key bootstrap support:

| Comparison | ΔNDCG@3 90% CI | ΔHit@1 90% CI | JS improvement 90% CI |
|---|---:|---:|---:|
| Full OL-Origin vs No-OL strong | +0.007 [+0.002, +0.012] | +0.006 [-0.003, +0.016] | +0.006 [+0.004, +0.007] |
| Full OL-Origin vs OL only | +0.196 [+0.159, +0.234] | +0.186 [+0.143, +0.229] | +0.052 [+0.036, +0.069] |
| Full OL-Origin vs Shuffled OL | +0.012 [+0.006, +0.017] | -0.001 [-0.014, +0.009] | +0.005 [+0.003, +0.007] |
| Full OL-Origin vs Follower replacement | +0.006 [+0.001, +0.011] | +0.006 [-0.004, +0.016] | +0.005 [+0.004, +0.007] |
| Full OL-Origin vs Raw OL | +0.006 [-0.004, +0.017] | -0.002 [-0.022, +0.014] | +0.001 [-0.004, +0.007] |

## Current Takeaway

Under the latest PIT 2024-2026 final-test setting:

1. **Qwen3-4B-origin text is the strongest completed main-experiment row** on
   NDCG@3, Hit@1, Mass@3, and JS.
2. **OL-Origin is the strongest non-embedding lightweight structured row**.
3. The OL-Origin gain over No-OL Strong is small but directionally stable:
   `+0.007` NDCG@3 and `+0.006` JS improvement, both with positive 90% CI for
   NDCG and JS.
4. Qwen3-4B beats OL-Origin by `+0.018` NDCG@3, `+0.042` Hit@1, and `+0.017`
   JS improvement, but the 90% CIs overlap zero for Qwen vs OL-Origin.
5. Surface-only diagnostics are clearly weaker than OL-Origin and No-OL Strong.
6. Ablation supports the mechanism most clearly through NDCG@3 and JS:
   Full OL-Origin beats OL-only, shuffled OL, follower replacement, and No-OL
   Strong on these metrics.
7. Raw OL remains close to residualized OL in this recent PIT setting, so the
   residualization result should be described carefully: it is theoretically
   cleaner and reduces timezone confounding, but the recent-test empirical edge
   over raw OL is not large.
