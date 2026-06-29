# Phase61 PIT Lightweight 2024-2026 Test

Protocol: train 2022-06-01 to 2023-06-01; validation 2023-06-01 to 2024-06-01; final test 2024-06-01 to 2026-06-01. O_k/history are estimated only from data before each block.

## Main Lightweight Rows

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
| Origin Role | **OL-Origin** | 4055 | 17 | 0.671 | 0.379 | 0.776 | 0.308 | +0.007 | +0.006 |

Bootstrap, OL-Origin vs selected baselines:

| Comparison | ΔNDCG@3 90% CI | ΔHit@1 90% CI | JS improvement 90% CI |
|---|---:|---:|---:|
| ol_origin_vs_followers | +0.155 [+0.106, +0.214] | +0.106 [+0.056, +0.155] | +0.047 [+0.029, +0.064] |
| ol_origin_vs_visibility | +0.151 [+0.097, +0.207] | +0.104 [+0.055, +0.154] | +0.046 [+0.028, +0.065] |
| ol_origin_vs_rank_time | +0.099 [+0.076, +0.122] | +0.131 [+0.097, +0.171] | +0.035 [+0.022, +0.047] |
| ol_origin_vs_sentiment | +0.103 [+0.075, +0.130] | +0.089 [+0.052, +0.130] | +0.052 [+0.036, +0.069] |
| ol_origin_vs_novelty | +0.067 [+0.038, +0.099] | +0.097 [+0.062, +0.137] | +0.027 [+0.012, +0.042] |
| ol_origin_vs_history | +0.054 [+0.041, +0.068] | +0.036 [+0.018, +0.057] | +0.026 [+0.020, +0.032] |
| ol_origin_vs_no_ol_strong | +0.007 [+0.002, +0.011] | +0.006 [-0.004, +0.017] | +0.006 [+0.004, +0.007] |

## Ablation Rows

| Method | Events | Symbols | NDCG@3 | Hit@1 | Mass@3 | JS ↓ | ΔNDCG vs No-OL | ΔHit vs No-OL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OL only | 4055 | 17 | 0.475 | 0.193 | 0.606 | 0.360 | -0.189 | -0.180 |
| No-OL strong | 4055 | 17 | 0.664 | 0.373 | 0.773 | 0.313 | +0.000 | +0.000 |
| Shuffled OL | 4055 | 17 | 0.659 | 0.380 | 0.765 | 0.313 | -0.005 | +0.007 |
| Follower replacement | 4055 | 17 | 0.664 | 0.373 | 0.774 | 0.313 | +0.000 | +0.000 |
| Raw OL | 4055 | 17 | 0.664 | 0.382 | 0.771 | 0.309 | +0.000 | +0.008 |
| **Full OL-Origin** | 4055 | 17 | 0.671 | 0.379 | 0.776 | 0.308 | +0.007 | +0.006 |

Bootstrap, Full OL-Origin vs ablation baselines:

| Comparison | ΔNDCG@3 90% CI | ΔHit@1 90% CI | JS improvement 90% CI |
|---|---:|---:|---:|
| ol_origin_full_vs_no_ol_strong | +0.007 [+0.002, +0.012] | +0.006 [-0.003, +0.016] | +0.006 [+0.004, +0.007] |
| ol_origin_full_vs_ol_only | +0.196 [+0.159, +0.234] | +0.186 [+0.143, +0.229] | +0.052 [+0.036, +0.069] |
| ol_origin_full_vs_shuffled_ol_origin | +0.012 [+0.006, +0.017] | -0.001 [-0.014, +0.009] | +0.005 [+0.003, +0.007] |
| ol_origin_full_vs_follower_replacement | +0.006 [+0.001, +0.011] | +0.006 [-0.004, +0.016] | +0.005 [+0.004, +0.007] |
| ol_origin_full_vs_raw_ol_origin | +0.006 [-0.004, +0.017] | -0.002 [-0.022, +0.014] | +0.001 [-0.004, +0.007] |
| raw_ol_origin_vs_no_ol_strong | +0.000 [-0.011, +0.011] | +0.008 [-0.009, +0.023] | +0.005 [-0.000, +0.009] |
| shuffled_ol_origin_vs_no_ol_strong | -0.005 [-0.011, -0.000] | +0.007 [-0.004, +0.018] | +0.001 [-0.001, +0.002] |
| follower_replacement_vs_no_ol_strong | +0.000 [-0.000, +0.001] | +0.000 [-0.001, +0.002] | +0.000 [+0.000, +0.000] |
