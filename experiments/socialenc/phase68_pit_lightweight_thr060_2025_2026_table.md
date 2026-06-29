# Phase68 PIT Lightweight thr=0.60 2025-2026 Test

Protocol: train 2022-06-01 to 2024-06-01; validation 2024-06-01 to 2025-06-01; final test 2025-06-01 to 2026-06-01. O_k/history are estimated only from data before each point-in-time block.

## Main Lightweight Rows

| Family | Method | Events | Symbols | NDCG@3 | Hit@1 | Mass@3 | JS ↓ | ΔNDCG vs No-OL | ΔHit vs No-OL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Scale | Follower | 1801 | 17 | 0.473 | 0.263 | 0.580 | 0.397 | -0.193 | -0.150 |
| Scale | Visibility | 1801 | 17 | 0.473 | 0.261 | 0.582 | 0.397 | -0.193 | -0.152 |
| Context | Rank/Time | 1801 | 17 | 0.516 | 0.248 | 0.654 | 0.387 | -0.149 | -0.165 |
| Context | Sentiment | 1801 | 17 | 0.505 | 0.247 | 0.622 | 0.401 | -0.160 | -0.166 |
| Context | Novelty | 1801 | 17 | 0.559 | 0.281 | 0.697 | 0.374 | -0.107 | -0.132 |
| Context | History | 1801 | 17 | 0.625 | 0.391 | 0.729 | 0.332 | -0.041 | -0.022 |
| Context | No-OL Strong | 1801 | 17 | 0.666 | 0.413 | 0.776 | 0.310 | +0.000 | +0.000 |
| Origin Role | OL Only | 1801 | 17 | 0.420 | 0.182 | 0.544 | 0.402 | -0.246 | -0.231 |
| Origin Role | **OL-Origin** | 1801 | 17 | 0.673 | 0.420 | 0.777 | 0.309 | +0.008 | +0.007 |

Bootstrap, OL-Origin vs selected baselines:

| Comparison | ΔNDCG@3 90% CI | ΔHit@1 90% CI | JS improvement 90% CI |
|---|---:|---:|---:|
| ol_origin_vs_followers | +0.201 [+0.132, +0.275] | +0.157 [+0.089, +0.232] | +0.088 [+0.061, +0.121] |
| ol_origin_vs_visibility | +0.200 [+0.135, +0.271] | +0.159 [+0.091, +0.238] | +0.088 [+0.062, +0.119] |
| ol_origin_vs_rank_time | +0.157 [+0.117, +0.199] | +0.172 [+0.123, +0.226] | +0.078 [+0.058, +0.099] |
| ol_origin_vs_sentiment | +0.168 [+0.135, +0.202] | +0.173 [+0.121, +0.224] | +0.092 [+0.068, +0.123] |
| ol_origin_vs_novelty | +0.114 [+0.079, +0.152] | +0.139 [+0.080, +0.202] | +0.065 [+0.044, +0.090] |
| ol_origin_vs_history | +0.048 [+0.027, +0.070] | +0.029 [-0.004, +0.066] | +0.024 [+0.017, +0.030] |
| ol_origin_vs_no_ol_strong | +0.008 [+0.002, +0.014] | +0.007 [-0.012, +0.024] | +0.001 [-0.000, +0.003] |

## Ablation Rows

| Method | Events | Symbols | NDCG@3 | Hit@1 | Mass@3 | JS ↓ | ΔNDCG vs No-OL | ΔHit vs No-OL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OL only | 1801 | 17 | 0.420 | 0.182 | 0.544 | 0.402 | -0.246 | -0.231 |
| No-OL strong | 1801 | 17 | 0.666 | 0.413 | 0.776 | 0.310 | +0.000 | +0.000 |
| Shuffled OL | 1801 | 17 | 0.668 | 0.419 | 0.774 | 0.310 | +0.002 | +0.006 |
| Follower replacement | 1801 | 17 | 0.666 | 0.414 | 0.775 | 0.310 | +0.000 | +0.001 |
| Raw OL | 1801 | 17 | 0.677 | 0.422 | 0.786 | 0.308 | +0.012 | +0.009 |
| **Full OL-Origin** | 1801 | 17 | 0.673 | 0.420 | 0.777 | 0.309 | +0.008 | +0.007 |

Bootstrap, Full OL-Origin vs ablation baselines:

| Comparison | ΔNDCG@3 90% CI | ΔHit@1 90% CI | JS improvement 90% CI |
|---|---:|---:|---:|
| ol_origin_full_vs_no_ol_strong | +0.008 [+0.001, +0.014] | +0.007 [-0.011, +0.022] | +0.001 [-0.000, +0.003] |
| ol_origin_full_vs_ol_only | +0.254 [+0.199, +0.314] | +0.238 [+0.174, +0.305] | +0.093 [+0.070, +0.124] |
| ol_origin_full_vs_shuffled_ol_origin | +0.006 [-0.001, +0.012] | +0.001 [-0.019, +0.017] | +0.001 [-0.001, +0.003] |
| ol_origin_full_vs_follower_replacement | +0.007 [+0.001, +0.014] | +0.006 [-0.013, +0.024] | +0.001 [-0.000, +0.003] |
| ol_origin_full_vs_raw_ol_origin | -0.004 [-0.013, +0.004] | -0.002 [-0.013, +0.012] | -0.000 [-0.003, +0.002] |
| raw_ol_origin_vs_no_ol_strong | +0.012 [+0.003, +0.021] | +0.009 [-0.013, +0.027] | +0.002 [-0.000, +0.004] |
| shuffled_ol_origin_vs_no_ol_strong | +0.002 [-0.002, +0.006] | +0.006 [-0.005, +0.015] | +0.001 [+0.000, +0.001] |
| follower_replacement_vs_no_ol_strong | +0.000 [-0.000, +0.001] | +0.001 [-0.001, +0.004] | +0.000 [-0.000, +0.000] |
