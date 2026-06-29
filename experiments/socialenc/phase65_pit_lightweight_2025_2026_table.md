# Phase65 PIT Lightweight 2025-2026 Test

Protocol: train 2022-06-01 to 2024-06-01; validation 2024-06-01 to 2025-06-01; final test 2025-06-01 to 2026-06-01. O_k/history are estimated only from data before each point-in-time block.

## Main Lightweight Rows

| Family | Method | Events | Symbols | NDCG@3 | Hit@1 | Mass@3 | JS ↓ | ΔNDCG vs No-OL | ΔHit vs No-OL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Scale | Follower | 2028 | 17 | 0.526 | 0.290 | 0.636 | 0.355 | -0.163 | -0.131 |
| Scale | Visibility | 2028 | 17 | 0.526 | 0.287 | 0.637 | 0.355 | -0.163 | -0.134 |
| Context | Rank/Time | 2028 | 17 | 0.587 | 0.277 | 0.722 | 0.336 | -0.102 | -0.144 |
| Context | Sentiment | 2028 | 17 | 0.572 | 0.282 | 0.686 | 0.359 | -0.118 | -0.139 |
| Context | Novelty | 2028 | 17 | 0.616 | 0.295 | 0.757 | 0.328 | -0.073 | -0.126 |
| Context | History | 2028 | 17 | 0.635 | 0.367 | 0.743 | 0.322 | -0.055 | -0.054 |
| Context | No-OL Strong | 2028 | 17 | 0.689 | 0.421 | 0.789 | 0.293 | +0.000 | +0.000 |
| Origin Role | OL Only | 2028 | 17 | 0.466 | 0.196 | 0.598 | 0.361 | -0.223 | -0.225 |
| Origin Role | **OL-Origin** | 2028 | 17 | 0.695 | 0.422 | 0.795 | 0.293 | +0.005 | +0.001 |

Bootstrap, OL-Origin vs selected baselines:

| Comparison | ΔNDCG@3 90% CI | ΔHit@1 90% CI | JS improvement 90% CI |
|---|---:|---:|---:|
| ol_origin_vs_followers | +0.168 [+0.115, +0.229] | +0.132 [+0.070, +0.200] | +0.062 [+0.040, +0.088] |
| ol_origin_vs_visibility | +0.169 [+0.111, +0.232] | +0.135 [+0.069, +0.200] | +0.062 [+0.038, +0.088] |
| ol_origin_vs_rank_time | +0.107 [+0.075, +0.140] | +0.145 [+0.098, +0.195] | +0.043 [+0.029, +0.058] |
| ol_origin_vs_sentiment | +0.123 [+0.081, +0.159] | +0.140 [+0.089, +0.192] | +0.066 [+0.045, +0.090] |
| ol_origin_vs_novelty | +0.078 [+0.042, +0.114] | +0.127 [+0.075, +0.179] | +0.035 [+0.014, +0.058] |
| ol_origin_vs_history | +0.060 [+0.040, +0.079] | +0.055 [+0.024, +0.089] | +0.030 [+0.018, +0.040] |
| ol_origin_vs_no_ol_strong | +0.005 [-0.002, +0.013] | +0.001 [-0.012, +0.015] | +0.001 [-0.001, +0.003] |

## Ablation Rows

| Method | Events | Symbols | NDCG@3 | Hit@1 | Mass@3 | JS ↓ | ΔNDCG vs No-OL | ΔHit vs No-OL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OL only | 2028 | 17 | 0.466 | 0.196 | 0.598 | 0.361 | -0.223 | -0.225 |
| No-OL strong | 2028 | 17 | 0.689 | 0.421 | 0.789 | 0.293 | +0.000 | +0.000 |
| Shuffled OL | 2028 | 17 | 0.685 | 0.413 | 0.785 | 0.292 | -0.004 | -0.008 |
| Follower replacement | 2028 | 17 | 0.688 | 0.418 | 0.789 | 0.293 | -0.001 | -0.003 |
| Raw OL | 2028 | 17 | 0.696 | 0.426 | 0.798 | 0.291 | +0.007 | +0.005 |
| **Full OL-Origin** | 2028 | 17 | 0.695 | 0.422 | 0.795 | 0.293 | +0.005 | +0.001 |

Bootstrap, Full OL-Origin vs ablation baselines:

| Comparison | ΔNDCG@3 90% CI | ΔHit@1 90% CI | JS improvement 90% CI |
|---|---:|---:|---:|
| ol_origin_full_vs_no_ol_strong | +0.005 [-0.002, +0.013] | +0.001 [-0.014, +0.017] | +0.001 [-0.001, +0.003] |
| ol_origin_full_vs_ol_only | +0.229 [+0.177, +0.275] | +0.226 [+0.168, +0.290] | +0.069 [+0.046, +0.096] |
| ol_origin_full_vs_shuffled_ol_origin | +0.009 [-0.002, +0.021] | +0.009 [-0.011, +0.029] | -0.000 [-0.003, +0.002] |
| ol_origin_full_vs_follower_replacement | +0.006 [-0.001, +0.014] | +0.004 [-0.012, +0.019] | +0.001 [-0.002, +0.003] |
| ol_origin_full_vs_raw_ol_origin | -0.002 [-0.011, +0.007] | -0.003 [-0.023, +0.016] | -0.002 [-0.006, +0.002] |
| raw_ol_origin_vs_no_ol_strong | +0.007 [-0.002, +0.017] | +0.005 [-0.009, +0.018] | +0.003 [-0.001, +0.007] |
| shuffled_ol_origin_vs_no_ol_strong | -0.004 [-0.015, +0.003] | -0.008 [-0.025, +0.004] | +0.001 [-0.000, +0.002] |
| follower_replacement_vs_no_ol_strong | -0.001 [-0.004, +0.000] | -0.003 [-0.010, +0.002] | +0.000 [+0.000, +0.000] |
