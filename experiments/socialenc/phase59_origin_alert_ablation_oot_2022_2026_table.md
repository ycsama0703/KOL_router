# Phase59 2022-2026 OOT Ablation

Frozen protocol: source-score history before 2020-01-01; router train 2020-01-01 to 2021-06-01; OOT test 2022-06-01 to 2026-06-22.

| Method | Events | Symbols | NDCG@3 | Hit@1 | Mass@3 | JS ↓ | ΔNDCG vs No-OL | ΔHit vs No-OL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OL only | 3623 | 17 | 0.695 | 0.343 | 0.891 | 0.289 | -0.077 | -0.142 |
| No-OL strong | 3623 | 17 | 0.773 | 0.485 | 0.929 | 0.268 | +0.000 | +0.000 |
| Shuffled OL | 3623 | 17 | 0.775 | 0.488 | 0.931 | 0.265 | +0.002 | +0.003 |
| Follower replacement | 3623 | 17 | 0.775 | 0.489 | 0.932 | 0.264 | +0.003 | +0.004 |
| Raw OL | 3623 | 17 | 0.781 | 0.500 | 0.936 | 0.259 | +0.008 | +0.015 |
| **Full OL-Origin** | 3623 | 17 | 0.779 | 0.498 | 0.931 | 0.259 | +0.006 | +0.013 |

Bootstrap support:

| Comparison | ΔNDCG@3 90% CI | ΔHit@1 90% CI | JS improvement 90% CI |
|---|---:|---:|---:|
| ol_origin_full_vs_no_ol_strong | +0.006 [-0.002, +0.015] | +0.013 [-0.013, +0.041] | +0.009 [+0.002, +0.014] |
| ol_origin_full_vs_ol_only | +0.083 [+0.060, +0.105] | +0.155 [+0.099, +0.220] | +0.030 [+0.012, +0.049] |
| ol_origin_full_vs_shuffled_ol_origin | +0.004 [-0.005, +0.013] | +0.010 [-0.018, +0.035] | +0.006 [-0.000, +0.011] |
| ol_origin_full_vs_follower_replacement | +0.004 [-0.006, +0.014] | +0.009 [-0.016, +0.035] | +0.005 [-0.001, +0.011] |
| ol_origin_full_vs_raw_ol_origin | -0.002 [-0.008, +0.004] | -0.002 [-0.020, +0.016] | -0.000 [-0.003, +0.002] |
| raw_ol_origin_vs_no_ol_strong | +0.008 [-0.001, +0.016] | +0.015 [-0.007, +0.040] | +0.009 [+0.004, +0.014] |
| shuffled_ol_origin_vs_no_ol_strong | +0.002 [-0.001, +0.005] | +0.003 [-0.005, +0.013] | +0.003 [+0.002, +0.004] |
| follower_replacement_vs_no_ol_strong | +0.003 [-0.003, +0.009] | +0.004 [-0.008, +0.018] | +0.003 [+0.001, +0.005] |
