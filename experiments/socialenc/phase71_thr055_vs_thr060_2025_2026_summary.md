# Phase71 Threshold Sensitivity Summary: 0.55 vs 0.60

Protocol for both thresholds:

| Role | Period |
|---|---|
| Train | 2022-06-01 to 2024-06-01 |
| Validation | 2024-06-01 to 2025-06-01 |
| Final test | 2025-06-01 to 2026-06-01 |

Point-in-time history cutoffs:

| Block | History cutoff |
|---|---|
| train_2022_2023 | before 2022-06-01 |
| train_2023_2024 | before 2023-06-01 |
| validation | before 2024-06-01 |
| test_2025_2026 | before 2025-06-01 |

## Panel Size

| Threshold | Rows | Train rows | Validation rows | Test rows | Test events | Symbols |
|---:|---:|---:|---:|---:|---:|---:|
| 0.55 | 59,802 | 29,687 | 15,330 | 14,785 | 2,028 | 17 |
| 0.60 | 66,553 | 32,759 | 17,285 | 16,509 | 1,801 | 17 |

Raising the threshold creates more origin-candidate rows because frames split
more aggressively, but fewer test events survive the metric filter.

## Main Comparison

| Threshold | Method | NDCG@3 | Hit@1 | Mass@3 | JS ↓ |
|---:|---|---:|---:|---:|---:|
| 0.55 | No-OL Strong | 0.689 | 0.421 | 0.789 | 0.293 |
| 0.55 | OL-Origin | 0.695 | 0.422 | 0.795 | 0.293 |
| 0.55 | Raw OL | 0.696 | 0.426 | 0.798 | 0.291 |
| 0.55 | Text surface | 0.599 | 0.314 | 0.715 | 0.334 |
| 0.55 | Qwen3-4B | 0.710 | 0.446 | 0.794 | 0.277 |
| 0.60 | No-OL Strong | 0.666 | 0.413 | 0.776 | 0.310 |
| 0.60 | OL-Origin | 0.673 | 0.420 | 0.777 | 0.309 |
| 0.60 | Raw OL | 0.677 | 0.422 | 0.786 | 0.308 |
| 0.60 | Text surface | 0.549 | 0.289 | 0.681 | 0.369 |
| 0.60 | Qwen3-4B | 0.702 | 0.445 | 0.805 | 0.284 |

## Key Bootstrap Comparisons

| Threshold | Comparison | ΔNDCG@3 90% CI | ΔHit@1 90% CI | JS improvement 90% CI |
|---:|---|---:|---:|---:|
| 0.55 | OL-Origin vs No-OL Strong | +0.005 [-0.002, +0.013] | +0.001 [-0.012, +0.015] | +0.001 [-0.001, +0.003] |
| 0.55 | Qwen3-4B vs OL-Origin | +0.015 [-0.014, +0.042] | +0.024 [-0.040, +0.080] | +0.016 [-0.003, +0.034] |
| 0.60 | OL-Origin vs No-OL Strong | +0.008 [+0.002, +0.014] | +0.007 [-0.012, +0.024] | +0.001 [-0.000, +0.003] |
| 0.60 | Qwen3-4B vs OL-Origin | +0.029 [-0.002, +0.061] | +0.025 [-0.030, +0.075] | +0.025 [+0.004, +0.047] |

## Current Read

Threshold 0.60 makes the OL-Origin gain over No-OL Strong cleaner on NDCG@3,
but it lowers the absolute NDCG of most lightweight and surface rows. Qwen3-4B
remains strongest under both thresholds.

Raw OL remains slightly above residualized OL-Origin under both thresholds, so
residualization should be framed as a cleaner deconfounded construction rather
than an empirically dominant variant on this split.
