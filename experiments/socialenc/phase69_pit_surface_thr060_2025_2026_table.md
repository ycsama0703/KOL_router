# Phase69 PIT Surface Diagnostics thr=0.60 2025-2026

Protocol matches phase68: train 2022-06-01 to 2024-06-01; validation 2024-06-01 to 2025-06-01; final test 2025-06-01 to 2026-06-01; semantic threshold=0.60.

| Method | Events | Symbols | NDCG@3 | Hit@1 | Mass@3 | JS ↓ | ΔNDCG vs No-OL | ΔHit vs No-OL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Symbol one-hot | 1801 | 17 | 0.517 | 0.250 | 0.654 | 0.403 | -0.149 | -0.163 |
| Text surface | 1801 | 17 | 0.549 | 0.289 | 0.681 | 0.369 | -0.116 | -0.124 |
| Symbol + surface | 1801 | 17 | 0.547 | 0.287 | 0.672 | 0.368 | -0.119 | -0.126 |

Bootstrap comparisons:

| Comparison | ΔNDCG@3 90% CI | ΔHit@1 90% CI | JS improvement 90% CI |
|---|---:|---:|---:|
| symbol_onehot_vs_no_ol_strong | -0.149 [-0.190, -0.109] | -0.163 [-0.214, -0.114] | -0.093 [-0.122, -0.069] |
| text_surface_vs_no_ol_strong | -0.116 [-0.171, -0.066] | -0.124 [-0.202, -0.054] | -0.059 [-0.090, -0.033] |
| symbol_plus_surface_vs_no_ol_strong | -0.119 [-0.168, -0.073] | -0.126 [-0.194, -0.061] | -0.058 [-0.084, -0.033] |
| ol_origin_vs_symbol_onehot | +0.156 [+0.117, +0.199] | +0.170 [+0.120, +0.217] | +0.094 [+0.070, +0.125] |
| ol_origin_vs_text_surface | +0.124 [+0.071, +0.175] | +0.131 [+0.056, +0.209] | +0.061 [+0.032, +0.095] |
| ol_origin_vs_symbol_plus_surface | +0.127 [+0.084, +0.171] | +0.133 [+0.066, +0.201] | +0.060 [+0.036, +0.089] |
