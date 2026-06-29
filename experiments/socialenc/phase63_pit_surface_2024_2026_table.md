# Phase63 PIT Surface Diagnostics 2024-2026

Protocol matches phase61: train 2022-06-01 to 2023-06-01; validation 2023-06-01 to 2024-06-01; final test 2024-06-01 to 2026-06-01.

| Method | Events | Symbols | NDCG@3 | Hit@1 | Mass@3 | JS ↓ | ΔNDCG vs No-OL | ΔHit vs No-OL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Symbol one-hot | 4055 | 17 | 0.587 | 0.283 | 0.714 | 0.360 | -0.077 | -0.090 |
| Text surface | 4055 | 17 | 0.576 | 0.328 | 0.677 | 0.344 | -0.088 | -0.045 |
| Symbol + surface | 4055 | 17 | 0.577 | 0.327 | 0.677 | 0.342 | -0.087 | -0.046 |

Bootstrap comparisons:

| Comparison | ΔNDCG@3 90% CI | ΔHit@1 90% CI | JS improvement 90% CI |
|---|---:|---:|---:|
| symbol_onehot_vs_no_ol_strong | -0.077 [-0.097, -0.053] | -0.090 [-0.128, -0.053] | -0.047 [-0.062, -0.031] |
| text_surface_vs_no_ol_strong | -0.088 [-0.153, -0.026] | -0.045 [-0.098, +0.013] | -0.031 [-0.061, -0.003] |
| symbol_plus_surface_vs_no_ol_strong | -0.087 [-0.151, -0.034] | -0.046 [-0.103, +0.018] | -0.028 [-0.058, +0.000] |
| ol_origin_vs_symbol_onehot | +0.083 [+0.059, +0.106] | +0.096 [+0.063, +0.136] | +0.053 [+0.036, +0.070] |
| ol_origin_vs_text_surface | +0.094 [+0.039, +0.166] | +0.051 [-0.008, +0.110] | +0.036 [+0.008, +0.067] |
| ol_origin_vs_symbol_plus_surface | +0.094 [+0.042, +0.151] | +0.052 [-0.006, +0.106] | +0.034 [+0.007, +0.060] |
