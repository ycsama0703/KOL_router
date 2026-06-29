# Phase73 PIT No-Val Surface Diagnostics thr=0.55 2025-2026

Protocol matches phase72: train 2022-06-01 to 2025-06-01; no validation split; final test 2025-06-01 to 2026-06-01; fixed ridge alpha=1000.

| Method | Events | Symbols | NDCG@3 | Hit@1 | Mass@3 | JS ↓ | ΔNDCG vs No-OL | ΔHit vs No-OL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Symbol one-hot | 2028 | 17 | 0.587 | 0.277 | 0.722 | 0.361 | -0.102 | -0.144 |
| Text surface | 2028 | 17 | 0.597 | 0.312 | 0.715 | 0.333 | -0.092 | -0.109 |
| Symbol + surface | 2028 | 17 | 0.595 | 0.308 | 0.710 | 0.333 | -0.094 | -0.113 |

Bootstrap comparisons:

| Comparison | ΔNDCG@3 90% CI | ΔHit@1 90% CI | JS improvement 90% CI |
|---|---:|---:|---:|
| symbol_onehot_vs_no_ol_strong | -0.102 [-0.133, -0.073] | -0.144 [-0.195, -0.099] | -0.067 [-0.092, -0.047] |
| text_surface_vs_no_ol_strong | -0.092 [-0.148, -0.040] | -0.109 [-0.185, -0.036] | -0.040 [-0.071, -0.011] |
| symbol_plus_surface_vs_no_ol_strong | -0.094 [-0.138, -0.043] | -0.113 [-0.180, -0.048] | -0.040 [-0.068, -0.015] |
| ol_origin_vs_symbol_onehot | +0.107 [+0.074, +0.141] | +0.145 [+0.096, +0.194] | +0.068 [+0.047, +0.092] |
| ol_origin_vs_text_surface | +0.097 [+0.038, +0.149] | +0.110 [+0.040, +0.184] | +0.041 [+0.014, +0.071] |
| ol_origin_vs_symbol_plus_surface | +0.099 [+0.054, +0.142] | +0.114 [+0.044, +0.178] | +0.040 [+0.013, +0.064] |
