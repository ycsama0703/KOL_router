# Phase67 PIT Qwen3-4B Benchmark 2025-2026

Protocol matches phase65: train 2022-06-01 to 2024-06-01; validation 2024-06-01 to 2025-06-01; final test 2025-06-01 to 2026-06-01.

| Method | Events | Symbols | NDCG@3 | Hit@1 | Mass@3 | JS ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B-origin text | 2028 | 17 | 0.710 | 0.446 | 0.794 | 0.277 |

Bootstrap comparisons:

| Comparison | ΔNDCG@3 90% CI | ΔHit@1 90% CI | JS improvement 90% CI |
|---|---:|---:|---:|
| qwen3_embedding_4b_st_origin_text_vs_no_ol_strong | +0.020 [-0.007, +0.047] | +0.025 [-0.039, +0.081] | +0.016 [-0.001, +0.034] |
| qwen3_embedding_4b_st_origin_text_vs_ol_origin | +0.015 [-0.014, +0.042] | +0.024 [-0.040, +0.080] | +0.016 [-0.003, +0.034] |
| ol_origin_vs_qwen3_embedding_4b_st_origin_text | -0.015 [-0.043, +0.011] | -0.024 [-0.081, +0.035] | -0.016 [-0.034, +0.002] |
