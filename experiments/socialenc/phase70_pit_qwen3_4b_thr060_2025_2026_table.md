# Phase70 PIT Qwen3-4B Benchmark thr=0.60 2025-2026

Protocol matches phase68: train 2022-06-01 to 2024-06-01; validation 2024-06-01 to 2025-06-01; final test 2025-06-01 to 2026-06-01; semantic threshold=0.60.

| Method | Events | Symbols | NDCG@3 | Hit@1 | Mass@3 | JS ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B-origin text | 1801 | 17 | 0.702 | 0.445 | 0.805 | 0.284 |

Bootstrap comparisons:

| Comparison | ΔNDCG@3 90% CI | ΔHit@1 90% CI | JS improvement 90% CI |
|---|---:|---:|---:|
| qwen3_embedding_4b_st_origin_text_vs_no_ol_strong | +0.037 [+0.004, +0.068] | +0.032 [-0.018, +0.082] | +0.027 [+0.007, +0.047] |
| qwen3_embedding_4b_st_origin_text_vs_ol_origin | +0.029 [-0.002, +0.061] | +0.025 [-0.030, +0.075] | +0.025 [+0.004, +0.047] |
| ol_origin_vs_qwen3_embedding_4b_st_origin_text | -0.029 [-0.059, +0.005] | -0.025 [-0.080, +0.030] | -0.025 [-0.048, -0.003] |
