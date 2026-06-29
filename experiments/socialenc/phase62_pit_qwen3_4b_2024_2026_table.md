# Phase62 PIT Qwen3-4B Benchmark 2024-2026

Protocol matches phase61: train 2022-06-01 to 2023-06-01; validation 2023-06-01 to 2024-06-01; final test 2024-06-01 to 2026-06-01.

| Method | Events | Symbols | NDCG@3 | Hit@1 | Mass@3 | JS ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B-origin text | 4055 | 17 | 0.688 | 0.421 | 0.781 | 0.290 |

Bootstrap comparisons:

| Comparison | ΔNDCG@3 90% CI | ΔHit@1 90% CI | JS improvement 90% CI |
|---|---:|---:|---:|
| qwen3_embedding_4b_st_origin_text_vs_no_ol_strong | +0.024 [-0.005, +0.052] | +0.048 [-0.001, +0.087] | +0.023 [+0.007, +0.040] |
| qwen3_embedding_4b_st_origin_text_vs_ol_origin | +0.018 [-0.008, +0.043] | +0.042 [-0.000, +0.085] | +0.017 [-0.001, +0.036] |
| ol_origin_vs_qwen3_embedding_4b_st_origin_text | -0.018 [-0.043, +0.010] | -0.042 [-0.081, +0.004] | -0.017 [-0.036, +0.000] |
