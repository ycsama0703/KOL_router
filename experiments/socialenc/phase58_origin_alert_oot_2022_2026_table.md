# Phase58 2022-2026 Out-of-Time Main Rows

Frozen protocol: source-score history before 2020-01-01; router train 2020-01-01 to 2021-06-01; OOT test 2022-06-01 to 2026-06-22.

| Family | Method | Events | Symbols | NDCG@3 | Hit@1 | Mass@3 | JS ↓ | ΔNDCG | ΔHit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Scale | Follower | 3623 | 17 | 0.752 | 0.479 | 0.898 | 0.271 | -0.021 | -0.006 |
| Scale | Visibility | 3623 | 17 | 0.749 | 0.474 | 0.898 | 0.271 | -0.024 | -0.011 |
| Context | Rank/Time | 3623 | 17 | 0.715 | 0.391 | 0.899 | 0.290 | -0.058 | -0.093 |
| Context | Sentiment | 3623 | 17 | 0.712 | 0.395 | 0.899 | 0.293 | -0.060 | -0.090 |
| Context | Novelty | 3623 | 17 | 0.721 | 0.388 | 0.906 | 0.286 | -0.052 | -0.097 |
| Context | History | 3623 | 17 | 0.707 | 0.397 | 0.874 | 0.282 | -0.065 | -0.087 |
| Context | No-OL Strong | 3623 | 17 | 0.773 | 0.485 | 0.929 | 0.268 | +0.000 | +0.000 |
| Origin Role | OL Only | 3623 | 17 | 0.695 | 0.343 | 0.891 | 0.289 | -0.077 | -0.142 |
| Origin Role | **OL-Origin** | 3623 | 17 | 0.779 | 0.498 | 0.931 | 0.259 | +0.006 | +0.013 |
| Surface Text | Symbol one-hot | 3623 | 17 | 0.715 | 0.391 | 0.899 | 0.291 | -0.058 | -0.093 |
| Surface Text | Text surface | 3623 | 17 | 0.785 | 0.522 | 0.926 | 0.246 | +0.013 | +0.038 |
| Surface Text | Symbol + surface | 3623 | 17 | 0.790 | 0.531 | 0.928 | 0.242 | +0.017 | +0.046 |

Bootstrap support, OL-Origin vs selected baselines:

| Comparison | ΔNDCG@3 90% CI | ΔHit@1 90% CI | JS improvement 90% CI |
|---|---:|---:|---:|
| ol_origin_vs_no_ol_strong | +0.006 [-0.002, +0.014] | +0.013 [-0.013, +0.039] | +0.009 [+0.002, +0.015] |
| ol_origin_vs_followers | +0.027 [+0.004, +0.052] | +0.019 [-0.028, +0.073] | +0.012 [-0.004, +0.029] |
| ol_origin_vs_visibility | +0.030 [+0.007, +0.052] | +0.024 [-0.028, +0.078] | +0.012 [-0.004, +0.029] |
| ol_origin_vs_text_surface | -0.007 [-0.032, +0.021] | -0.024 [-0.080, +0.037] | -0.013 [-0.035, +0.005] |
| ol_origin_vs_symbol_plus_surface | -0.011 [-0.036, +0.014] | -0.033 [-0.086, +0.023] | -0.017 [-0.038, +0.005] |
