# KOL Router Model and Econometric Appendix

This note records two appendix-facing checks for the OL-Origin router:

1. the fitted linear ridge parameters used by the main-table OL-Origin model;
2. an econometric fixed-effects diagnostic testing whether the OL channel adds
   explanatory power beyond non-OL contextual controls.

The purpose is not to replace the main ranking benchmark. The purpose is to make
the simple model transparent and to show that the discovered KOL-originator
structure has measurable within-event signal.

## A. Main OL-Origin Ridge Parameters

Source artifacts:

- Script: `experiments/socialenc/phase57_origin_router_ridge_params.py`
- JSON: `experiments/socialenc/phase57_origin_router_ridge_params_result.json`
- Table: `experiments/socialenc/phase57_origin_router_ridge_params_table.md`

Main-table setting:

| Item | Value |
|---|---:|
| Semantic threshold | 0.55 |
| Origin window | first10 |
| Target | `log_future_reach` |
| Feature set | `ol_origin` |
| Ridge alpha | 3.0 |
| Train rows | 6,907 |
| Validation rows | 5,915 |
| Validation events | 785 |

Reproduced validation metrics:

| Metric | Value |
|---|---:|
| NDCG@3 | 0.7551 |
| Hit@1 | 0.4934 |
| Mass@3 | 0.9012 |
| JS | 0.2626 |

The fitted model uses standardized features:

```text
score_i = 2.1691 + sum_j beta_j * standardized(x_ij)
```

Key standardized coefficients:

| Feature | Standardized coef |
|---|---:|
| `origin_ol` | -5.0147 |
| `origin_logfoll` | +0.4057 |
| `elapsed_hours` | -0.3146 |
| `prior_frame_count` | -0.5733 |
| `novelty_global` | -0.8810 |
| `hist_success_rate` | +0.4637 |
| `ol_x_visibility` | +4.9235 |
| `ol_x_novelty` | +0.3500 |

The raw-space OL channel is approximately:

```text
d score / d origin_ol
  ~= -2.6655 + 0.2015 * origin_logfoll + 0.6074 * novelty_global
```

Interpretation:

`origin_ol` should not be read as a standalone KOL ranking. The fitted model says
that the originator role becomes useful when coupled with current frame context,
especially visibility and novelty. This supports the paper narrative: the model
does not simply label some KOLs as always good; it routes newly originated frames
using `KOL role × current context`.

## B. Econometric Fixed-Effects Diagnostic

Source artifacts:

- Script: `experiments/socialenc/phase56_origin_router_econometric_appendix.py`
- JSON: `experiments/socialenc/phase56_origin_router_econometric_appendix_result.json`
- Table: `experiments/socialenc/phase56_origin_router_econometric_appendix_table.md`

Specification:

```text
log_future_reach_{i,e}
  = event FE_e
  + X_{i,e}' beta
  + gamma origin_ol_{i,e}
  + delta_1 ol_x_visibility_{i,e}
  + delta_2 ol_x_novelty_{i,e}
  + eps_{i,e}
```

Where:

- `i` indexes candidate origin frames;
- `e` indexes symbol-day events;
- `X` is the No-OL Strong control set;
- event fixed effects compare candidates within the same event;
- standard errors are clustered by event;
- features are standardized using the main phase7 training split standardizer.

Nested model fit:

| Split | Rows | Events | Symbols | Baseline within R2 | Full within R2 | Delta R2 | SSR reduction | Wald chi2(3) | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 6,891 | 1,278 | 13 | 0.0926 | 0.1058 | +0.0132 | 1.45% | 85.69 | <0.001 |
| validation | 5,818 | 1,477 | 16 | 0.0778 | 0.1255 | +0.0477 | 5.18% | 185.38 | <0.001 |

Added OL-term coefficients:

| Split | Term | Coef, standardized | Cluster SE | t | p |
|---|---|---:|---:|---:|---:|
| train | `origin_ol` | -6.8587 | 0.8316 | -8.25 | <0.001 |
| train | `ol_x_visibility` | +6.6677 | 0.8403 | +7.93 | <0.001 |
| train | `ol_x_novelty` | +0.4201 | 0.1349 | +3.11 | 0.002 |
| validation | `origin_ol` | -11.0281 | 0.9471 | -11.64 | <0.001 |
| validation | `ol_x_visibility` | +11.8350 | 0.9656 | +12.26 | <0.001 |
| validation | `ol_x_novelty` | -0.2686 | 0.1166 | -2.30 | 0.021 |

Appendix interpretation:

The validation split is the cleanest diagnostic result. Within the same
symbol-day event, adding the OL channel increases within-event R2 from 0.0778 to
0.1255 and reduces SSR by 5.18%. The three OL-channel terms are jointly
significant with event-clustered standard errors, Wald chi2(3) = 185.38,
p < 0.001.

This supports a modest but useful claim: the discovered originator-role channel
contains incremental within-event information beyond follower scale, origin
rank/time, sentiment, novelty, and historical non-OL controls. It should still
be presented as a diagnostic appendix, while the ranking benchmark remains the
primary evidence for router performance.
