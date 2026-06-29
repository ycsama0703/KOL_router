# Phase56 Econometric Appendix: Originator Role Incremental Test

Specification: candidate-level OLS with event fixed effects. Standard errors are clustered by event. Features are standardized using the main phase7 training split standardizer.

Added OL terms: `origin_ol`, `ol_x_visibility`, and `ol_x_novelty`. Baseline controls are the No-OL Strong feature set.

## Nested Model Fit

| Split | Rows | Events | Symbols | Baseline within R2 | Full within R2 | Delta R2 | SSR reduction | Wald chi2(3) | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 6891 | 1278 | 13 | 0.0926 | 0.1058 | +0.0132 | 1.45% | 85.69 | <0.001 |
| val | 5818 | 1477 | 16 | 0.0778 | 0.1255 | +0.0477 | 5.18% | 185.38 | <0.001 |

## Added OL-Term Coefficients

| Split | Term | Coef, standardized | Cluster SE | t | p |
|---|---|---:|---:|---:|---:|
| train | `origin_ol` | -6.8587 | 0.8316 | -8.25 | <0.001 |
| train | `ol_x_visibility` | +6.6677 | 0.8403 | +7.93 | <0.001 |
| train | `ol_x_novelty` | +0.4201 | 0.1349 | +3.11 | 0.002 |
| val | `origin_ol` | -11.0281 | 0.9471 | -11.64 | <0.001 |
| val | `ol_x_visibility` | +11.8350 | 0.9656 | +12.26 | <0.001 |
| val | `ol_x_novelty` | -0.2686 | 0.1166 | -2.30 | 0.021 |

Interpretation: this is an econometric diagnostic, not the main predictive benchmark. The validation split is the cleaner appendix result because it tests the within-event association in the held-out period. The joint Wald test asks whether the three OL-channel terms add explanatory power beyond the No-OL contextual controls.
