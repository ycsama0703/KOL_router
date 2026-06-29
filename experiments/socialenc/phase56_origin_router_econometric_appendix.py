"""Econometric appendix checks for the OL-Origin router.

This phase is not the main predictive experiment. It estimates candidate-level
within-event regressions to document whether the originator role channel has
incremental explanatory power after the non-OL contextual controls are held
fixed.

Specification:
  log_future_reach_{i,e} =
      event FE_e + X_{i,e}' beta + gamma origin_ol_{i,e}
      + delta1 ol_x_visibility_{i,e} + delta2 ol_x_novelty_{i,e} + eps_{i,e}

Inference:
  - event fixed effects are implemented by demeaning y and X within event_id;
  - standard errors are clustered by event_id;
  - all features are standardized using the training split standardizer from
    the main phase7 ridge setup.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import phase5_sentiment_reconstruction as p5  # noqa: E402
import phase7_origin_alert as p7  # noqa: E402

OUT_JSON = pathlib.Path(__file__).with_name("phase56_origin_router_econometric_appendix_result.json")
OUT_MD = pathlib.Path(__file__).with_name("phase56_origin_router_econometric_appendix_table.md")

THR = 0.55
ORIGIN_WINDOW = {"name": "first10", "max_rank": 10}
TARGET = "log_future_reach"
NO_OL_FEATURES = p7.FEATURE_SETS["no_ol_strong"]
ADDED_FEATURES = ["origin_ol", "ol_x_visibility", "ol_x_novelty"]
FULL_FEATURES = NO_OL_FEATURES + ADDED_FEATURES


def log(msg: str) -> None:
    print(msg, flush=True)


def build_main_panel():
    rows_by = {}
    emb_by = {}
    all_rows = []
    for sym in p5.SYMS:
        rows, emb = p5.load_symbol(sym)
        rows_by[sym] = rows
        emb_by[sym] = emb
        all_rows.extend(rows)
    meta = p5.compute_metadata(all_rows)
    ol = p5.compute_oltrait(all_rows, meta)
    hist = p7.compute_origin_history(rows_by, emb_by, THR)
    panel, events = p7.build_origin_panel(rows_by, emb_by, meta, ol, hist, THR, ORIGIN_WINDOW)
    return panel, events


def train_feature_standardizer(panel, features):
    X = np.array([[r.get(f, np.nan) for f in features] for r in panel], dtype=float)
    y = np.array([r[TARGET] for r in panel], dtype=float)
    train = np.array([r["split"] == "train" for r in panel], dtype=bool)
    good_train = train & np.isfinite(y)
    med, mu, sd = p5.train_standardizer(X, good_train)
    return {
        f: {"median": float(a), "mean": float(b), "sd": float(c)}
        for f, a, b, c in zip(features, med, mu, sd)
    }


def standardizer_arrays(standardizer, features):
    med = np.array([standardizer[f]["median"] for f in features], dtype=float)
    mu = np.array([standardizer[f]["mean"] for f in features], dtype=float)
    sd = np.array([standardizer[f]["sd"] for f in features], dtype=float)
    return med, mu, sd


def design(panel, split, features, standardizer):
    med, mu, sd = standardizer_arrays(standardizer, features)
    rows = [r for r in panel if r["split"] == split and np.isfinite(r[TARGET])]
    counts = collections.Counter(r["event_id"] for r in rows)
    rows = [r for r in rows if counts[r["event_id"]] >= 2]
    X = np.array([[r.get(f, np.nan) for f in features] for r in rows], dtype=float)
    Xs = p5.apply_standardizer(X, med, mu, sd)
    y = np.array([r[TARGET] for r in rows], dtype=float)
    groups = np.array([r["event_id"] for r in rows], dtype=object)
    syms = np.array([r["sym"] for r in rows], dtype=object)
    return rows, within_demean(y, Xs, groups), groups, syms


def within_demean(y, X, groups):
    y2 = y.copy()
    X2 = X.copy()
    by = collections.defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    for idx in by.values():
        ii = np.array(idx, dtype=int)
        y2[ii] -= y2[ii].mean()
        X2[ii] -= X2[ii].mean(axis=0)
    return y2, X2


def fit_cluster_ols(y, X, groups):
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    ssr = float(resid @ resid)
    sst = float(y @ y)
    r2 = float(1.0 - ssr / max(sst, 1e-12))

    meat = np.zeros((k, k), dtype=float)
    unique_groups = list(dict.fromkeys(groups.tolist()))
    for g in unique_groups:
        idx = np.where(groups == g)[0]
        xg = X[idx]
        ug = resid[idx][:, None]
        xu = xg.T @ ug
        meat += xu @ xu.T
    g = len(unique_groups)
    correction = (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
    vcov = correction * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    t = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p = 2 * stats.t.sf(np.abs(t), df=max(g - 1, 1))
    return {
        "n": n,
        "k": k,
        "n_clusters": g,
        "beta": beta,
        "se_cluster_event": se,
        "t_cluster_event": t,
        "p_cluster_event": p,
        "vcov_cluster_event": vcov,
        "ssr": ssr,
        "sst_within": sst,
        "r2_within": r2,
    }


def wald_added(full_fit, feature_names, added):
    idx = [feature_names.index(f) for f in added]
    b = full_fit["beta"][idx]
    v = full_fit["vcov_cluster_event"][np.ix_(idx, idx)]
    w = float(b.T @ np.linalg.pinv(v) @ b)
    df = len(idx)
    return {"wald_chi2": w, "df": df, "p_value": float(stats.chi2.sf(w, df))}


def split_result(panel, split, standardizer):
    rows_full, (y_full, X_full), groups, syms = design(panel, split, FULL_FEATURES, standardizer)
    _rows_base, (y_base, X_base), groups_base, _syms_base = design(panel, split, NO_OL_FEATURES, standardizer)
    if not np.array_equal(groups, groups_base) or not np.allclose(y_full, y_base):
        raise RuntimeError("baseline/full designs are not aligned")
    base = fit_cluster_ols(y_base, X_base, groups)
    full = fit_cluster_ols(y_full, X_full, groups)
    added_rows = []
    for f in ADDED_FEATURES:
        j = FULL_FEATURES.index(f)
        added_rows.append({
            "feature": f,
            "coef_standardized": float(full["beta"][j]),
            "cluster_se": float(full["se_cluster_event"][j]),
            "t": float(full["t_cluster_event"][j]),
            "p": float(full["p_cluster_event"][j]),
        })
    return {
        "split": split,
        "n_rows": len(rows_full),
        "n_events": int(len(set(groups.tolist()))),
        "n_symbols": int(len(set(syms.tolist()))),
        "baseline_no_ol": {
            "r2_within": base["r2_within"],
            "ssr": base["ssr"],
            "n_features": len(NO_OL_FEATURES),
        },
        "full_origin": {
            "r2_within": full["r2_within"],
            "ssr": full["ssr"],
            "n_features": len(FULL_FEATURES),
            "added_feature_coefficients": added_rows,
        },
        "incremental": {
            "delta_r2_within": float(full["r2_within"] - base["r2_within"]),
            "ssr_reduction_pct": float((base["ssr"] - full["ssr"]) / max(base["ssr"], 1e-12)),
            "wald_added_terms": wald_added(full, FULL_FEATURES, ADDED_FEATURES),
        },
    }


def fmt_p(p):
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def write_markdown(result):
    lines = [
        "# Phase56 Econometric Appendix: Originator Role Incremental Test",
        "",
        "Specification: candidate-level OLS with event fixed effects. Standard errors are clustered by event. Features are standardized using the main phase7 training split standardizer.",
        "",
        "Added OL terms: `origin_ol`, `ol_x_visibility`, and `ol_x_novelty`. Baseline controls are the No-OL Strong feature set.",
        "",
        "## Nested Model Fit",
        "",
        "| Split | Rows | Events | Symbols | Baseline within R2 | Full within R2 | Delta R2 | SSR reduction | Wald chi2(3) | p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in result["splits"]:
        w = s["incremental"]["wald_added_terms"]
        lines.append(
            f"| {s['split']} | {s['n_rows']} | {s['n_events']} | {s['n_symbols']} | "
            f"{s['baseline_no_ol']['r2_within']:.4f} | {s['full_origin']['r2_within']:.4f} | "
            f"{s['incremental']['delta_r2_within']:+.4f} | {100*s['incremental']['ssr_reduction_pct']:.2f}% | "
            f"{w['wald_chi2']:.2f} | {fmt_p(w['p_value'])} |"
        )
    lines += [
        "",
        "## Added OL-Term Coefficients",
        "",
        "| Split | Term | Coef, standardized | Cluster SE | t | p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for s in result["splits"]:
        for r in s["full_origin"]["added_feature_coefficients"]:
            lines.append(
                f"| {s['split']} | `{r['feature']}` | {r['coef_standardized']:+.4f} | "
                f"{r['cluster_se']:.4f} | {r['t']:+.2f} | {fmt_p(r['p'])} |"
            )
    lines += [
        "",
        "Interpretation: this is an econometric diagnostic, not the main predictive benchmark. The validation split is the cleaner appendix result because it tests the within-event association in the held-out period. The joint Wald test asks whether the three OL-channel terms add explanatory power beyond the No-OL contextual controls.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    t0 = time.time()
    log("[1/4] Rebuilding main phase7 panel")
    panel, events = build_main_panel()
    log("[2/4] Training feature standardizer")
    standardizer = train_feature_standardizer(panel, FULL_FEATURES)
    result = {
        "task": "econometric_appendix_originator_role_incremental_test",
        "setting": {
            "threshold": THR,
            "origin_window": ORIGIN_WINDOW["name"],
            "target": TARGET,
            "baseline_features": NO_OL_FEATURES,
            "added_features": ADDED_FEATURES,
            "full_features": FULL_FEATURES,
        },
        "panel": {
            "n_rows": len(panel),
            "n_events_raw": len(events),
            "n_train_rows": sum(r["split"] == "train" for r in panel),
            "n_val_rows": sum(r["split"] == "val" for r in panel),
        },
        "splits": [],
    }
    log("[3/4] Estimating within-event regressions")
    for split in ["train", "val"]:
        result["splits"].append(split_result(panel, split, standardizer))
    result["elapsed_sec"] = time.time() - t0
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result)
    log(f"[4/4] wrote {OUT_JSON.name} and {OUT_MD.name}")
    log(f"elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
