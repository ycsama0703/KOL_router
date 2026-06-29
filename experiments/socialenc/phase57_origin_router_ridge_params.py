"""Export the fitted OL-Origin ridge parameters for the main table setting."""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import phase5_sentiment_reconstruction as p5  # noqa: E402
import phase7_origin_alert as p7  # noqa: E402

OUT_JSON = pathlib.Path(__file__).with_name("phase57_origin_router_ridge_params_result.json")
OUT_MD = pathlib.Path(__file__).with_name("phase57_origin_router_ridge_params_table.md")

THR = 0.55
ORIGIN_WINDOW = {"name": "first10", "max_rank": 10}
TARGET = "log_future_reach"
FEATURE_SET = "ol_origin"
ALPHA = 3.0


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
    return p7.build_origin_panel(rows_by, emb_by, meta, ol, hist, THR, ORIGIN_WINDOW)[0]


def fit_params(panel):
    features = p7.FEATURE_SETS[FEATURE_SET]
    X = np.array([[r.get(f, np.nan) for f in features] for r in panel], dtype=float)
    y = np.array([r[TARGET] for r in panel], dtype=float)
    train = np.array([r["split"] == "train" for r in panel], dtype=bool)
    good_train = train & np.isfinite(y)
    med, mu, sd = p5.train_standardizer(X, good_train)
    Xs = p5.apply_standardizer(X, med, mu, sd)
    Xtr = np.column_stack([np.ones(good_train.sum()), Xs[good_train]])
    ytr = y[good_train]
    lam = np.eye(Xtr.shape[1]) * ALPHA
    lam[0, 0] = 0.0
    beta = np.linalg.solve(Xtr.T @ Xtr + lam, Xtr.T @ ytr)
    scores = np.column_stack([np.ones(len(panel)), Xs]) @ beta
    evs = p7.event_rows(panel, scores, TARGET)
    raw_coef = beta[1:] / sd
    raw_intercept = beta[0] - np.sum(beta[1:] * mu / sd)
    return {
        "features": features,
        "beta": beta,
        "med": med,
        "mu": mu,
        "sd": sd,
        "raw_coef": raw_coef,
        "raw_intercept": raw_intercept,
        "metrics": {m: p7.symbal_mean(evs, m) for m in p7.METRICS},
        "n_val_events": len(evs),
        "n_train_rows": int(good_train.sum()),
        "n_val_rows": int((~train).sum()),
    }


def write_markdown(result):
    lines = [
        "# Phase57 Main OL-Origin Ridge Parameters",
        "",
        "Main-table setting: threshold 0.55, origin window first10, target `log_future_reach`, ridge alpha 3.0.",
        "",
        f"Validation metrics reproduced: NDCG@3 {result['metrics']['ndcg3']:.4f}, Hit@1 {result['metrics']['hit1']:.4f}, Mass@3 {result['metrics']['mass3']:.4f}, JS {result['metrics']['js']:.4f}.",
        "",
        "The fitted model uses standardized features:",
        "",
        "```text",
        "score = intercept + sum_j beta_j * standardized(feature_j)",
        "```",
        "",
        f"Standardized intercept: `{result['intercept_standardized']:.4f}`",
        "",
        "| Feature | Standardized coef | Raw-space coef | Train median | Train mean | Train sd |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in result["feature_rows"]:
        lines.append(
            f"| `{r['feature']}` | {r['coef_standardized']:+.4f} | "
            f"{r['coef_raw_nonmissing']:+.4f} | {r['train_median']:.4f} | "
            f"{r['train_mean_after_impute']:.4f} | {r['train_sd_after_impute']:.4f} |"
        )
    lines += [
        "",
        "Raw-space expression for the OL channel, holding other variables fixed:",
        "",
        "```text",
        "d score / d origin_ol ~= -2.6655 + 0.2015 * origin_logfoll + 0.6074 * novelty_global",
        "```",
        "",
        "This is why `origin_ol` should not be read as a standalone KOL ranking. The positive effect is context-dependent and mainly appears through `O_k × visibility` and `O_k × novelty` interactions.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    t0 = time.time()
    panel = build_main_panel()
    params = fit_params(panel)
    beta = params["beta"]
    result = {
        "task": "main_ol_origin_ridge_parameter_export",
        "setting": {
            "threshold": THR,
            "origin_window": ORIGIN_WINDOW["name"],
            "target": TARGET,
            "feature_set": FEATURE_SET,
            "alpha": ALPHA,
        },
        "n_train_rows": params["n_train_rows"],
        "n_val_rows": params["n_val_rows"],
        "n_val_events": params["n_val_events"],
        "metrics": params["metrics"],
        "intercept_standardized": float(beta[0]),
        "intercept_raw_nonmissing": float(params["raw_intercept"]),
        "feature_rows": [
            {
                "feature": f,
                "coef_standardized": float(b),
                "train_median": float(me),
                "train_mean_after_impute": float(m),
                "train_sd_after_impute": float(s),
                "coef_raw_nonmissing": float(rc),
            }
            for f, b, me, m, s, rc in zip(
                params["features"], beta[1:], params["med"], params["mu"], params["sd"], params["raw_coef"]
            )
        ],
        "elapsed_sec": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result)
    print(f"wrote {OUT_JSON.name} and {OUT_MD.name}")


if __name__ == "__main__":
    main()
