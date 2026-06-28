"""Phase28: BERT-family encoder baselines for Experiment 3 origin alert.

Main question:
  At frame origin time, before broad KOL adoption is visible, does a frozen
  text encoder over the origin tweet beat the simple OL-Origin structure?

This keeps Experiment 3 fixed:
  - origin window: first10
  - thresholds: 0.55, 0.60, 0.65
  - no early popularity features

Encoder rows use only the origin tweet text. They do not receive OLtrait,
followers, verification, sentiment scalars, KOL identity, symbol/date, or future
tweets. A ridge readout is fit chronologically, matching earlier encoder
baselines.
"""
from __future__ import annotations

import collections
import json
import pathlib
import time

import numpy as np

import phase5_sentiment_reconstruction as p5
import phase7_origin_alert as p7
import phase21_streaming_agent_encoder_baselines as p21
import phase25_streaming_residual_ol_plugin as p25


OUT = pathlib.Path(__file__).with_name("phase28_origin_alert_encoder_baselines_result.json")

THRESHOLDS = [0.55, 0.60, 0.65]
ORIGIN_WINDOW = {"name": "first10", "max_rank": 10}
ENCODER_SLUGS = ["bert_base", "finbert_encoder", "e5_base", "bge_base"]
INNER_SPLIT = "2021-01-01"
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]


def log(message: str) -> None:
    print(message, flush=True)


def standardize_matrix(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    med = np.nanmedian(X[mask], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X2 = np.where(np.isfinite(X), X, med)
    mu = X2[mask].mean(axis=0)
    sd = X2[mask].std(axis=0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    return (X2 - mu) / sd


def ridge_predict(X: np.ndarray, y: np.ndarray, fit_mask: np.ndarray, standardize_mask: np.ndarray, alpha: float) -> np.ndarray:
    Xs = standardize_matrix(X, standardize_mask)
    y_fit = y[fit_mask]
    y_mean = float(y_fit.mean())
    X_fit = Xs[fit_mask]
    U, singular, Vt = np.linalg.svd(X_fit, full_matrices=False)
    beta = Vt.T @ ((singular / (singular * singular + alpha)) * (U.T @ (y_fit - y_mean)))
    return y_mean + Xs @ beta


def symbol_balanced_ndcg3(rows: list[dict], scores: np.ndarray, target: str, mask: np.ndarray) -> float:
    groups = collections.defaultdict(list)
    for i, row in enumerate(rows):
        if mask[i] and np.isfinite(scores[i]):
            groups[row["event_id"]].append((row["sym"], float(row[target]), float(scores[i])))
    by_symbol = collections.defaultdict(list)
    for values in groups.values():
        if len(values) < 2:
            continue
        y = np.asarray([value[1] for value in values], dtype=float)
        score = np.asarray([value[2] for value in values], dtype=float)
        if not np.isfinite(y).all() or np.nanmax(y) <= 0:
            continue
        k = min(3, len(y))
        order = np.argsort(-score, kind="stable")[:k]
        ideal = np.argsort(-y, kind="stable")[:k]
        ndcg = p5.dcg(y[order]) / max(p5.dcg(y[ideal]), 1e-12)
        by_symbol[values[0][0]].append(float(ndcg))
    return float(np.mean([np.mean(v) for v in by_symbol.values()])) if by_symbol else -np.inf


def fit_matrix_score(rows: list[dict], X: np.ndarray, target: str, method: str, model_selection: dict) -> np.ndarray | None:
    y = np.asarray([row[target] for row in rows], dtype=float)
    train = np.asarray([row["split"] == "train" for row in rows], dtype=bool)
    inner_fit = train & np.asarray([row["day"] < INNER_SPLIT for row in rows], dtype=bool) & np.isfinite(y)
    inner_dev = train & np.asarray([row["day"] >= INNER_SPLIT for row in rows], dtype=bool) & np.isfinite(y)
    all_train = train & np.isfinite(y)
    if inner_fit.sum() < 100 or inner_dev.sum() < 30 or all_train.sum() < 150:
        return None
    alpha_scores = {}
    for alpha in RIDGE_ALPHAS:
        pred = ridge_predict(X, y, inner_fit, inner_fit, alpha)
        alpha_scores[alpha] = symbol_balanced_ndcg3(rows, pred, target, inner_dev)
    best_alpha = max(RIDGE_ALPHAS, key=lambda alpha: (alpha_scores[alpha], alpha))
    model_selection[f"thr{rows[0]['thr']:.2f}_{ORIGIN_WINDOW['name']}:{target}:{method}"] = {
        "selected_alpha": best_alpha,
        "inner_symbol_balanced_ndcg3": alpha_scores,
        "dimension": int(X.shape[1]),
    }
    log(f"  {method:<28} {target}: alpha={best_alpha:g} inner={alpha_scores[best_alpha]:.3f}")
    return ridge_predict(X, y, all_train, all_train, best_alpha)


def origin_text_matrix(rows: list[dict], cache: dict[str, np.ndarray]) -> np.ndarray:
    dim = len(next(iter(cache.values())))
    vectors = []
    for row in rows:
        text = p21.clean_text(row.get("origin_text", ""))
        vectors.append(cache.get(text, np.zeros(dim, dtype=np.float32)))
    return np.vstack(vectors).astype(np.float64)


def scalar_matrix(rows: list[dict], features: list[str]) -> np.ndarray:
    return np.asarray([[row.get(feature, np.nan) for feature in features] for row in rows], dtype=np.float64)


def evaluate_scores(rows: list[dict], scores_by_method: dict[str, np.ndarray], target: str) -> dict:
    evs = {name: p7.event_rows(rows, scores, target) for name, scores in scores_by_method.items()}
    means = {}
    for name, ee in evs.items():
        means[name] = {
            "global_top10": p7.evaluate_global(rows, scores_by_method[name], target),
            "n_events": len(ee),
            "n_symbols": len(set(e["sym"] for e in ee)),
            "pooled": {metric: p7.pooled_mean(ee, metric) for metric in p7.METRICS},
            "symbol_balanced": {metric: p7.symbal_mean(ee, metric) for metric in p7.METRICS},
        }
    comparisons = {}
    for base in [*p7.FEATURE_SETS.keys(), *[f"{slug}_origin_text" for slug in ENCODER_SLUGS]]:
        if "ol_origin" not in evs or base not in evs or base == "ol_origin":
            continue
        pairs = p7.aligned_pairs(evs["ol_origin"], evs[base])
        comparisons[f"ol_origin_vs_{base}"] = {
            metric: {
                "pooled_bootstrap": p7.bootstrap_pooled(pairs, metric),
                "symbol_balanced_bootstrap": p7.bootstrap_symbal(pairs, metric),
            }
            for metric in p7.METRICS
        }
    return {"means": means, "comparisons": comparisons}


def aggregate_main_region(by_setting: dict) -> dict:
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for setting in by_setting.values():
        for target, out in setting.get("targets", {}).items():
            for method, metrics in out.get("means", {}).items():
                for metric in ["ndcg3", "hit1", "mass3", "js"]:
                    value = metrics.get("symbol_balanced", {}).get(metric)
                    if value is not None and np.isfinite(value):
                        acc[method][target][metric].append(float(value))
    return {
        method: {
            target: {
                metric: float(np.mean(values)) if values else None
                for metric, values in metric_map.items()
            }
            for target, metric_map in target_map.items()
        }
        for method, target_map in acc.items()
    }


def main() -> None:
    started = time.time()
    log("[1/6] Loading tweets and embeddings")
    rows_by = {}
    emb_by = {}
    all_rows = []
    for i, sym in enumerate(p5.SYMS, 1):
        rows, emb = p5.load_symbol(sym)
        rows_by[sym] = rows
        emb_by[sym] = emb
        all_rows.extend(rows)
        log(f"  {i:02d}/{len(p5.SYMS)} {sym:<5} rows={len(rows):>6}")
    metadata = p5.compute_metadata(all_rows)
    ol = p5.compute_oltrait(all_rows, metadata)

    log("[2/6] Building origin-alert panels")
    panels = {}
    event_counts = {}
    unique_texts = set()
    for threshold in THRESHOLDS:
        key = f"thr{threshold:.2f}_{ORIGIN_WINDOW['name']}"
        hist = p7.compute_origin_history(rows_by, emb_by, threshold)
        rows, events = p7.build_origin_panel(rows_by, emb_by, metadata, ol, hist, threshold, ORIGIN_WINDOW)
        panels[key] = rows
        event_counts[key] = len(events)
        unique_texts.update(p21.clean_text(row.get("origin_text", "")) for row in rows if p21.clean_text(row.get("origin_text", "")))

    log(f"[3/6] Encoding origin tweet texts: n={len(unique_texts)}")
    caches = {
        slug: p21.encode_texts(slug, p21.MODEL_CONFIGS[slug], sorted(unique_texts))
        for slug in ENCODER_SLUGS
    }

    result = {
        "task": "origin_alert_encoder_baselines",
        "origin_window": ORIGIN_WINDOW,
        "thresholds": THRESHOLDS,
        "targets": p7.TARGETS,
        "metrics": p7.METRICS,
        "encoder_models": {slug: p21.MODEL_CONFIGS[slug] for slug in ENCODER_SLUGS},
        "input_boundary": "origin tweet text only for encoder rows; no OL/KOL metadata/future text",
        "linear_feature_sets": p7.FEATURE_SETS,
        "model_selection": {},
        "by_setting": {},
    }

    log("[4/6] Evaluating linear and encoder rows")
    for key, rows in panels.items():
        log(f"\n--- {key} ---")
        matrices = {f"{slug}_origin_text": origin_text_matrix(rows, caches[slug]) for slug in ENCODER_SLUGS}
        out = {
            "n_rows": len(rows),
            "n_train_rows": sum(row["split"] == "train" for row in rows),
            "n_val_rows": sum(row["split"] == "val" for row in rows),
            "n_events": event_counts[key],
            "targets": {},
        }
        for target in p7.TARGETS:
            scores = p7.train_scores(rows, target)
            for method, X in matrices.items():
                score = fit_matrix_score(rows, X, target, method, result["model_selection"])
                if score is not None:
                    scores[method] = score
            out["targets"][target] = evaluate_scores(rows, scores, target)
        result["by_setting"][key] = out

    log("[5/6] Aggregating main region")
    result["main_region_mean"] = aggregate_main_region(result["by_setting"])
    result["elapsed_sec"] = time.time() - started
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"[6/6] wrote {OUT}")
    log(f"elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
