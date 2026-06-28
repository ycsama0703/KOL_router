"""Phase25: residual OL-timing plug-in for streaming agent triage.

Phase24 showed that naive feature concatenation can hurt strong text encoders.
This experiment treats OL-timing memory as a residual calibration layer:

  text_score = frozen_encoder(current_tweet)
  residual_score = linear_ol_component(prefix_memory)
  final_score = text_score + lambda * residual_score

All hyperparameters are selected on the chronological inner split, then refit on
the full training split and evaluated on validation.
"""
from __future__ import annotations

import collections
import json
import pathlib
import time

import numpy as np

import phase5_sentiment_reconstruction as p5
import phase7_origin_alert as p7
import phase19_streaming_agent_memory as p19
import phase21_streaming_agent_encoder_baselines as p21
import phase22_streaming_ol_timing_roles as p22


OUT = pathlib.Path(__file__).with_name("phase25_streaming_residual_ol_plugin_result.json")

THRESHOLDS = [0.55, 0.60, 0.65]
PREFIX_MIN_PRIOR = 9
BACKBONES = ["minilm", "bert_base", "finbert_encoder", "e5_base", "bge_base"]
COMPONENTS = {
    "no_ol": p22.BASE_NO_OL,
    "old_ol": p22.FEATURE_SETS["old_ol_memory"],
    "ol_timing": p22.FEATURE_SETS["ol_timing_roles"],
}
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
LAMBDAS = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50, -0.05, -0.10, -0.20]
INNER_SPLIT = "2021-01-01"


def log(message: str) -> None:
    print(message, flush=True)


def scalar_matrix(rows: list[dict], features: list[str]) -> np.ndarray:
    return np.asarray([[row.get(feature, np.nan) for feature in features] for row in rows], dtype=np.float64)


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


def fit_text_score(rows: list[dict], X_text: np.ndarray, target: str, selection: dict, method: str) -> np.ndarray:
    y = np.asarray([row[target] for row in rows], dtype=float)
    train = np.asarray([row["split"] == "train" for row in rows], dtype=bool)
    inner_fit = train & np.asarray([row["day"] < INNER_SPLIT for row in rows], dtype=bool) & np.isfinite(y)
    inner_dev = train & np.asarray([row["day"] >= INNER_SPLIT for row in rows], dtype=bool) & np.isfinite(y)
    all_train = train & np.isfinite(y)
    scores = {}
    for alpha in RIDGE_ALPHAS:
        pred = ridge_predict(X_text, y, inner_fit, inner_fit, alpha)
        scores[alpha] = symbol_balanced_ndcg3(rows, pred, target, inner_dev)
    best_alpha = max(RIDGE_ALPHAS, key=lambda alpha: (scores[alpha], alpha))
    selection[f"{method}:{target}:text"] = {
        "selected_alpha": best_alpha,
        "inner_symbol_balanced_ndcg3": scores,
    }
    return ridge_predict(X_text, y, all_train, all_train, best_alpha)


def fit_residual_score(rows: list[dict], X_text: np.ndarray, X_component: np.ndarray, target: str, selection: dict, method: str) -> np.ndarray | None:
    y = np.asarray([row[target] for row in rows], dtype=float)
    train = np.asarray([row["split"] == "train" for row in rows], dtype=bool)
    inner_fit = train & np.asarray([row["day"] < INNER_SPLIT for row in rows], dtype=bool) & np.isfinite(y)
    inner_dev = train & np.asarray([row["day"] >= INNER_SPLIT for row in rows], dtype=bool) & np.isfinite(y)
    all_train = train & np.isfinite(y)
    if inner_fit.sum() < 100 or inner_dev.sum() < 30:
        return None

    base_inner = fit_text_score(rows, X_text, target, selection, f"{method}:inner_base")
    residual_inner_target = y - base_inner

    best = None
    for alpha in RIDGE_ALPHAS:
        residual_pred = ridge_predict(X_component, residual_inner_target, inner_fit, inner_fit, alpha)
        for lam in LAMBDAS:
            score = base_inner + lam * residual_pred
            ndcg = symbol_balanced_ndcg3(rows, score, target, inner_dev)
            candidate = (ndcg, alpha, lam)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        return None
    _best_ndcg, best_alpha, best_lambda = best

    base_final = fit_text_score(rows, X_text, target, selection, f"{method}:final_base")
    residual_final_target = y - base_final
    residual_final = ridge_predict(X_component, residual_final_target, all_train, all_train, best_alpha)
    final_score = base_final + best_lambda * residual_final
    selection[f"{method}:{target}:residual"] = {
        "selected_alpha": best_alpha,
        "selected_lambda": best_lambda,
        "inner_symbol_balanced_ndcg3": _best_ndcg,
        "component_dim": int(X_component.shape[1]),
    }
    log(f"  {method:<30} {target}: alpha={best_alpha:g} lambda={best_lambda:+.2f} inner={_best_ndcg:.3f}")
    return final_score


def build_backbone_matrices(rows: list[dict], emb_by: dict[str, np.ndarray], caches: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    matrices = {"minilm": p21.minilm_matrix(rows, emb_by)}
    for slug, cache in caches.items():
        matrices[slug] = p21.cached_encoder_matrix(rows, cache)
    return matrices


def evaluate_scores(rows: list[dict], scores_by_method: dict[str, np.ndarray], target: str) -> dict:
    evs = {name: p7.event_rows(rows, scores, target) for name, scores in scores_by_method.items()}
    means = {}
    for name, ee in evs.items():
        means[name] = {
            "global_top10": p7.evaluate_global(rows, scores_by_method[name], target),
            "n_events": len(ee),
            "n_symbols": len(set(e["sym"] for e in ee)),
            "pooled": {metric: p7.pooled_mean(ee, metric) for metric in p19.METRICS},
            "symbol_balanced": {metric: p7.symbal_mean(ee, metric) for metric in p19.METRICS},
        }
    comparisons = {}
    for backbone in BACKBONES:
        pairs_to_compare = [
            (f"{backbone}__resid_ol_timing", f"{backbone}__text"),
            (f"{backbone}__resid_ol_timing", f"{backbone}__resid_no_ol"),
            (f"{backbone}__resid_ol_timing", f"{backbone}__resid_old_ol"),
            (f"{backbone}__resid_no_ol", f"{backbone}__text"),
        ]
        for left, right in pairs_to_compare:
            if left not in evs or right not in evs:
                continue
            pairs = p7.aligned_pairs(evs[left], evs[right])
            comparisons[f"{left}_vs_{right}"] = {
                metric: {
                    "pooled_bootstrap": p7.bootstrap_pooled(pairs, metric),
                    "symbol_balanced_bootstrap": p7.bootstrap_symbal(pairs, metric),
                }
                for metric in p19.METRICS
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


def aggregate_delta_summary(by_setting: dict) -> dict:
    labels = [
        ("resid_ol_timing_vs_text", "__resid_ol_timing_vs_", "__text"),
        ("resid_ol_timing_vs_resid_no_ol", "__resid_ol_timing_vs_", "__resid_no_ol"),
        ("resid_ol_timing_vs_resid_old_ol", "__resid_ol_timing_vs_", "__resid_old_ol"),
    ]
    summary = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for setting in by_setting.values():
        for target, out in setting.get("targets", {}).items():
            comparisons = out.get("comparisons", {})
            for backbone in BACKBONES:
                for label, mid, suffix in labels:
                    name = f"{backbone}{mid}{backbone}{suffix}"
                    comp = comparisons.get(name)
                    if not comp:
                        continue
                    for metric in ["ndcg3", "hit1"]:
                        observed = comp[metric]["symbol_balanced_bootstrap"].get("observed")
                        if observed is not None and np.isfinite(observed):
                            summary[label][target][metric].append(float(observed))
    return {
        label: {
            target: {
                metric: {
                    "mean_delta": float(np.mean(values)) if values else None,
                    "n": len(values),
                    "positive_count": int(sum(1 for value in values if value > 0)),
                }
                for metric, values in metric_map.items()
            }
            for target, metric_map in target_map.items()
        }
        for label, target_map in summary.items()
    }


def main() -> None:
    started = time.time()
    log("[1/7] Loading tweets and MiniLM embeddings")
    rows_by = {}
    emb_by = {}
    all_rows = []
    for i, sym in enumerate(p19.SYMS, 1):
        rows, emb = p5.load_symbol(sym)
        rows_by[sym] = rows
        emb_by[sym] = emb
        all_rows.extend(rows)
        log(f"  {i:02d}/{len(p19.SYMS)} {sym:<5} rows={len(rows):>6}")
    metadata = p5.compute_metadata(all_rows)
    ol = p5.compute_oltrait(all_rows, metadata)

    log("[2/7] Building timing-role panels")
    panels = {}
    event_counts = {}
    unique_texts = set()
    for threshold in [0.55, 0.60, 0.65]:
        key = f"thr{threshold:.2f}_prefix9"
        hist = p19.compute_origin_history(rows_by, emb_by, threshold)
        rows, events = p22.build_timing_panel(rows_by, emb_by, metadata, ol, hist, threshold)
        panels[key] = rows
        event_counts[key] = len(events)
        unique_texts.update(p21.clean_text(row.get("current_text", "")) for row in rows if p21.clean_text(row.get("current_text", "")))

    log(f"[3/7] Loading/encoding current tweet texts: n={len(unique_texts)}")
    caches = {
        slug: p21.encode_texts(slug, config, sorted(unique_texts))
        for slug, config in p21.MODEL_CONFIGS.items()
    }

    result = {
        "task": "streaming_residual_ol_plugin",
        "thresholds": [0.55, 0.60, 0.65],
        "prefix_min_prior": 9,
        "backbones": BACKBONES,
        "components": list(COMPONENTS),
        "targets": p19.TARGETS,
        "metrics": p19.METRICS,
        "ridge_alphas": RIDGE_ALPHAS,
        "lambdas": LAMBDAS,
        "model_selection": {},
        "by_setting": {},
    }

    log("[4/7] Evaluating residual plug-ins")
    for key, rows in panels.items():
        log(f"\n--- {key} ---")
        backbone_matrices = build_backbone_matrices(rows, emb_by, caches)
        component_matrices = {name: scalar_matrix(rows, features) for name, features in COMPONENTS.items()}
        out = {
            "n_rows": len(rows),
            "n_train_rows": sum(row["split"] == "train" for row in rows),
            "n_val_rows": sum(row["split"] == "val" for row in rows),
            "n_events": event_counts[key],
            "targets": {},
        }
        for target in p19.TARGETS:
            scores = {}
            for backbone, X_text in backbone_matrices.items():
                text_method = f"{backbone}__text"
                scores[text_method] = fit_text_score(rows, X_text, target, result["model_selection"], f"{key}:{text_method}")
                for component, X_component in component_matrices.items():
                    method = f"{backbone}__resid_{component}"
                    score = fit_residual_score(rows, X_text, X_component, target, result["model_selection"], f"{key}:{method}")
                    if score is not None:
                        scores[method] = score
            out["targets"][target] = evaluate_scores(rows, scores, target)
            for backbone in ["bert_base", "e5_base", "finbert_encoder"]:
                comp_name = f"{backbone}__resid_ol_timing_vs_{backbone}__text"
                comp = out["targets"][target].get("comparisons", {}).get(comp_name)
                if comp:
                    nd = comp["ndcg3"]["symbol_balanced_bootstrap"]
                    log(
                        f"  {target:<16} {backbone:<15} residual OL-vs-text "
                        f"NDCG={nd.get('observed'):+.3f} "
                        f"CI[{nd.get('ci05'):+.3f},{nd.get('ci95'):+.3f}]"
                    )
        result["by_setting"][key] = out

    log("[6/7] Aggregating main region")
    result["main_region_mean"] = aggregate_main_region(result["by_setting"])
    result["delta_summary"] = aggregate_delta_summary(result["by_setting"])
    result["elapsed_sec"] = time.time() - started
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"[7/7] wrote {OUT}")
    log(f"elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
