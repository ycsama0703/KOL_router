"""Streaming agent-memory frozen encoder baselines.

This complements phase20 local-LLM baselines for the replacement Experiment 1.
Each frozen encoder embeds the current tweet only. A chronological ridge scorer
is trained either on:

  - current tweet embedding only; or
  - current tweet embedding + no-OL structured prefix-memory features.

No OLtrait, KOL identity, symbol/date, or future tweets are given to the encoder
baselines. OL-Memory from phase19 is included unchanged for comparison.
"""
from __future__ import annotations

import collections
import json
import pathlib
import re
import time

import numpy as np

import phase5_sentiment_reconstruction as p5
import phase7_origin_alert as p7
import phase15_agent_routing_bert_embedding_baselines as p15
import phase19_streaming_agent_memory as p19


OUT = pathlib.Path(__file__).with_name("phase21_streaming_agent_encoder_baselines_result.json")
CACHE_DIR = pathlib.Path(__file__).with_name("phase15_embedding_cache")

MAIN_THRESHOLDS = [0.55, 0.60, 0.65]
PREFIX_MIN_PRIOR = 9
INNER_SPLIT = "2021-01-01"
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]

MODEL_CONFIGS = p15.MODEL_CONFIGS
BATCH_SIZE = 64

NO_OL_MEMORY_SCALARS = (
    p19.CURRENT_KOL
    + p19.CURRENT_SENTIMENT
    + p19.HISTORY
    + p19.SEMANTIC_MEMORY
)

ENCODER_METHODS = [
    "minilm_current_ridge",
    "minilm_memory_ridge",
    *[f"{slug}_current_ridge" for slug in MODEL_CONFIGS],
    *[f"{slug}_memory_ridge" for slug in MODEL_CONFIGS],
]


def log(message: str) -> None:
    print(message, flush=True)


def clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text or "")
    return re.sub(r"\s+", " ", text).strip()[:1000]


def cache_path(slug: str) -> pathlib.Path:
    return CACHE_DIR / f"{slug}.npz"


def load_embedding_cache(slug: str) -> dict[str, np.ndarray]:
    path = cache_path(slug)
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=False)
    return {
        str(text): emb.astype(np.float32)
        for text, emb in zip(data["texts"], data["embeddings"])
    }


def save_embedding_cache(slug: str, cache: dict[str, np.ndarray]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    texts = sorted(cache)
    embeddings = np.vstack([cache[text] for text in texts]).astype(np.float32)
    np.savez_compressed(cache_path(slug), texts=np.asarray(texts), embeddings=embeddings)


def encode_texts(slug: str, config: dict, texts: list[str]) -> dict[str, np.ndarray]:
    cache = load_embedding_cache(slug)
    missing = [text for text in texts if text not in cache]
    log(f"  {slug}: cached={len(cache)} missing={len(missing)}")
    if not missing:
        return cache

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    model = AutoModel.from_pretrained(config["model"]).eval().to(device)
    if device == "cuda":
        model.half()
    for start in range(0, len(missing), BATCH_SIZE):
        batch = [config["prefix"] + text for text in missing[start:start + BATCH_SIZE]]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state.float()
            if config["pooling"] == "cls":
                pooled = hidden[:, 0]
            else:
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()
        for text, vector in zip(missing[start:start + BATCH_SIZE], pooled):
            cache[text] = vector.astype(np.float32)
        done = min(start + BATCH_SIZE, len(missing))
        if done % (BATCH_SIZE * 10) == 0 or done == len(missing):
            log(f"  {slug}: encoded {done}/{len(missing)}")
    save_embedding_cache(slug, cache)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return cache


def standardize_matrix(X: np.ndarray, mask: np.ndarray):
    med = np.nanmedian(X[mask], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X2 = np.where(np.isfinite(X), X, med)
    mu = X2[mask].mean(axis=0)
    sd = X2[mask].std(axis=0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    return (X2 - mu) / sd


def ridge_scores_from_svd(Xs: np.ndarray, y: np.ndarray, fit_mask: np.ndarray, alpha: float) -> np.ndarray:
    ytr = y[fit_mask]
    y_mean = float(ytr.mean())
    Xtr = Xs[fit_mask]
    U, singular, Vt = np.linalg.svd(Xtr, full_matrices=False)
    beta = Vt.T @ ((singular / (singular * singular + alpha)) * (U.T @ (ytr - y_mean)))
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


def fit_matrix_scores(rows: list[dict], X: np.ndarray, target: str, method: str, model_selection: dict) -> np.ndarray | None:
    y = np.asarray([row[target] for row in rows], dtype=float)
    train = np.asarray([row["split"] == "train" for row in rows], dtype=bool)
    inner_fit = train & np.asarray([row["day"] < INNER_SPLIT for row in rows], dtype=bool) & np.isfinite(y)
    inner_dev = train & np.asarray([row["day"] >= INNER_SPLIT for row in rows], dtype=bool) & np.isfinite(y)
    all_train = train & np.isfinite(y)
    if inner_fit.sum() < 100 or inner_dev.sum() < 30 or all_train.sum() < 150:
        return None

    X_inner = standardize_matrix(X, inner_fit)
    alpha_scores = {}
    for alpha in RIDGE_ALPHAS:
        scores = ridge_scores_from_svd(X_inner, y, inner_fit, alpha)
        alpha_scores[alpha] = symbol_balanced_ndcg3(rows, scores, target, inner_dev)
    best_alpha = max(RIDGE_ALPHAS, key=lambda alpha: (alpha_scores[alpha], alpha))

    X_final = standardize_matrix(X, all_train)
    scores = ridge_scores_from_svd(X_final, y, all_train, best_alpha)
    setting = f"thr{rows[0]['thr']:.2f}_prefix{rows[0]['min_prior']}"
    model_selection[f"{setting}:{target}:{method}"] = {
        "selected_alpha": best_alpha,
        "inner_symbol_balanced_ndcg3": alpha_scores,
        "dimension": int(X.shape[1]),
    }
    log(f"  {method:<28} {target}: alpha={best_alpha:g} inner={alpha_scores[best_alpha]:.3f}")
    return scores


def scalar_matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray([[row.get(feature, np.nan) for feature in NO_OL_MEMORY_SCALARS] for row in rows], dtype=np.float64)


def minilm_matrix(rows: list[dict], emb_by: dict[str, np.ndarray]) -> np.ndarray:
    return np.vstack([
        p5.norm_vec(emb_by[row["sym"]], int(row["current_idx"]))
        for row in rows
    ]).astype(np.float64)


def cached_encoder_matrix(rows: list[dict], cache: dict[str, np.ndarray]) -> np.ndarray:
    vectors = []
    dim = len(next(iter(cache.values())))
    for row in rows:
        text = clean_text(row.get("current_text", ""))
        vectors.append(cache.get(text, np.zeros(dim, dtype=np.float32)))
    return np.vstack(vectors).astype(np.float64)


def evaluate_scores(rows: list[dict], scores_by_method: dict[str, np.ndarray], target: str) -> dict:
    evs = {name: p7.event_rows(rows, scores, target) for name, scores in scores_by_method.items()}
    means = {}
    for name, ee in evs.items():
        means[name] = {
            "global_top10": p7.evaluate_global(rows, scores_by_method[name], target),
            "n_events": len(ee),
            "n_symbols": len(set(e["sym"] for e in ee)),
            "pooled": {m: p7.pooled_mean(ee, m) for m in p19.METRICS},
            "symbol_balanced": {m: p7.symbal_mean(ee, m) for m in p19.METRICS},
        }
    comparisons = {}
    for base in ENCODER_METHODS + ["no_ol_memory", "semantic_memory"]:
        if "ol_memory" not in evs or base not in evs:
            continue
        pairs = p7.aligned_pairs(evs["ol_memory"], evs[base])
        comparisons[f"ol_memory_vs_{base}"] = {
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
                    val = metrics.get("symbol_balanced", {}).get(metric)
                    if val is not None and np.isfinite(val):
                        acc[method][target][metric].append(float(val))
    return {
        method: {
            target: {
                metric: float(np.mean(vals)) if vals else None
                for metric, vals in metric_map.items()
            }
            for target, metric_map in target_map.items()
        }
        for method, target_map in acc.items()
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

    log("[2/7] Building streaming prefix9 panels")
    panels = {}
    event_counts = {}
    unique_texts = set()
    for threshold in MAIN_THRESHOLDS:
        key = f"thr{threshold:.2f}_prefix{PREFIX_MIN_PRIOR}"
        hist = p19.compute_origin_history(rows_by, emb_by, threshold)
        rows, events = p19.build_streaming_panel(rows_by, emb_by, metadata, ol, hist, threshold, PREFIX_MIN_PRIOR)
        panels[key] = rows
        event_counts[key] = len(events)
        unique_texts.update(clean_text(row.get("current_text", "")) for row in rows if clean_text(row.get("current_text", "")))

    log(f"[3/7] Encoding current tweet texts: n={len(unique_texts)}")
    caches = {
        slug: encode_texts(slug, config, sorted(unique_texts))
        for slug, config in MODEL_CONFIGS.items()
    }

    result = {
        "task": "streaming_agent_memory_encoder_baselines",
        "thresholds": MAIN_THRESHOLDS,
        "prefix_min_prior": PREFIX_MIN_PRIOR,
        "targets": p19.TARGETS,
        "metrics": p19.METRICS,
        "encoder_models": MODEL_CONFIGS,
        "encoder_methods": ENCODER_METHODS,
        "input_boundary": "current tweet embedding plus optional no-OL structured prefix memory; no OL labels",
        "linear_feature_sets": p19.FEATURE_SETS,
        "n_unique_texts": len(unique_texts),
        "cache_entries": {slug: len(cache) for slug, cache in caches.items()},
        "model_selection": {},
        "by_setting": {},
    }

    log("[4/7] Evaluating encoder baselines")
    for key, rows in panels.items():
        log(f"\n--- {key} ---")
        scalar = scalar_matrix(rows)
        matrix_by_method = {}
        minilm = minilm_matrix(rows, emb_by)
        matrix_by_method["minilm_current_ridge"] = minilm
        matrix_by_method["minilm_memory_ridge"] = np.column_stack([minilm, scalar])
        for slug, cache in caches.items():
            X = cached_encoder_matrix(rows, cache)
            matrix_by_method[f"{slug}_current_ridge"] = X
            matrix_by_method[f"{slug}_memory_ridge"] = np.column_stack([X, scalar])

        out = {
            "n_rows": len(rows),
            "n_train_rows": sum(row["split"] == "train" for row in rows),
            "n_val_rows": sum(row["split"] == "val" for row in rows),
            "n_events": event_counts[key],
            "targets": {},
        }
        for target in p19.TARGETS:
            scores = p19.train_scores(rows, target)
            for method in ENCODER_METHODS:
                score = fit_matrix_scores(rows, matrix_by_method[method], target, method, result["model_selection"])
                if score is not None:
                    scores[method] = score
            out["targets"][target] = evaluate_scores(rows, scores, target)
        result["by_setting"][key] = out

    log("[6/7] Aggregating main region")
    result["main_region_mean"] = aggregate_main_region(result["by_setting"])
    result["elapsed_sec"] = time.time() - started
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"[7/7] wrote {OUT}")
    log(f"elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
