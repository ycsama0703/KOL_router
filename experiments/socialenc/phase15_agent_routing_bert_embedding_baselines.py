"""Experiment 1 frozen BERT-family frame-embedding baselines.

Each encoder embeds only texts visible in the first10 frame. Per-text vectors
are pooled into a normalized frame vector, then a target-specific Ridge model
is trained with the same chronological inner alpha selection as MiniLM-Ridge.

The encoders remain frozen. No sentiment logits, KOL metadata, OLtrait, early
popularity counts, or future text are used as model inputs.
"""
from __future__ import annotations

import collections
import json
import pathlib
import re
import time

import numpy as np

import phase5_sentiment_reconstruction as p5
import phase9_agent_routing as p9


OUT = pathlib.Path(__file__).with_name("phase15_agent_routing_bert_embedding_baselines_result.json")
CACHE_DIR = pathlib.Path(__file__).with_name("phase15_embedding_cache")
INNER_SPLIT = "2021-01-01"
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
BATCH_SIZE = 64

MODEL_CONFIGS = {
    "bert_base": {
        "model": "bert-base-uncased",
        "prefix": "",
        "pooling": "mean",
    },
    "finbert_encoder": {
        "model": "ProsusAI/finbert",
        "prefix": "",
        "pooling": "mean",
    },
    "e5_base": {
        "model": "intfloat/e5-base-v2",
        "prefix": "passage: ",
        "pooling": "mean",
    },
    "bge_base": {
        "model": "BAAI/bge-base-en-v1.5",
        "prefix": "",
        "pooling": "cls",
    },
}

p5.THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65]
p5.EARLY_MODES = [{"name": "first10", "kind": "count", "value": 10}]

p9.OUT = OUT
p9.RNG = np.random.default_rng(1515)

OL_ORIGIN_FEATURES = [
    "origin_ol", "origin_logfoll", "origin_verified", "origin_rank_frac",
    "novelty", "cohesion", "origin_stance", "frame_stance_mean",
    "frame_stance_abs", "frame_bull_share", "frame_bear_share",
    "follower_weighted_stance", "event_stance_mean", "event_stance_abs",
    "event_sentiment_disagreement", "ol_x_visibility", "ol_x_novelty",
    "log_event_early_kols", "cutoff_elapsed_hours",
]

EMBEDDING_METHODS = {
    "minilm_text_ridge": "frame_embedding",
    **{f"{slug}_ridge": f"{slug}_embedding" for slug in MODEL_CONFIGS},
}

p9.RAW_ROUTERS = []
p9.LEARNED_ROUTERS = {
    "learned_semantic_only": p5.FEATURE_SETS["semantic"],
    **{name: [field] for name, field in EMBEDDING_METHODS.items()},
    "ol_origin_router": OL_ORIGIN_FEATURES,
}

p9.COMPARISONS = [
    *[("ol_origin_router", name) for name in EMBEDDING_METHODS],
    ("bert_base_ridge", "minilm_text_ridge"),
    ("finbert_encoder_ridge", "bert_base_ridge"),
    ("e5_base_ridge", "minilm_text_ridge"),
    ("bge_base_ridge", "minilm_text_ridge"),
]

SELECTED_ALPHAS: dict[str, dict] = {}
MATRIX_CACHE: dict[str, dict] = {}
MATRIX_CACHE_SETTING: str | None = None


def log(message: str) -> None:
    print(message, flush=True)


def clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text or "")
    return re.sub(r"\s+", " ", text).strip()[:1000]


def cache_path(slug: str) -> pathlib.Path:
    return CACHE_DIR / f"{slug}.npz"


def load_embedding_cache(slug: str):
    path = cache_path(slug)
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=False)
    return {str(text): emb.astype(np.float32) for text, emb in zip(data["texts"], data["embeddings"])}


def save_embedding_cache(slug: str, cache) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    texts = sorted(cache)
    embeddings = np.vstack([cache[text] for text in texts]).astype(np.float32)
    np.savez_compressed(cache_path(slug), texts=np.asarray(texts), embeddings=embeddings)


def encode_texts(slug: str, config: dict, texts: list[str]):
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
        batch_texts = [config["prefix"] + text for text in missing[start:start + BATCH_SIZE]]
        encoded = tokenizer(
            batch_texts,
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


def add_frame_embeddings(panels: dict[str, list[dict]]) -> dict[str, dict[str, np.ndarray]]:
    texts = sorted({
        clean_text(text)
        for rows in panels.values()
        for row in rows
        for text in row.get("frame_all_texts", row.get("frame_texts", []))
        if clean_text(text)
    })
    log(f"[4/7] Encoding unique point-in-time texts: n={len(texts)}")
    caches = {slug: encode_texts(slug, config, texts) for slug, config in MODEL_CONFIGS.items()}
    for rows in panels.values():
        for row in rows:
            frame_texts = [
                clean_text(text)
                for text in row.get("frame_all_texts", row.get("frame_texts", []))
                if clean_text(text)
            ]
            for slug, cache in caches.items():
                vectors = [cache[text] for text in frame_texts if text in cache]
                vector = np.mean(vectors, axis=0) if vectors else np.zeros(next(iter(cache.values())).shape)
                vector = vector / (np.linalg.norm(vector) + 1e-12)
                row[f"{slug}_embedding"] = vector.astype(np.float32)
    return caches


def standardize(X: np.ndarray, mask: np.ndarray):
    med = np.nanmedian(X[mask], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X2 = np.where(np.isfinite(X), X, med)
    mu = X2[mask].mean(axis=0)
    sd = X2[mask].std(axis=0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    return (X2 - mu) / sd


def matrix_bundle(rows, field):
    global MATRIX_CACHE_SETTING
    setting = f"thr{rows[0]['thr']:.2f}_{rows[0]['mode']}"
    if MATRIX_CACHE_SETTING != setting:
        MATRIX_CACHE.clear()
        MATRIX_CACHE_SETTING = setting
    if field in MATRIX_CACHE:
        return MATRIX_CACHE[field]
    X = np.vstack([np.asarray(row[field], dtype=np.float64) for row in rows])
    train = np.asarray([row["split"] == "train" for row in rows], dtype=bool)
    inner_fit = train & np.asarray([row["day"] < INNER_SPLIT for row in rows])
    inner_dev = train & np.asarray([row["day"] >= INNER_SPLIT for row in rows])
    X_inner = standardize(X, inner_fit)
    X_final = standardize(X, train)
    U_inner, singular_inner, Vt_inner = np.linalg.svd(X_inner[inner_fit], full_matrices=False)
    U_final, singular_final, Vt_final = np.linalg.svd(X_final[train], full_matrices=False)
    bundle = {
        "X": X,
        "train": train,
        "inner_fit": inner_fit,
        "inner_dev": inner_dev,
        "X_inner": X_inner,
        "X_final": X_final,
        "inner_decomp": (U_inner, singular_inner, Vt_inner),
        "final_decomp": (U_final, singular_final, Vt_final),
    }
    MATRIX_CACHE[field] = bundle
    return bundle


def ridge_scores_from_decomp(Xs, y, fit_mask, decomposition, alpha):
    U, singular, Vt = decomposition
    ytr = y[fit_mask]
    y_mean = float(ytr.mean())
    beta = Vt.T @ ((singular / (singular * singular + alpha)) * (U.T @ (ytr - y_mean)))
    return y_mean + Xs @ beta


def symbol_balanced_ndcg3(rows, scores, target, mask):
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
        if not np.isfinite(y).all() or y.sum() <= 0:
            continue
        k = min(3, len(y))
        order = np.argsort(-score, kind="stable")[:k]
        ideal = np.argsort(-y, kind="stable")[:k]
        ndcg = p9.dcg(y[order]) / max(p9.dcg(y[ideal]), 1e-12)
        by_symbol[values[0][0]].append(float(ndcg))
    return float(np.mean([np.mean(v) for v in by_symbol.values()])) if by_symbol else -np.inf


def fit_embedding_scores(rows, train_target, method, field):
    bundle = matrix_bundle(rows, field)
    X = bundle["X"]
    y = np.asarray([row[train_target] for row in rows], dtype=float)
    inner_fit = bundle["inner_fit"] & np.isfinite(y)
    inner_dev = bundle["inner_dev"] & np.isfinite(y)
    all_train = bundle["train"] & np.isfinite(y)
    if inner_fit.sum() < 100 or inner_dev.sum() < 30:
        return None
    eval_target = "future_adopt" if train_target == "log_future_adopt" else "future_reach"
    X_inner = bundle["X_inner"]
    alpha_scores = {}
    for alpha in RIDGE_ALPHAS:
        score = ridge_scores_from_decomp(X_inner, y, inner_fit, bundle["inner_decomp"], alpha)
        alpha_scores[alpha] = symbol_balanced_ndcg3(rows, score, eval_target, inner_dev)
    best_alpha = max(RIDGE_ALPHAS, key=lambda alpha: (alpha_scores[alpha], alpha))
    X_final = bundle["X_final"]
    score = ridge_scores_from_decomp(X_final, y, all_train, bundle["final_decomp"], best_alpha)
    setting = f"thr{rows[0]['thr']:.2f}_{rows[0]['mode']}"
    SELECTED_ALPHAS[f"{setting}:{train_target}:{method}"] = {
        "selected_alpha": best_alpha,
        "inner_symbol_balanced_ndcg@3": alpha_scores,
        "dimension": int(X.shape[1]),
    }
    log(f"  {method} {train_target}: alpha={best_alpha:g} inner={alpha_scores[best_alpha]:.3f}")
    return score


def score_all_routers(rows, train_target):
    scores = {}
    for name, features in p9.LEARNED_ROUTERS.items():
        if name in EMBEDDING_METHODS:
            score = fit_embedding_scores(rows, train_target, name, EMBEDDING_METHODS[name])
        else:
            score = p5.fit_ridge_scores(rows, features, train_target)
        if score is not None:
            scores[name] = score
    return scores


p9.score_all_routers = score_all_routers


def main() -> None:
    started = time.time()
    log("[1/7] Loading tweets and MiniLM embeddings")
    rows_by = {}
    emb_by = {}
    all_rows = []
    for i, symbol in enumerate(p5.SYMS, 1):
        rows, embeddings = p5.load_symbol(symbol)
        rows_by[symbol] = rows
        emb_by[symbol] = embeddings
        all_rows.extend(rows)
        log(f"  {i:02d}/{len(p5.SYMS)} {symbol:<5} rows={len(rows):>6}")
    metadata = p5.compute_metadata(all_rows)
    ol = p5.compute_oltrait(all_rows, metadata)

    log("[3/7] Building all point-in-time panels")
    panels = {}
    event_counts = {}
    for threshold in p5.THRESHOLDS:
        mode = p5.EARLY_MODES[0]
        key = f"thr{threshold:.2f}_{mode['name']}"
        log(f"\n--- {key} ---")
        rows, events = p5.build_panel_for(rows_by, emb_by, metadata, ol, threshold, mode)
        panels[key] = rows
        event_counts[key] = len(events)

    caches = add_frame_embeddings(panels)
    result = {
        "task": "agent_attention_bert_embedding_routing",
        "bootstrap_B": p9.B,
        "thresholds": p5.THRESHOLDS,
        "early_modes": p5.EARLY_MODES,
        "ks": p9.KS,
        "learned_routers": p9.LEARNED_ROUTERS,
        "comparisons": p9.COMPARISONS,
        "targets": p9.TARGETS,
        "n_ol_kols": len(ol),
        "embedding_models": MODEL_CONFIGS,
        "by_setting": {},
    }

    log("[5/7] Evaluating routers")
    for key, rows in panels.items():
        output = {
            "n_rows": len(rows),
            "n_train_rows": sum(row["split"] == "train" for row in rows),
            "n_val_rows": sum(row["split"] == "val" for row in rows),
            "n_events": event_counts[key],
            "targets": {},
        }
        for label, train_target in p9.TARGETS.items():
            eval_target = "future_adopt" if label == "future_adopt" else "future_reach"
            output["targets"][label] = p9.evaluate_setting(rows, train_target, eval_target)
        result["by_setting"][key] = output

    result["model_selection"] = SELECTED_ALPHAS
    result["cache_entries"] = {slug: len(cache) for slug, cache in caches.items()}
    result["elapsed_sec"] = time.time() - started
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"[7/7] wrote {OUT}")
    log(f"elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
