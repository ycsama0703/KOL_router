"""Ablation study for the pre-popularity OL-Origin router.

This is intentionally narrow:
  - semantic threshold = 0.55
  - origin window = first10
  - main target = future follower-weighted Reach

The goal is to test whether the stable originator role is doing real work:
  1) remove OLtrait from the full model;
  2) use OLtrait alone;
  3) shuffle OLtrait across KOLs while preserving its marginal distribution;
  4) replace the OL role channel with follower-scale analogues;
  5) compare residualized OLtrait with raw, hour-confounded OLtrait.
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import time

import numpy as np

import phase5_sentiment_reconstruction as p5
import phase7_origin_alert as p7

OUT = pathlib.Path(__file__).with_name("phase33_origin_alert_ablation_result.json")

THR = 0.55
ORIGIN_WINDOW = {"name": "first10", "max_rank": 10}
TARGET = "log_future_reach"
BOOTSTRAP_B = 600
RNG = np.random.default_rng(33033)

FEATURE_SETS = {
    "no_ol_strong": [
        "origin_logfoll", "origin_verified", "log_origin_rank", "elapsed_hours",
        "prior_frame_count", "origin_stance", "origin_stance_abs",
        "novelty_global", "novelty_event", "hist_log_origin_count",
        "hist_mean_log_adopt", "hist_success_rate",
    ],
    "ol_only": ["origin_ol"],
    "ol_origin_full": [
        "origin_ol", "origin_logfoll", "origin_verified", "log_origin_rank",
        "elapsed_hours", "prior_frame_count", "origin_stance",
        "origin_stance_abs", "novelty_global", "novelty_event",
        "hist_log_origin_count", "hist_mean_log_adopt", "hist_success_rate",
        "ol_x_visibility", "ol_x_novelty",
    ],
    "shuffled_ol_origin": [
        "origin_ol_shuffled", "origin_logfoll", "origin_verified",
        "log_origin_rank", "elapsed_hours", "prior_frame_count",
        "origin_stance", "origin_stance_abs", "novelty_global",
        "novelty_event", "hist_log_origin_count", "hist_mean_log_adopt",
        "hist_success_rate", "shuffled_ol_x_visibility",
        "shuffled_ol_x_novelty",
    ],
    "follower_replacement": [
        "origin_logfoll", "origin_verified", "log_origin_rank",
        "elapsed_hours", "prior_frame_count", "origin_stance",
        "origin_stance_abs", "novelty_global", "novelty_event",
        "hist_log_origin_count", "hist_mean_log_adopt", "hist_success_rate",
        "logfoll_x_visibility", "logfoll_x_novelty",
    ],
    "raw_ol_origin": [
        "origin_ol_raw", "origin_logfoll", "origin_verified", "log_origin_rank",
        "elapsed_hours", "prior_frame_count", "origin_stance",
        "origin_stance_abs", "novelty_global", "novelty_event",
        "hist_log_origin_count", "hist_mean_log_adopt", "hist_success_rate",
        "raw_ol_x_visibility", "raw_ol_x_novelty",
    ],
}

COMPARISONS = [
    ("ol_origin_full", "no_ol_strong"),
    ("ol_origin_full", "ol_only"),
    ("ol_origin_full", "shuffled_ol_origin"),
    ("ol_origin_full", "follower_replacement"),
    ("ol_origin_full", "raw_ol_origin"),
    ("raw_ol_origin", "no_ol_strong"),
    ("shuffled_ol_origin", "no_ol_strong"),
    ("follower_replacement", "no_ol_strong"),
]

METRICS = p7.METRICS


def log(msg: str) -> None:
    print(msg, flush=True)


def compute_raw_oltrait(all_rows, meta):
    ev = p5.first_by_event(all_rows, end=p5.TRAIN_END)
    s = collections.defaultdict(float)
    n = collections.defaultdict(float)
    for d in ev.values():
        if len(d) < 5:
            continue
        parts = sorted(d.items(), key=lambda kv: kv[1]["ts"])
        kk = len(parts)
        for i, (k, _) in enumerate(parts):
            s[k] += kk + 1 - 2 * (i + 1)
            n[k] += 1
    return {k: float(s[k] / n[k]) for k in s if n[k] >= 4 and k in meta}


def make_shuffled_ol(ol):
    keys = sorted(ol)
    vals = np.array([ol[k] for k in keys], dtype=float)
    vals = vals[RNG.permutation(len(vals))]
    return {k: float(v) for k, v in zip(keys, vals)}


def add_ablation_columns(panel, ol_raw, ol_shuffled):
    for r in panel:
        novelty = 0.0 if not np.isfinite(r["novelty_global"]) else float(r["novelty_global"])
        logf = float(r["origin_logfoll"])
        raw = float(ol_raw.get(r["origin_kol"], r["origin_ol"]))
        shuf = float(ol_shuffled.get(r["origin_kol"], 0.0))
        r["origin_ol_raw"] = raw
        r["raw_ol_x_visibility"] = raw * logf
        r["raw_ol_x_novelty"] = raw * novelty
        r["origin_ol_shuffled"] = shuf
        r["shuffled_ol_x_visibility"] = shuf * logf
        r["shuffled_ol_x_novelty"] = shuf * novelty
        r["logfoll_x_visibility"] = logf * logf
        r["logfoll_x_novelty"] = logf * novelty


def train_scores(rows, target):
    out = {}
    for name, feats in FEATURE_SETS.items():
        sc = p5.fit_ridge_scores(rows, feats, target)
        if sc is not None:
            out[name] = sc
    return out


def summarize_events(rows, scores):
    evs = {name: p7.event_rows(rows, sc, TARGET) for name, sc in scores.items()}
    means = {}
    for name, ee in evs.items():
        means[name] = {
            "n_events": len(ee),
            "n_symbols": len(set(e["sym"] for e in ee)),
            "pooled": {m: p7.pooled_mean(ee, m) for m in METRICS},
            "symbol_balanced": {m: p7.symbal_mean(ee, m) for m in METRICS},
            "global_top10": p7.evaluate_global(rows, scores[name], TARGET),
        }
    return evs, means


def delta(a, b, metric):
    return b[metric] - a[metric] if metric == "js" else a[metric] - b[metric]


def summarize(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"observed": None, "ci05": None, "ci95": None}
    return {
        "observed": float(np.mean(vals)),
        "ci05": float(np.quantile(vals, 0.05)),
        "ci95": float(np.quantile(vals, 0.95)),
    }


def bootstrap_symbol_balanced(pairs, metric):
    by = collections.defaultdict(list)
    for a, b in pairs:
        d = delta(a, b, metric)
        if np.isfinite(d):
            by[a["sym"]].append(d)
    syms = sorted(k for k, v in by.items() if v)
    if not syms:
        return {"n_symbols": 0, "n_events": 0, **summarize([])}
    obs = float(np.mean([np.mean(by[s]) for s in syms]))
    boots = []
    for _ in range(BOOTSTRAP_B):
        ss = RNG.choice(syms, size=len(syms), replace=True)
        vals = []
        for s in ss:
            arr = np.asarray(by[s], dtype=float)
            ii = RNG.integers(0, len(arr), len(arr))
            vals.append(float(arr[ii].mean()))
        boots.append(float(np.mean(vals)))
    out = summarize(boots)
    out["observed"] = obs
    out["n_symbols"] = int(len(syms))
    out["n_events"] = int(sum(len(v) for v in by.values()))
    return out


def aligned_pairs(a, b):
    aa = {e["event_id"]: e for e in a}
    bb = {e["event_id"]: e for e in b}
    return [(aa[k], bb[k]) for k in sorted(set(aa) & set(bb))]


def run_comparisons(evs):
    comps = {}
    for model, base in COMPARISONS:
        if model not in evs or base not in evs:
            continue
        pairs = aligned_pairs(evs[model], evs[base])
        comps[f"{model}_vs_{base}"] = {
            m: {"symbol_balanced_bootstrap": bootstrap_symbol_balanced(pairs, m)}
            for m in METRICS
        }
    return comps


def main():
    t0 = time.time()
    log("[1/5] Loading tweets and embeddings")
    rows_by = {}
    emb_by = {}
    all_rows = []
    for i, sym in enumerate(p5.SYMS, 1):
        rows, emb = p5.load_symbol(sym)
        rows_by[sym] = rows
        emb_by[sym] = emb
        all_rows.extend(rows)
        log(f"  {i:02d}/{len(p5.SYMS)} {sym:<5} rows={len(rows):>6}")

    log("[2/5] Computing KOL metadata and OL traits")
    meta = p5.compute_metadata(all_rows)
    ol = p5.compute_oltrait(all_rows, meta)
    ol_raw = compute_raw_oltrait(all_rows, meta)
    ol_shuffled = make_shuffled_ol(ol)

    log("[3/5] Building first10 thr=0.55 origin panel")
    hist = p7.compute_origin_history(rows_by, emb_by, THR)
    panel, events = p7.build_origin_panel(rows_by, emb_by, meta, ol, hist, THR, ORIGIN_WINDOW)
    add_ablation_columns(panel, ol_raw, ol_shuffled)

    log("[4/5] Training linear ablation models")
    scores = train_scores(panel, TARGET)
    evs, means = summarize_events(panel, scores)
    comps = run_comparisons(evs)

    for name in FEATURE_SETS:
        mb = means.get(name, {}).get("symbol_balanced", {})
        if mb:
            log(
                f"  {name:<22} NDCG={mb.get('ndcg3', float('nan')):.3f} "
                f"Hit={mb.get('hit1', float('nan')):.3f} "
                f"Mass={mb.get('mass3', float('nan')):.3f} "
                f"JS={mb.get('js', float('nan')):.3f}"
            )
    key = "ol_origin_full_vs_shuffled_ol_origin"
    if key in comps:
        nd = comps[key]["ndcg3"]["symbol_balanced_bootstrap"]
        ht = comps[key]["hit1"]["symbol_balanced_bootstrap"]
        js = comps[key]["js"]["symbol_balanced_bootstrap"]
        log(
            "  Full vs shuffled: "
            f"NDCG {nd['observed']:+.3f} CI[{nd['ci05']:+.3f},{nd['ci95']:+.3f}], "
            f"Hit {ht['observed']:+.3f} CI[{ht['ci05']:+.3f},{ht['ci95']:+.3f}], "
            f"JS {js['observed']:+.3f} CI[{js['ci05']:+.3f},{js['ci95']:+.3f}]"
        )

    result = {
        "task": "origin_alert_ablation",
        "threshold": THR,
        "origin_window": ORIGIN_WINDOW,
        "target": TARGET,
        "metrics": METRICS,
        "feature_sets": FEATURE_SETS,
        "comparisons": COMPARISONS,
        "bootstrap_B": BOOTSTRAP_B,
        "n_ol_kols": len(ol),
        "n_raw_ol_kols": len(ol_raw),
        "n_rows": len(panel),
        "n_train_rows": sum(r["split"] == "train" for r in panel),
        "n_val_rows": sum(r["split"] == "val" for r in panel),
        "n_panel_events": len(events),
        "means": means,
        "comparison_bootstrap": comps,
        "elapsed_sec": time.time() - t0,
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"[5/5] wrote {OUT}")
    log(f"elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
