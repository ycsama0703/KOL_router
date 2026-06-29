"""Phase58: 2022-2026 out-of-time test for origin-alert main rows.

This phase freezes the original training protocol:
  - historical source score: before 2020-01-01
  - model training: 2020-01-01 to 2021-06-01

It then evaluates on a modern out-of-time window:
  - OOT test: 2022-06-01 to latest available data in the local archive

Batch 1 covers rows that are fully reproducible without API calls or large text
encoder downloads:
  - Scale, Context, Origin Role from phase7
  - Surface Text rows from phase29
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
import phase28_origin_alert_encoder_baselines as p28
import phase29_origin_alert_text_surface_diagnostic as p29

OUT_JSON = pathlib.Path(__file__).with_name("phase58_origin_alert_oot_2022_2026_result.json")
OUT_MD = pathlib.Path(__file__).with_name("phase58_origin_alert_oot_2022_2026_table.md")

THR = 0.55
ORIGIN_WINDOW = {"name": "first10", "max_rank": 10}
TARGET = "log_future_reach"
MODEL_SPLIT = "2021-06-01"
OOT_START = "2022-06-01"
OOT_END = "2026-06-23"  # exclusive; local archive currently reaches 2026-06-22

STRUCTURAL_METHODS = [
    ("Scale", "Follower", "followers"),
    ("Scale", "Visibility", "visibility"),
    ("Context", "Rank/Time", "rank_time"),
    ("Context", "Sentiment", "sentiment"),
    ("Context", "Novelty", "novelty"),
    ("Context", "History", "history"),
    ("Context", "No-OL Strong", "no_ol_strong"),
    ("Origin Role", "OL Only", "ol_only"),
    ("Origin Role", "OL-Origin", "ol_origin"),
]
SURFACE_METHODS = [
    ("Surface Text", "Symbol one-hot", "symbol_onehot"),
    ("Surface Text", "Text surface", "text_surface"),
    ("Surface Text", "Symbol + surface", "symbol_plus_surface"),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def configure_time_bounds() -> None:
    p5.TRAIN_END = "2020-01-01"
    p5.MODEL_SPLIT = MODEL_SPLIT
    p5.VAL_END = OOT_END
    p7.B = 600


def load_panel():
    rows_by = {}
    emb_by = {}
    all_rows = []
    latest = {}
    for sym in p5.SYMS:
        rows, emb = p5.load_symbol(sym)
        rows_by[sym] = rows
        emb_by[sym] = emb
        all_rows.extend(rows)
        latest[sym] = max(r["day"] for r in rows) if rows else None
    meta = p5.compute_metadata(all_rows)
    ol = p5.compute_oltrait(all_rows, meta)
    hist = p7.compute_origin_history(rows_by, emb_by, THR)
    panel, events = p7.build_origin_panel(rows_by, emb_by, meta, ol, hist, THR, ORIGIN_WINDOW)
    return panel, events, latest


def event_rows_oot(rows, scores, target):
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r["split"] == "val" and OOT_START <= r["day"] < OOT_END and np.isfinite(scores[i]):
            groups[r["event_id"]].append((r["sym"], float(r[target]), float(scores[i])))
    out = []
    for event_id, vals in groups.items():
        if len(vals) < 2:
            continue
        y = np.array([v[1] for v in vals], dtype=float)
        s = np.array([v[2] for v in vals], dtype=float)
        if not np.isfinite(y).all() or not np.isfinite(s).all() or np.nanmax(y) <= 0:
            continue
        p = y / max(float(y.sum()), 1e-12)
        q = p5.softmax(s)
        order = np.argsort(-s)
        ideal = np.argsort(-y)
        k = min(3, len(y))
        best = set(np.where(y == np.nanmax(y))[0].tolist())
        out.append({
            "event_id": event_id,
            "sym": vals[0][0],
            "ndcg3": float(p5.dcg(y[order[:k]]) / max(p5.dcg(y[ideal[:k]]), 1e-12)),
            "hit1": float(1.0 if order[0] in best else 0.0),
            "mass3": float(p[order[:k]].sum()),
            "js": float(p5.js_divergence(p, q)),
        })
    return out


def summarize_scores(rows, scores_by_method):
    means = {}
    evs = {}
    for name, scores in scores_by_method.items():
        ee = event_rows_oot(rows, scores, TARGET)
        evs[name] = ee
        means[name] = {
            "n_events": len(ee),
            "n_symbols": len(set(e["sym"] for e in ee)),
            "symbol_balanced": {m: p7.symbal_mean(ee, m) for m in p7.METRICS},
            "pooled": {m: p7.pooled_mean(ee, m) for m in p7.METRICS},
        }
    return evs, means


def bootstrap_vs_ol(evs):
    comps = {}
    if "ol_origin" not in evs:
        return comps
    for base in ["no_ol_strong", "followers", "visibility", "text_surface", "symbol_plus_surface"]:
        if base not in evs:
            continue
        pairs = p7.aligned_pairs(evs["ol_origin"], evs[base])
        comps[f"ol_origin_vs_{base}"] = {
            metric: {
                "symbol_balanced_bootstrap": p7.bootstrap_symbal(pairs, metric),
                "pooled_bootstrap": p7.bootstrap_pooled(pairs, metric),
            }
            for metric in p7.METRICS
        }
    return comps


def build_table_rows(means):
    rows = []
    no_ol = means["no_ol_strong"]["symbol_balanced"]
    for family, label, key in STRUCTURAL_METHODS + SURFACE_METHODS:
        if key not in means:
            continue
        m = means[key]
        sb = m["symbol_balanced"]
        rows.append({
            "family": family,
            "method": label,
            "key": key,
            "events": m["n_events"],
            "symbols": m["n_symbols"],
            "ndcg3": sb["ndcg3"],
            "hit1": sb["hit1"],
            "mass3": sb["mass3"],
            "js": sb["js"],
            "delta_ndcg": sb["ndcg3"] - no_ol["ndcg3"],
            "delta_hit": sb["hit1"] - no_ol["hit1"],
        })
    return rows


def fmt(x, digits=3):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def write_md(result):
    lines = [
        "# Phase58 2022-2026 Out-of-Time Main Rows",
        "",
        "Frozen protocol: source-score history before 2020-01-01; router train 2020-01-01 to 2021-06-01; OOT test 2022-06-01 to 2026-06-22.",
        "",
        "| Family | Method | Events | Symbols | NDCG@3 | Hit@1 | Mass@3 | JS ↓ | ΔNDCG | ΔHit |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in result["table_rows"]:
        method = f"**{r['method']}**" if r["key"] == "ol_origin" else r["method"]
        lines.append(
            f"| {r['family']} | {method} | {r['events']} | {r['symbols']} | "
            f"{fmt(r['ndcg3'])} | {fmt(r['hit1'])} | {fmt(r['mass3'])} | {fmt(r['js'])} | "
            f"{r['delta_ndcg']:+.3f} | {r['delta_hit']:+.3f} |"
        )
    lines += [
        "",
        "Bootstrap support, OL-Origin vs selected baselines:",
        "",
        "| Comparison | ΔNDCG@3 90% CI | ΔHit@1 90% CI | JS improvement 90% CI |",
        "|---|---:|---:|---:|",
    ]
    for name, comp in result["comparisons"].items():
        nd = comp["ndcg3"]["symbol_balanced_bootstrap"]
        hit = comp["hit1"]["symbol_balanced_bootstrap"]
        js = comp["js"]["symbol_balanced_bootstrap"]
        lines.append(
            f"| {name} | {nd['observed']:+.3f} [{nd['ci05']:+.3f}, {nd['ci95']:+.3f}] | "
            f"{hit['observed']:+.3f} [{hit['ci05']:+.3f}, {hit['ci95']:+.3f}] | "
            f"{js['observed']:+.3f} [{js['ci05']:+.3f}, {js['ci95']:+.3f}] |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    started = time.time()
    configure_time_bounds()
    log("[1/4] Building 2020-2026 panel with frozen 2020-2021 training split")
    panel, events, latest = load_panel()

    log("[2/4] Fitting structural rows")
    scores = p7.train_scores(panel, TARGET)

    log("[3/4] Fitting surface-text rows")
    matrices = p29.build_matrices(panel)
    model_selection = {}
    for method, X in matrices.items():
        score = p28.fit_matrix_score(panel, X, TARGET, method, model_selection)
        if score is not None:
            scores[method] = score

    evs, means = summarize_scores(panel, scores)
    result = {
        "task": "origin_alert_out_of_time_2022_2026_batch1",
        "protocol": {
            "source_score_history_end": p5.TRAIN_END,
            "router_train_period": [p5.TRAIN_END, MODEL_SPLIT],
            "development_validation_period": [MODEL_SPLIT, OOT_START],
            "oot_test_period": [OOT_START, OOT_END],
            "semantic_threshold": THR,
            "origin_window": ORIGIN_WINDOW,
            "target": TARGET,
        },
        "latest_by_symbol": latest,
        "panel_counts": {
            "n_rows": len(panel),
            "n_train_rows": sum(r["split"] == "train" for r in panel),
            "n_post_train_rows": sum(r["split"] == "val" for r in panel),
            "n_oot_rows": sum(r["split"] == "val" and OOT_START <= r["day"] < OOT_END for r in panel),
            "n_events_raw": len(events),
        },
        "model_selection": model_selection,
        "means": means,
        "comparisons": bootstrap_vs_ol(evs),
        "table_rows": [],
        "elapsed_sec": None,
    }
    result["table_rows"] = build_table_rows(means)
    result["elapsed_sec"] = time.time() - started
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_md(result)
    log(f"[4/4] wrote {OUT_JSON.name} and {OUT_MD.name}")
    log(f"elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
