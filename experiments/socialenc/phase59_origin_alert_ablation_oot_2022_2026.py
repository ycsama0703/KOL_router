"""Phase59: 2022-2026 out-of-time ablation for the OL-Origin router.

This keeps the phase33 ablation definitions fixed, but uses the modern OOT
protocol:
  - historical source score: before 2020-01-01
  - model training: 2020-01-01 to 2021-06-01
  - OOT test: 2022-06-01 to latest available data
"""
from __future__ import annotations

import collections
import json
import pathlib
import time

import numpy as np

import phase5_sentiment_reconstruction as p5
import phase7_origin_alert as p7
import phase33_origin_alert_ablation as p33

OUT_JSON = pathlib.Path(__file__).with_name("phase59_origin_alert_ablation_oot_2022_2026_result.json")
OUT_MD = pathlib.Path(__file__).with_name("phase59_origin_alert_ablation_oot_2022_2026_table.md")

MODEL_SPLIT = "2021-06-01"
OOT_START = "2022-06-01"
OOT_END = "2026-06-23"


def log(msg: str) -> None:
    print(msg, flush=True)


def configure_time_bounds() -> None:
    p5.TRAIN_END = "2020-01-01"
    p5.MODEL_SPLIT = MODEL_SPLIT
    p5.VAL_END = OOT_END
    p7.B = 600
    p33.BOOTSTRAP_B = 600


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


def summarize_oot(rows, scores):
    evs = {name: event_rows_oot(rows, sc, p33.TARGET) for name, sc in scores.items()}
    means = {}
    for name, ee in evs.items():
        means[name] = {
            "n_events": len(ee),
            "n_symbols": len(set(e["sym"] for e in ee)),
            "symbol_balanced": {m: p7.symbal_mean(ee, m) for m in p33.METRICS},
            "pooled": {m: p7.pooled_mean(ee, m) for m in p33.METRICS},
        }
    return evs, means


def comparison_bootstrap(evs):
    comps = {}
    for model, base in p33.COMPARISONS:
        if model not in evs or base not in evs:
            continue
        pairs = p33.aligned_pairs(evs[model], evs[base])
        comps[f"{model}_vs_{base}"] = {
            m: {"symbol_balanced_bootstrap": p33.bootstrap_symbol_balanced(pairs, m)}
            for m in p33.METRICS
        }
    return comps


def fmt(x, digits=3):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def write_md(result):
    no_ol = result["means"]["no_ol_strong"]["symbol_balanced"]
    order = [
        "ol_only",
        "no_ol_strong",
        "shuffled_ol_origin",
        "follower_replacement",
        "raw_ol_origin",
        "ol_origin_full",
    ]
    labels = {
        "ol_only": "OL only",
        "no_ol_strong": "No-OL strong",
        "shuffled_ol_origin": "Shuffled OL",
        "follower_replacement": "Follower replacement",
        "raw_ol_origin": "Raw OL",
        "ol_origin_full": "Full OL-Origin",
    }
    lines = [
        "# Phase59 2022-2026 OOT Ablation",
        "",
        "Frozen protocol: source-score history before 2020-01-01; router train 2020-01-01 to 2021-06-01; OOT test 2022-06-01 to 2026-06-22.",
        "",
        "| Method | Events | Symbols | NDCG@3 | Hit@1 | Mass@3 | JS ↓ | ΔNDCG vs No-OL | ΔHit vs No-OL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in order:
        if key not in result["means"]:
            continue
        m = result["means"][key]
        sb = m["symbol_balanced"]
        label = f"**{labels[key]}**" if key == "ol_origin_full" else labels[key]
        lines.append(
            f"| {label} | {m['n_events']} | {m['n_symbols']} | "
            f"{fmt(sb['ndcg3'])} | {fmt(sb['hit1'])} | {fmt(sb['mass3'])} | {fmt(sb['js'])} | "
            f"{sb['ndcg3'] - no_ol['ndcg3']:+.3f} | {sb['hit1'] - no_ol['hit1']:+.3f} |"
        )
    lines += [
        "",
        "Bootstrap support:",
        "",
        "| Comparison | ΔNDCG@3 90% CI | ΔHit@1 90% CI | JS improvement 90% CI |",
        "|---|---:|---:|---:|",
    ]
    for name, comp in result["comparison_bootstrap"].items():
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
    t0 = time.time()
    configure_time_bounds()
    log("[1/5] Loading tweets and embeddings")
    rows_by = {}
    emb_by = {}
    all_rows = []
    latest = {}
    for i, sym in enumerate(p5.SYMS, 1):
        rows, emb = p5.load_symbol(sym)
        rows_by[sym] = rows
        emb_by[sym] = emb
        all_rows.extend(rows)
        latest[sym] = max(r["day"] for r in rows) if rows else None
        log(f"  {i:02d}/{len(p5.SYMS)} {sym:<5} rows={len(rows):>6}")

    log("[2/5] Computing KOL metadata and OL traits")
    meta = p5.compute_metadata(all_rows)
    ol = p5.compute_oltrait(all_rows, meta)
    ol_raw = p33.compute_raw_oltrait(all_rows, meta)
    ol_shuffled = p33.make_shuffled_ol(ol)

    log("[3/5] Building first10 thr=0.55 origin panel")
    hist = p7.compute_origin_history(rows_by, emb_by, p33.THR)
    panel, events = p7.build_origin_panel(rows_by, emb_by, meta, ol, hist, p33.THR, p33.ORIGIN_WINDOW)
    p33.add_ablation_columns(panel, ol_raw, ol_shuffled)

    log("[4/5] Training ablation rows and evaluating OOT")
    scores = p33.train_scores(panel, p33.TARGET)
    evs, means = summarize_oot(panel, scores)
    comps = comparison_bootstrap(evs)

    result = {
        "task": "origin_alert_ablation_oot_2022_2026",
        "protocol": {
            "source_score_history_end": p5.TRAIN_END,
            "router_train_period": [p5.TRAIN_END, MODEL_SPLIT],
            "development_validation_period": [MODEL_SPLIT, OOT_START],
            "oot_test_period": [OOT_START, OOT_END],
            "semantic_threshold": p33.THR,
            "origin_window": p33.ORIGIN_WINDOW,
            "target": p33.TARGET,
        },
        "latest_by_symbol": latest,
        "feature_sets": p33.FEATURE_SETS,
        "comparisons": p33.COMPARISONS,
        "bootstrap_B": p33.BOOTSTRAP_B,
        "n_ol_kols": len(ol),
        "n_raw_ol_kols": len(ol_raw),
        "panel_counts": {
            "n_rows": len(panel),
            "n_train_rows": sum(r["split"] == "train" for r in panel),
            "n_post_train_rows": sum(r["split"] == "val" for r in panel),
            "n_oot_rows": sum(r["split"] == "val" and OOT_START <= r["day"] < OOT_END for r in panel),
            "n_events_raw": len(events),
        },
        "means": means,
        "comparison_bootstrap": comps,
        "elapsed_sec": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_md(result)
    log(f"[5/5] wrote {OUT_JSON.name} and {OUT_MD.name}")
    log(f"elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
