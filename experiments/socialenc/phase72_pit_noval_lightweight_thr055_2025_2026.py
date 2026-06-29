"""Phase72: PIT no-validation lightweight test at thr=0.55 on 2025-2026.

This replaces the superseded frozen 2020-2021 -> 2022-2026 diagnostic.

Protocol:
  - Discovery remains descriptive and may use the full archive.
  - Model experiment is point-in-time.
  - Train:      2022-06-01 <= day < 2025-06-01
  - Validation: none
  - Test:       2025-06-01 <= day < 2026-06-01

For each block, originator role O_k and historical origin statistics are
estimated only from data before that block starts:
  - 2022-2023 train rows use history before 2022-06-01
  - 2023-2024 train rows use history before 2023-06-01
  - 2024-2025 train rows use history before 2024-06-01
  - test rows use history before 2025-06-01

This file runs only lightweight methods:
  - main structural baselines from phase7
  - ablation rows from phase33
No encoder, local LLM, or API rows are run here.
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
import phase33_origin_alert_ablation as p33

OUT_JSON = pathlib.Path(__file__).with_name("phase72_pit_noval_lightweight_thr055_2025_2026_result.json")
OUT_MD = pathlib.Path(__file__).with_name("phase72_pit_noval_lightweight_thr055_2025_2026_table.md")

TRAIN_START = "2022-06-01"
TRAIN_MID = "2023-06-01"
TRAIN_MID2 = "2024-06-01"
TRAIN_END = "2025-06-01"
VAL_END = TRAIN_END
TEST_END = "2026-06-01"

THR = 0.55
ORIGIN_WINDOW = {"name": "first10", "max_rank": 10}
TARGET = "log_future_reach"
RIDGE_ALPHAS = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
FIXED_ALPHA = 1000.0
BOOTSTRAP_B = 600
RNG = np.random.default_rng(61061)

MAIN_METHODS = [
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

ABLATION_METHODS = [
    ("Ablation", "OL only", "ol_only"),
    ("Ablation", "No-OL strong", "no_ol_strong"),
    ("Ablation", "Shuffled OL", "shuffled_ol_origin"),
    ("Ablation", "Follower replacement", "follower_replacement"),
    ("Ablation", "Raw OL", "raw_ol_origin"),
    ("Ablation", "Full OL-Origin", "ol_origin_full"),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def block_for_day(day: str) -> str | None:
    if TRAIN_START <= day < TRAIN_MID:
        return "train_2022_2023"
    if TRAIN_MID <= day < TRAIN_MID2:
        return "train_2023_2024"
    if TRAIN_MID2 <= day < TRAIN_END:
        return "train_2024_2025"
    if VAL_END <= day < TEST_END:
        return "test_2025_2026"
    return None


def cutoff_for_block(block: str) -> str:
    return {
        "train_2022_2023": TRAIN_START,
        "train_2023_2024": TRAIN_MID,
        "train_2024_2025": TRAIN_MID2,
        "test_2025_2026": VAL_END,
    }[block]


def split_for_block(block: str) -> str:
    if block.startswith("train"):
        return "train"
    return "test" if block.startswith("test") else block


def rows_before(all_rows: list[dict], cutoff: str) -> list[dict]:
    return [r for r in all_rows if r["day"] < cutoff]


def with_train_end(cutoff: str, fn, *args):
    old_train_end = p5.TRAIN_END
    old_val_end = p5.VAL_END
    try:
        p5.TRAIN_END = cutoff
        p5.VAL_END = cutoff
        return fn(*args)
    finally:
        p5.TRAIN_END = old_train_end
        p5.VAL_END = old_val_end


def compute_block_state(rows_by, emb_by, all_rows):
    states = {}
    for block in ["train_2022_2023", "train_2023_2024", "train_2024_2025", "test_2025_2026"]:
        cutoff = cutoff_for_block(block)
        hist_rows = rows_before(all_rows, cutoff)
        meta = p5.compute_metadata(hist_rows)
        ol = with_train_end(cutoff, p5.compute_oltrait, hist_rows, meta)
        hist = with_train_end(cutoff, p7.compute_origin_history, rows_by, emb_by, THR)
        raw_ol = with_train_end(cutoff, p33.compute_raw_oltrait, hist_rows, meta)
        shuffled_ol = make_shuffled_ol(ol, block)
        states[block] = {
            "cutoff": cutoff,
            "meta": meta,
            "ol": ol,
            "hist": hist,
            "raw_ol": raw_ol,
            "shuffled_ol": shuffled_ol,
        }
        log(
            f"  {block:<14} cutoff={cutoff} "
            f"history_rows={len(hist_rows):>6} OL_KOLs={len(ol):>4} raw_OL_KOLs={len(raw_ol):>4}"
        )
    return states


def make_shuffled_ol(ol: dict[str, float], block: str) -> dict[str, float]:
    keys = sorted(ol)
    vals = np.array([ol[k] for k in keys], dtype=float)
    seed = 61061 + sum(ord(c) for c in block)
    vals = vals[np.random.default_rng(seed).permutation(len(vals))]
    return {k: float(v) for k, v in zip(keys, vals)}


def build_pit_origin_panel(rows_by, emb_by, states):
    panel = []
    event_summaries = []
    max_rank = ORIGIN_WINDOW["max_rank"]
    for si, sym in enumerate(p5.SYMS, 1):
        ev = p5.first_by_event(rows_by[sym], start=TRAIN_START, end=TEST_END)
        past_cents = []
        used_events = 0
        for (_, day), d in sorted(ev.items(), key=lambda kv: kv[0][1]):
            block = block_for_day(day)
            if block is None:
                continue
            state = states[block]
            ol = state["ol"]
            hist = state["hist"]
            meta = state["meta"]
            raw_ol = state["raw_ol"]
            shuffled_ol = state["shuffled_ol"]

            items = sorted(d.values(), key=lambda r: r["ts"])
            if len(items) < p7.MIN_EVENT_KOLS:
                continue
            clusters = []
            cents = []
            event_start = items[0]["ts"]
            event_candidates = []
            for pos, r in enumerate(items):
                rank = pos + 1
                v = p5.norm_vec(emb_by[sym], r["idx"])
                best = -1.0
                best_j = -1
                for j, c in enumerate(cents):
                    sim = float(v @ c)
                    if sim > best:
                        best = sim
                        best_j = j
                creates_new = not (best_j >= 0 and best >= THR)
                if creates_new:
                    if past_cents:
                        P = np.vstack(past_cents[-1000:])
                        novelty_global = 1.0 - float(np.max(v @ P.T))
                    else:
                        novelty_global = np.nan
                    if cents:
                        E = np.vstack(cents)
                        novelty_event = 1.0 - float(np.max(v @ E.T))
                    else:
                        novelty_event = np.nan
                    clusters.append([r])
                    cents.append(v)
                    frame_j = len(clusters) - 1
                    if r["kol"] in ol and rank <= max_rank:
                        m = meta.get(r["kol"], {})
                        h = hist.get(r["kol"], {})
                        logf = float(m.get("log_followers", math.log1p(max(0.0, float(r["followers"])))))
                        residual_ol = float(ol[r["kol"]])
                        raw = float(raw_ol.get(r["kol"], residual_ol))
                        shuf = float(shuffled_ol.get(r["kol"], 0.0))
                        nov = 0.0 if not np.isfinite(novelty_global) else float(novelty_global)
                        row = {
                            "event_id": f"{sym}:{day}:thr{THR:.2f}:{ORIGIN_WINDOW['name']}",
                            "sym": sym,
                            "day": day,
                            "block": block,
                            "split": split_for_block(block),
                            "history_cutoff": state["cutoff"],
                            "thr": THR,
                            "origin_window": ORIGIN_WINDOW["name"],
                            "frame_j": frame_j,
                            "origin_kol": r["kol"],
                            "origin_text": r.get("text", ""),
                            "origin_ol": residual_ol,
                            "origin_logfoll": logf,
                            "origin_verified": float(m.get("verified", r["verified"])),
                            "log_origin_rank": math.log(rank),
                            "elapsed_hours": (r["ts"] - event_start) / 3600.0,
                            "prior_frame_count": float(len(cents) - 1),
                            "origin_stance": float(r["stance"]),
                            "origin_stance_abs": abs(float(r["stance"])),
                            "novelty_global": novelty_global,
                            "novelty_event": novelty_event,
                            "hist_log_origin_count": float(h.get("hist_log_origin_count", 0.0)),
                            "hist_mean_log_adopt": float(h.get("hist_mean_log_adopt", 0.0)),
                            "hist_success_rate": float(h.get("hist_success_rate", 0.0)),
                            "future_adopt": 0.0,
                            "future_reach": 0.0,
                            "origin_ol_raw": raw,
                            "origin_ol_shuffled": shuf,
                        }
                        row["ol_x_visibility"] = row["origin_ol"] * row["origin_logfoll"]
                        row["ol_x_novelty"] = row["origin_ol"] * nov
                        row["raw_ol_x_visibility"] = raw * row["origin_logfoll"]
                        row["raw_ol_x_novelty"] = raw * nov
                        row["shuffled_ol_x_visibility"] = shuf * row["origin_logfoll"]
                        row["shuffled_ol_x_novelty"] = shuf * nov
                        row["logfoll_x_visibility"] = row["origin_logfoll"] * row["origin_logfoll"]
                        row["logfoll_x_novelty"] = row["origin_logfoll"] * nov
                        event_candidates.append(row)
                else:
                    clusters[best_j].append(r)
                    c = cents[best_j] * (len(clusters[best_j]) - 1) + v
                    cents[best_j] = c / (np.linalg.norm(c) + 1e-12)

            if not event_candidates:
                past_cents.extend(cents)
                continue
            for row in event_candidates:
                cl = clusters[row["frame_j"]]
                origin_ts = cl[0]["ts"]
                followers = [r for r in cl[1:] if r["ts"] > origin_ts]
                row["future_adopt"] = float(len(followers))
                row["future_reach"] = float(sum(max(0.0, float(r["followers"])) for r in followers))
                row["log_future_adopt"] = math.log1p(row["future_adopt"])
                row["log_future_reach"] = math.log1p(row["future_reach"])
                panel.append(row)
            used_events += 1
            event_summaries.append({
                "event_id": event_candidates[0]["event_id"],
                "sym": sym,
                "day": day,
                "block": block,
                "n_candidates": len(event_candidates),
                "event_kols": len(items),
            })
            past_cents.extend(cents)
        log(f"  thr={THR:.2f} {ORIGIN_WINDOW['name']:<7} {si:02d}/{len(p5.SYMS)} {sym:<5} events={used_events:>4} rows={len(panel):>6}")
    return panel, event_summaries


def matrix(rows: list[dict], features: list[str]) -> np.ndarray:
    return np.array([[r.get(f, np.nan) for f in features] for r in rows], dtype=float)


def train_standardizer(X, mask):
    med = np.nanmedian(X[mask], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X2 = np.where(np.isfinite(X), X, med)
    mu = X2[mask].mean(axis=0)
    sd = X2[mask].std(axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return med, mu, sd


def apply_standardizer(X, med, mu, sd):
    X2 = np.where(np.isfinite(X), X, med)
    return (X2 - mu) / sd


def ridge_predict_from_mask(rows, features, target, fit_mask, alpha):
    X = matrix(rows, features)
    y = np.array([r[target] for r in rows], dtype=float)
    fit_mask = fit_mask & np.isfinite(y)
    if fit_mask.sum() < len(features) + 30:
        return None
    med, mu, sd = train_standardizer(X, fit_mask)
    Xs = apply_standardizer(X, med, mu, sd)
    Xtr = np.column_stack([np.ones(fit_mask.sum()), Xs[fit_mask]])
    ytr = y[fit_mask]
    lam = np.eye(Xtr.shape[1]) * alpha
    lam[0, 0] = 0.0
    beta = np.linalg.solve(Xtr.T @ Xtr + lam, Xtr.T @ ytr)
    return np.column_stack([np.ones(len(rows)), Xs]) @ beta


def event_rows_for_split(rows, scores, target, split):
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r["split"] == split and np.isfinite(scores[i]):
            groups[r["event_id"]].append((r["sym"], float(r[target]), float(scores[i])))
    return event_metrics_from_groups(groups)


def event_metrics_from_groups(groups):
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


def fit_select_score(rows, features, target):
    splits = np.array([r["split"] for r in rows])
    train = splits == "train"
    final = ridge_predict_from_mask(rows, features, target, train, FIXED_ALPHA)
    return final, {"selected_alpha": FIXED_ALPHA, "selection": "fixed_no_validation"}


def evaluate_methods(rows, feature_sets):
    scores = {}
    selection = {}
    for name, feats in feature_sets.items():
        pred, sel = fit_select_score(rows, feats, TARGET)
        if pred is not None:
            scores[name] = pred
            selection[name] = sel
            log(f"  {name:<22} alpha={sel['selected_alpha']:g} fixed_no_val")
    evs = {name: event_rows_for_split(rows, pred, TARGET, "test") for name, pred in scores.items()}
    means = {
        name: {
            "n_events": len(ee),
            "n_symbols": len(set(e["sym"] for e in ee)),
            "symbol_balanced": {m: p7.symbal_mean(ee, m) for m in p7.METRICS},
            "pooled": {m: p7.pooled_mean(ee, m) for m in p7.METRICS},
        }
        for name, ee in evs.items()
    }
    return scores, evs, means, selection


def aligned_pairs(a, b):
    aa = {e["event_id"]: e for e in a}
    bb = {e["event_id"]: e for e in b}
    return [(aa[k], bb[k]) for k in sorted(set(aa) & set(bb))]


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


def bootstrap_symbal(pairs, metric):
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


def comparisons(evs, pairs):
    comps = {}
    for model, base in pairs:
        if model not in evs or base not in evs:
            continue
        aligned = aligned_pairs(evs[model], evs[base])
        comps[f"{model}_vs_{base}"] = {
            m: {"symbol_balanced_bootstrap": bootstrap_symbal(aligned, m)}
            for m in p7.METRICS
        }
    return comps


def fmt(x, digits=3):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def make_table_rows(means, rowspec, baseline="no_ol_strong"):
    out = []
    base = means.get(baseline, {}).get("symbol_balanced", {})
    for family, label, key in rowspec:
        if key not in means:
            continue
        m = means[key]
        sb = m["symbol_balanced"]
        out.append({
            "family": family,
            "method": label,
            "key": key,
            "events": m["n_events"],
            "symbols": m["n_symbols"],
            "ndcg3": sb["ndcg3"],
            "hit1": sb["hit1"],
            "mass3": sb["mass3"],
            "js": sb["js"],
            "delta_ndcg": sb["ndcg3"] - base.get("ndcg3", np.nan),
            "delta_hit": sb["hit1"] - base.get("hit1", np.nan),
        })
    return out


def write_md(result):
    lines = [
        "# Phase72 PIT No-Val Lightweight thr=0.55 2025-2026 Test",
        "",
        "Protocol: train 2022-06-01 to 2025-06-01; no validation split; final test 2025-06-01 to 2026-06-01. Ridge alpha is fixed at 1000 to avoid test leakage. O_k/history are estimated only from data before each point-in-time block.",
        "",
        "## Main Lightweight Rows",
        "",
        "| Family | Method | Events | Symbols | NDCG@3 | Hit@1 | Mass@3 | JS ↓ | ΔNDCG vs No-OL | ΔHit vs No-OL |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in result["main_table_rows"]:
        method = f"**{r['method']}**" if r["key"] == "ol_origin" else r["method"]
        lines.append(
            f"| {r['family']} | {method} | {r['events']} | {r['symbols']} | "
            f"{fmt(r['ndcg3'])} | {fmt(r['hit1'])} | {fmt(r['mass3'])} | {fmt(r['js'])} | "
            f"{r['delta_ndcg']:+.3f} | {r['delta_hit']:+.3f} |"
        )
    lines += [
        "",
        "Bootstrap, OL-Origin vs selected baselines:",
        "",
        "| Comparison | ΔNDCG@3 90% CI | ΔHit@1 90% CI | JS improvement 90% CI |",
        "|---|---:|---:|---:|",
    ]
    for name, comp in result["main_comparisons"].items():
        nd = comp["ndcg3"]["symbol_balanced_bootstrap"]
        hit = comp["hit1"]["symbol_balanced_bootstrap"]
        js = comp["js"]["symbol_balanced_bootstrap"]
        lines.append(
            f"| {name} | {nd['observed']:+.3f} [{nd['ci05']:+.3f}, {nd['ci95']:+.3f}] | "
            f"{hit['observed']:+.3f} [{hit['ci05']:+.3f}, {hit['ci95']:+.3f}] | "
            f"{js['observed']:+.3f} [{js['ci05']:+.3f}, {js['ci95']:+.3f}] |"
        )
    lines += [
        "",
        "## Ablation Rows",
        "",
        "| Method | Events | Symbols | NDCG@3 | Hit@1 | Mass@3 | JS ↓ | ΔNDCG vs No-OL | ΔHit vs No-OL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in result["ablation_table_rows"]:
        method = f"**{r['method']}**" if r["key"] == "ol_origin_full" else r["method"]
        lines.append(
            f"| {method} | {r['events']} | {r['symbols']} | "
            f"{fmt(r['ndcg3'])} | {fmt(r['hit1'])} | {fmt(r['mass3'])} | {fmt(r['js'])} | "
            f"{r['delta_ndcg']:+.3f} | {r['delta_hit']:+.3f} |"
        )
    lines += [
        "",
        "Bootstrap, Full OL-Origin vs ablation baselines:",
        "",
        "| Comparison | ΔNDCG@3 90% CI | ΔHit@1 90% CI | JS improvement 90% CI |",
        "|---|---:|---:|---:|",
    ]
    for name, comp in result["ablation_comparisons"].items():
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
    log("[1/6] Loading tweets and embeddings")
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

    log("[2/6] Computing PIT block states")
    states = compute_block_state(rows_by, emb_by, all_rows)

    log("[3/6] Building continuous PIT origin panel")
    panel, events = build_pit_origin_panel(rows_by, emb_by, states)
    log(
        f"  rows={len(panel)} train={sum(r['split']=='train' for r in panel)} "
        f"val={sum(r['split']=='val' for r in panel)} test={sum(r['split']=='test' for r in panel)} "
        f"events={len(events)}"
    )

    log("[4/6] Main lightweight rows")
    main_scores, main_evs, main_means, main_selection = evaluate_methods(panel, p7.FEATURE_SETS)
    main_comps = comparisons(
        main_evs,
        [
            ("ol_origin", "followers"),
            ("ol_origin", "visibility"),
            ("ol_origin", "rank_time"),
            ("ol_origin", "sentiment"),
            ("ol_origin", "novelty"),
            ("ol_origin", "history"),
            ("ol_origin", "no_ol_strong"),
        ],
    )

    log("[5/6] Ablation rows")
    _abl_scores, abl_evs, abl_means, abl_selection = evaluate_methods(panel, p33.FEATURE_SETS)
    abl_comps = comparisons(
        abl_evs,
        [
            ("ol_origin_full", "no_ol_strong"),
            ("ol_origin_full", "ol_only"),
            ("ol_origin_full", "shuffled_ol_origin"),
            ("ol_origin_full", "follower_replacement"),
            ("ol_origin_full", "raw_ol_origin"),
            ("raw_ol_origin", "no_ol_strong"),
            ("shuffled_ol_origin", "no_ol_strong"),
            ("follower_replacement", "no_ol_strong"),
        ],
    )

    result = {
        "task": "pit_noval_lightweight_main_ablation_thr055_2025_2026",
        "protocol": {
            "train": [TRAIN_START, TRAIN_END],
            "validation": None,
            "test": [VAL_END, TEST_END],
            "train_blocks": {
                "train_2022_2023": [TRAIN_START, TRAIN_MID],
                "train_2023_2024": [TRAIN_MID, TRAIN_MID2],
                "train_2024_2025": [TRAIN_MID2, TRAIN_END],
            },
            "test_blocks": {
                "test_2025_2026": [VAL_END, TEST_END],
            },
            "history_cutoffs": {block: states[block]["cutoff"] for block in states},
            "semantic_threshold": THR,
            "origin_window": ORIGIN_WINDOW,
            "target": TARGET,
            "alpha_selection": "fixed alpha=1000; no validation split",
        },
        "latest_by_symbol": latest,
        "panel_counts": {
            "n_rows": len(panel),
            "n_train_rows": sum(r["split"] == "train" for r in panel),
            "n_val_rows": sum(r["split"] == "val" for r in panel),
            "n_test_rows": sum(r["split"] == "test" for r in panel),
            "n_events": len(events),
            "n_test_events_raw": len(set(r["event_id"] for r in panel if r["split"] == "test")),
        },
        "block_state_counts": {
            block: {
                "history_cutoff": state["cutoff"],
                "n_ol_kols": len(state["ol"]),
                "n_raw_ol_kols": len(state["raw_ol"]),
                "n_hist_kols": len(state["hist"]),
            }
            for block, state in states.items()
        },
        "main_feature_sets": p7.FEATURE_SETS,
        "ablation_feature_sets": p33.FEATURE_SETS,
        "main_model_selection": main_selection,
        "ablation_model_selection": abl_selection,
        "main_means": main_means,
        "ablation_means": abl_means,
        "main_comparisons": main_comps,
        "ablation_comparisons": abl_comps,
        "main_table_rows": make_table_rows(main_means, MAIN_METHODS),
        "ablation_table_rows": make_table_rows(abl_means, ABLATION_METHODS),
        "elapsed_sec": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_md(result)
    log(f"[6/6] wrote {OUT_JSON.name} and {OUT_MD.name}")
    log(f"elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
