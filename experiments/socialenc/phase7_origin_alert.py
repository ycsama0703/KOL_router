"""Pre-popularity origin alert for KOL narrative diffusion.

Application point:
  A trading/research agent sees a newly originated semantic frame and must decide
  whether to put it on the watchlist before multiple KOLs have followed it.

This experiment forbids early popularity features:
  - no early_adopt;
  - no early_reach;
  - no early_frame_share.

Candidate:
  A new frame created online inside a symbol-day event. The frame identity is
  created from current and prior tweets only. Labels are computed from subsequent
  KOLs that later attach to the same online frame.

Main question:
  Does the deconfounded OLtrait add value beyond fair origin-time baselines
  such as originator followers, verified status, rank/time, sentiment, novelty,
  and pre-2020 historical adoption rate?
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import time

import numpy as np

import phase5_sentiment_reconstruction as p5

OUT = pathlib.Path(__file__).with_name("phase7_origin_alert_result.json")

THRESHOLDS = [0.45, 0.55, 0.65]
ORIGIN_WINDOWS = [
    {"name": "first3", "max_rank": 3},
    {"name": "first5", "max_rank": 5},
    {"name": "first10", "max_rank": 10},
    {"name": "all", "max_rank": None},
]
MIN_EVENT_KOLS = 8
B = 600
RNG = np.random.default_rng(707)

FEATURE_SETS = {
    "followers": ["origin_logfoll"],
    "visibility": ["origin_logfoll", "origin_verified"],
    "rank_time": ["log_origin_rank", "elapsed_hours", "prior_frame_count"],
    "sentiment": ["origin_stance", "origin_stance_abs"],
    "novelty": ["novelty_global", "novelty_event"],
    "history": ["hist_log_origin_count", "hist_mean_log_adopt", "hist_success_rate"],
    "no_ol_strong": [
        "origin_logfoll", "origin_verified", "log_origin_rank", "elapsed_hours",
        "prior_frame_count", "origin_stance", "origin_stance_abs",
        "novelty_global", "novelty_event", "hist_log_origin_count",
        "hist_mean_log_adopt", "hist_success_rate",
    ],
    "ol_only": ["origin_ol"],
    "ol_origin": [
        "origin_ol", "origin_logfoll", "origin_verified", "log_origin_rank",
        "elapsed_hours", "prior_frame_count", "origin_stance",
        "origin_stance_abs", "novelty_global", "novelty_event",
        "hist_log_origin_count", "hist_mean_log_adopt", "hist_success_rate",
        "ol_x_visibility", "ol_x_novelty",
    ],
}

COMPARISONS = [
    ("ol_origin", "followers"),
    ("ol_origin", "visibility"),
    ("ol_origin", "rank_time"),
    ("ol_origin", "sentiment"),
    ("ol_origin", "novelty"),
    ("ol_origin", "history"),
    ("ol_origin", "no_ol_strong"),
    ("no_ol_strong", "followers"),
    ("no_ol_strong", "sentiment"),
]

TARGETS = ["log_future_adopt", "log_future_reach"]
METRICS = ["ndcg3", "hit1", "mass3", "js"]


def log(msg: str) -> None:
    print(msg, flush=True)


def online_cluster(items, emb, thr):
    clusters = []
    cents = []
    assignments = []
    new_flags = []
    sims = []
    for r in items:
        v = p5.norm_vec(emb, r["idx"])
        best = -1.0
        best_j = -1
        for j, c in enumerate(cents):
            sim = float(v @ c)
            if sim > best:
                best = sim
                best_j = j
        if best_j >= 0 and best >= thr:
            clusters[best_j].append(r)
            c = cents[best_j] * (len(clusters[best_j]) - 1) + v
            cents[best_j] = c / (np.linalg.norm(c) + 1e-12)
            assignments.append(best_j)
            new_flags.append(False)
            sims.append(best)
        else:
            clusters.append([r])
            cents.append(v)
            assignments.append(len(clusters) - 1)
            new_flags.append(True)
            sims.append(np.nan)
    return clusters, cents, assignments, new_flags, sims


def compute_origin_history(rows_by, emb_by, thr):
    """Pre-2020 originator history, strictly before model/eval period."""
    stats = collections.defaultdict(lambda: {
        "n_origin": 0.0,
        "sum_log_adopt": 0.0,
        "n_success": 0.0,
        "sum_log_reach": 0.0,
    })
    for sym in p5.SYMS:
        ev = p5.first_by_event(rows_by[sym], end=p5.TRAIN_END)
        for (_, _day), d in ev.items():
            items = sorted(d.values(), key=lambda r: r["ts"])
            if len(items) < MIN_EVENT_KOLS:
                continue
            clusters, _cents, _assign, new_flags, _sims = online_cluster(items, emb_by[sym], thr)
            for j, cl in enumerate(clusters):
                origin = cl[0]
                adopt = max(0, len(cl) - 1)
                reach = sum(max(0.0, float(r["followers"])) for r in cl[1:])
                s = stats[origin["kol"]]
                s["n_origin"] += 1.0
                s["sum_log_adopt"] += math.log1p(adopt)
                s["sum_log_reach"] += math.log1p(reach)
                if adopt >= 1:
                    s["n_success"] += 1.0
    hist = {}
    for k, s in stats.items():
        n = max(1.0, s["n_origin"])
        hist[k] = {
            "hist_log_origin_count": math.log1p(s["n_origin"]),
            "hist_mean_log_adopt": s["sum_log_adopt"] / n,
            "hist_mean_log_reach": s["sum_log_reach"] / n,
            "hist_success_rate": s["n_success"] / n,
        }
    return hist


def build_origin_panel(rows_by, emb_by, meta, ol, hist, thr, origin_window):
    panel = []
    event_summaries = []
    max_rank = origin_window["max_rank"]
    for si, sym in enumerate(p5.SYMS, 1):
        ev = p5.first_by_event(rows_by[sym], start=p5.TRAIN_END, end=p5.VAL_END)
        past_cents = []
        used_events = 0
        for (_, day), d in sorted(ev.items(), key=lambda kv: kv[0][1]):
            items = sorted(d.values(), key=lambda r: r["ts"])
            if len(items) < MIN_EVENT_KOLS:
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
                creates_new = not (best_j >= 0 and best >= thr)
                if creates_new:
                    prior_event_cents = cents
                    if past_cents:
                        P = np.vstack(past_cents[-1000:])
                        novelty_global = 1.0 - float(np.max(v @ P.T))
                    else:
                        novelty_global = np.nan
                    if prior_event_cents:
                        E = np.vstack(prior_event_cents)
                        novelty_event = 1.0 - float(np.max(v @ E.T))
                    else:
                        novelty_event = np.nan
                    clusters.append([r])
                    cents.append(v)
                    frame_j = len(clusters) - 1
                    if r["kol"] in ol and (max_rank is None or rank <= max_rank):
                        m = meta.get(r["kol"], {})
                        h = hist.get(r["kol"], {})
                        row = {
                            "event_id": f"{sym}:{day}:thr{thr:.2f}:{origin_window['name']}",
                            "sym": sym,
                            "day": day,
                            "split": "train" if day < p5.MODEL_SPLIT else "val",
                            "thr": thr,
                            "origin_window": origin_window["name"],
                            "frame_j": frame_j,
                            "origin_kol": r["kol"],
                            "origin_text": r.get("text", ""),
                            "origin_ol": float(ol[r["kol"]]),
                            "origin_logfoll": float(m.get("log_followers", math.log1p(r["followers"]))),
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
                        }
                        row["ol_x_visibility"] = row["origin_ol"] * row["origin_logfoll"]
                        row["ol_x_novelty"] = row["origin_ol"] * (0.0 if not np.isfinite(row["novelty_global"]) else row["novelty_global"])
                        event_candidates.append(row)
                else:
                    clusters[best_j].append(r)
                    c = cents[best_j] * (len(clusters[best_j]) - 1) + v
                    cents[best_j] = c / (np.linalg.norm(c) + 1e-12)
            if not event_candidates:
                past_cents.extend(cents)
                continue
            # Fill labels after observing subsequent event diffusion. A candidate's
            # future adoption excludes its own origin tweet and includes later KOLs
            # assigned online to the same frame.
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
                "n_candidates": len(event_candidates),
                "event_kols": len(items),
            })
            past_cents.extend(cents)
        log(f"  thr={thr:.2f} {origin_window['name']:<7} {si:02d}/{len(p5.SYMS)} {sym:<5} events={used_events:>4} rows={len(panel):>6}")
    return panel, event_summaries


def train_scores(rows, target):
    scores = {}
    for name, feats in FEATURE_SETS.items():
        sc = p5.fit_ridge_scores(rows, feats, target)
        if sc is not None:
            scores[name] = sc
    return scores


def event_rows(rows, scores, target):
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r["split"] == "val" and np.isfinite(scores[i]):
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


def pooled_mean(evs, metric):
    vals = [e[metric] for e in evs if np.isfinite(e[metric])]
    return float(np.mean(vals)) if vals else np.nan


def symbal_mean(evs, metric):
    by = collections.defaultdict(list)
    for e in evs:
        if np.isfinite(e[metric]):
            by[e["sym"]].append(e[metric])
    vals = [float(np.mean(v)) for v in by.values() if v]
    return float(np.mean(vals)) if vals else np.nan


def aligned_pairs(a, b):
    aa = {e["event_id"]: e for e in a}
    bb = {e["event_id"]: e for e in b}
    return [(aa[k], bb[k]) for k in sorted(set(aa) & set(bb))]


def delta(a, b, metric):
    if metric == "js":
        return b[metric] - a[metric]
    return a[metric] - b[metric]


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


def bootstrap_pooled(pairs, metric):
    vals = np.array([delta(a, b, metric) for a, b in pairs], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"n_events": 0, **summarize([])}
    boots = []
    for _ in range(B):
        ii = RNG.integers(0, len(vals), len(vals))
        boots.append(float(vals[ii].mean()))
    out = summarize(boots)
    out["observed"] = float(vals.mean())
    out["n_events"] = int(len(vals))
    return out


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
    for _ in range(B):
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


def evaluate_global(rows, scores, target):
    y = np.array([r[target] for r in rows], dtype=float)
    train = np.array([r["split"] == "train" for r in rows], dtype=bool)
    val = ~train
    if train.sum() < 30 or val.sum() < 30:
        return None
    q90 = float(np.nanquantile(y[train], 0.90))
    ybin = (y >= q90).astype(int)
    return {
        "train_q90": q90,
        "n_train": int(train.sum()),
        "n_val": int(val.sum()),
        "val_positive_rate": float(ybin[val].mean()),
        "auc": p5.auc_score(ybin[val], np.asarray(scores)[val]),
        "ap": p5.average_precision(ybin[val], np.asarray(scores)[val]),
    }


def run_setting(rows, target):
    scores = train_scores(rows, target)
    evs = {name: event_rows(rows, sc, target) for name, sc in scores.items()}
    means = {}
    for name, ee in evs.items():
        means[name] = {
            "global_top10": evaluate_global(rows, scores[name], target),
            "n_events": len(ee),
            "n_symbols": len(set(e["sym"] for e in ee)),
            "pooled": {m: pooled_mean(ee, m) for m in METRICS},
            "symbol_balanced": {m: symbal_mean(ee, m) for m in METRICS},
        }
    comps = {}
    for model, base in COMPARISONS:
        if model not in evs or base not in evs:
            continue
        pairs = aligned_pairs(evs[model], evs[base])
        comps[f"{model}_vs_{base}"] = {
            m: {
                "pooled_bootstrap": bootstrap_pooled(pairs, m),
                "symbol_balanced_bootstrap": bootstrap_symbal(pairs, m),
            }
            for m in METRICS
        }
    return {"means": means, "comparisons": comps}


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
    meta = p5.compute_metadata(all_rows)
    ol = p5.compute_oltrait(all_rows, meta)

    result = {
        "task": "pre_popularity_origin_alert",
        "bootstrap_B": B,
        "thresholds": THRESHOLDS,
        "origin_windows": ORIGIN_WINDOWS,
        "targets": TARGETS,
        "metrics": METRICS,
        "feature_sets": FEATURE_SETS,
        "comparisons": COMPARISONS,
        "positive_delta_means": "left model better; for JS, delta=baseline_js-model_js",
        "n_ol_kols": len(ol),
        "by_setting": {},
    }

    log("[2/5] Building origin-only panels")
    for thr in THRESHOLDS:
        log(f"\n[history] threshold={thr:.2f}")
        hist = compute_origin_history(rows_by, emb_by, thr)
        for win in ORIGIN_WINDOWS:
            key = f"thr{thr:.2f}_{win['name']}"
            log(f"\n--- {key} ---")
            panel, events = build_origin_panel(rows_by, emb_by, meta, ol, hist, thr, win)
            out = {
                "n_rows": len(panel),
                "n_train_rows": sum(r["split"] == "train" for r in panel),
                "n_val_rows": sum(r["split"] == "val" for r in panel),
                "n_events": len(events),
                "targets": {},
            }
            for target in TARGETS:
                out["targets"][target] = run_setting(panel, target) if panel else {}
                comps = out["targets"][target].get("comparisons", {})
                k = "ol_origin_vs_no_ol_strong"
                if k in comps:
                    nd = comps[k]["ndcg3"]["symbol_balanced_bootstrap"]
                    js = comps[k]["js"]["symbol_balanced_bootstrap"]
                    gl = out["targets"][target].get("means", {}).get("ol_origin", {}).get("global_top10")
                    ap = gl.get("ap") if gl else np.nan
                    log(
                        f"  {target:<16} OL-vs-noOL symbal NDCG={nd.get('observed'):+.3f} "
                        f"CI[{nd.get('ci05'):+.3f},{nd.get('ci95'):+.3f}] | "
                        f"JS={js.get('observed'):+.3f} CI[{js.get('ci05'):+.3f},{js.get('ci95'):+.3f}] | "
                        f"OL global AP={ap:.3f}"
                    )
            result["by_setting"][key] = out

    log("\n[4/5] Aggregating robust wins")
    summary = collections.defaultdict(lambda: collections.defaultdict(int))
    totals = collections.defaultdict(int)
    for setting, s in result["by_setting"].items():
        for target, t in s.get("targets", {}).items():
            for cname, comp in t.get("comparisons", {}).items():
                for metric, blocks in comp.items():
                    b = blocks["symbol_balanced_bootstrap"]
                    lo = b.get("ci05")
                    hi = b.get("ci95")
                    if lo is None or hi is None:
                        continue
                    key = f"{target}:{cname}:{metric}"
                    totals[key] += 1
                    if lo > 0:
                        summary[key]["positive_ci"] += 1
                    elif hi < 0:
                        summary[key]["negative_ci"] += 1
                    else:
                        summary[key]["cross_zero"] += 1
    result["robust_win_summary_symbol_balanced"] = {
        k: {"n_settings": totals[k], **dict(v)}
        for k, v in sorted(summary.items())
    }
    result["elapsed_sec"] = time.time() - t0
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"[5/5] wrote {OUT}")
    log(f"elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
