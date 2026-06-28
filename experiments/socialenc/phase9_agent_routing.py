"""Experiment A: Agent attention / RAG routing over early KOL frames.

Application point:
  A financial research/trading agent has observed an early window of KOL tweets
  and has a limited token budget. It must choose Top-K semantic frames to read,
  summarize, or put into memory.

This experiment asks:
  Which routing policy best covers the frames that later become dominant in
  future adoption or follower-weighted reach?

Unlike pre-popularity origin alert, this task allows early popularity features
because the early window is already visible to the agent.
"""
from __future__ import annotations

import collections
import json
import pathlib
import time

import numpy as np

import phase5_sentiment_reconstruction as p5

OUT = pathlib.Path(__file__).with_name("phase9_agent_routing_result.json")

B = 600
RNG = np.random.default_rng(909)
KS = [1, 2, 3, 5]

RAW_ROUTERS = [
    "earliest_origin",
    "follower_first",
    "verified_first",
    "sentiment_strength",
    "bullish_first",
    "novelty_first",
    "early_adopt",
    "early_reach",
    "early_share",
]

LEARNED_ROUTERS = {
    "learned_early_pop": p5.FEATURE_SETS["early_pop"],
    "learned_pop_sent": p5.FEATURE_SETS["early_pop_sentiment"],
    "learned_semantic_no_kol": p5.FEATURE_SETS["semantic_no_kol"],
    "learned_full": p5.FEATURE_SETS["full"],
}

COMPARISONS = [
    ("learned_full", "earliest_origin"),
    ("learned_full", "follower_first"),
    ("learned_full", "verified_first"),
    ("learned_full", "sentiment_strength"),
    ("learned_full", "novelty_first"),
    ("learned_full", "early_adopt"),
    ("learned_full", "early_reach"),
    ("learned_full", "learned_early_pop"),
    ("learned_full", "learned_pop_sent"),
    ("learned_full", "learned_semantic_no_kol"),
    ("learned_early_pop", "sentiment_strength"),
    ("learned_early_pop", "follower_first"),
    ("learned_pop_sent", "learned_early_pop"),
]

TARGETS = {
    "future_adopt": "log_future_adopt",
    "future_reach": "log_future_reach",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def clean_score(vals):
    vals = np.asarray(vals, dtype=float)
    finite = vals[np.isfinite(vals)]
    fill = float(np.nanmedian(finite)) if len(finite) else 0.0
    return np.where(np.isfinite(vals), vals, fill)


def raw_scores(rows, router):
    vals = []
    for r in rows:
        if router == "earliest_origin":
            vals.append(-r["origin_rank_frac"])
        elif router == "follower_first":
            vals.append(r["origin_logfoll"])
        elif router == "verified_first":
            vals.append(r["origin_verified"] + 1e-3 * r["origin_logfoll"])
        elif router == "sentiment_strength":
            vals.append(r["frame_stance_abs"])
        elif router == "bullish_first":
            vals.append(r["frame_stance_mean"])
        elif router == "novelty_first":
            vals.append(r["novelty"])
        elif router == "early_adopt":
            vals.append(r["log_early_adopt"])
        elif router == "early_reach":
            vals.append(r["log_early_reach"])
        elif router == "early_share":
            vals.append(r["early_frame_share"])
        else:
            vals.append(0.0)
    return clean_score(vals)


def score_all_routers(rows, train_target):
    scores = {}
    for name in RAW_ROUTERS:
        scores[name] = raw_scores(rows, name)
    for name, feats in LEARNED_ROUTERS.items():
        sc = p5.fit_ridge_scores(rows, feats, train_target)
        if sc is not None:
            scores[name] = sc
    return scores


def dcg(vals):
    vals = np.asarray(vals, dtype=float)
    denom = np.log2(np.arange(2, len(vals) + 2))
    return float((vals / denom).sum()) if len(vals) else 0.0


def event_metrics_for_scores(rows, scores, eval_target):
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r["split"] == "val" and np.isfinite(scores[i]):
            groups[r["event_id"]].append((r["sym"], float(r[eval_target]), float(scores[i])))
    out = []
    for event_id, vals in groups.items():
        if len(vals) < 2:
            continue
        y = np.array([v[1] for v in vals], dtype=float)
        s = np.array([v[2] for v in vals], dtype=float)
        if not np.isfinite(y).all() or not np.isfinite(s).all() or np.nanmax(y) <= 0 or y.sum() <= 0:
            continue
        # Keep tied router scores reproducible across NumPy versions. Frames are
        # constructed in point-in-time origination order, which is the fixed
        # secondary key for otherwise identical scores.
        order = np.argsort(-s, kind="stable")
        ideal = np.argsort(-y, kind="stable")
        best = set(np.where(y == np.nanmax(y))[0].tolist())
        row = {
            "event_id": event_id,
            "sym": vals[0][0],
            "n_frames": int(len(vals)),
            "js": float(p5.js_divergence(y / max(float(y.sum()), 1e-12), p5.softmax(s))),
        }
        for k0 in KS:
            k = min(k0, len(y))
            row[f"ndcg@{k0}"] = float(dcg(y[order[:k]]) / max(dcg(y[ideal[:k]]), 1e-12))
            row[f"hit@{k0}"] = float(1.0 if any(i in best for i in order[:k]) else 0.0)
            row[f"mass@{k0}"] = float(y[order[:k]].sum() / max(float(y.sum()), 1e-12))
        out.append(row)
    return out


def mean_metric(evs, metric):
    vals = [e[metric] for e in evs if np.isfinite(e[metric])]
    return float(np.mean(vals)) if vals else np.nan


def symbal_metric(evs, metric):
    by = collections.defaultdict(list)
    for e in evs:
        if np.isfinite(e[metric]):
            by[e["sym"]].append(e[metric])
    vals = [float(np.mean(v)) for v in by.values() if v]
    return float(np.mean(vals)) if vals else np.nan


def aligned(a, b):
    aa = {e["event_id"]: e for e in a}
    bb = {e["event_id"]: e for e in b}
    return [(aa[k], bb[k]) for k in sorted(set(aa) & set(bb))]


def metric_delta(a, b, metric):
    if metric == "js":
        return b[metric] - a[metric]
    return a[metric] - b[metric]


def summarize(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"observed": None, "ci05": None, "ci95": None}
    return {
        "observed": float(vals.mean()),
        "ci05": float(np.quantile(vals, 0.05)),
        "ci95": float(np.quantile(vals, 0.95)),
    }


def bootstrap_symbal(pairs, metric):
    by = collections.defaultdict(list)
    for a, b in pairs:
        d = metric_delta(a, b, metric)
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


def bootstrap_pooled(pairs, metric):
    vals = np.array([metric_delta(a, b, metric) for a, b in pairs], dtype=float)
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


def evaluate_setting(rows, train_target, eval_target):
    scores = score_all_routers(rows, train_target)
    evs = {name: event_metrics_for_scores(rows, sc, eval_target) for name, sc in scores.items()}
    metric_names = ["js"]
    for k in KS:
        metric_names.extend([f"ndcg@{k}", f"hit@{k}", f"mass@{k}"])
    means = {}
    for name, ee in evs.items():
        means[name] = {
            "n_events": len(ee),
            "n_symbols": len(set(e["sym"] for e in ee)),
            "pooled": {m: mean_metric(ee, m) for m in metric_names},
            "symbol_balanced": {m: symbal_metric(ee, m) for m in metric_names},
        }
    comps = {}
    for model, base in COMPARISONS:
        if model not in evs or base not in evs:
            continue
        pairs = aligned(evs[model], evs[base])
        comps[f"{model}_vs_{base}"] = {
            m: {
                "pooled_bootstrap": bootstrap_pooled(pairs, m),
                "symbol_balanced_bootstrap": bootstrap_symbal(pairs, m),
            }
            for m in metric_names
        }
    return {"means": means, "comparisons": comps}


def main():
    t0 = time.time()
    log("[1/5] Loading tweets and MiniLM embeddings")
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
        "task": "agent_attention_rag_routing",
        "bootstrap_B": B,
        "thresholds": p5.THRESHOLDS,
        "early_modes": p5.EARLY_MODES,
        "ks": KS,
        "raw_routers": RAW_ROUTERS,
        "learned_routers": {k: v for k, v in LEARNED_ROUTERS.items()},
        "comparisons": COMPARISONS,
        "targets": TARGETS,
        "positive_delta_means": "left router better; for JS, delta=baseline_js-router_js",
        "n_ol_kols": len(ol),
        "by_setting": {},
    }

    log("[3/5] Building early-frame panels and evaluating routers")
    for thr in p5.THRESHOLDS:
        for mode in p5.EARLY_MODES:
            key = f"thr{thr:.2f}_{mode['name']}"
            log(f"\n--- {key} ---")
            rows, events = p5.build_panel_for(rows_by, emb_by, meta, ol, thr, mode)
            out = {
                "n_rows": len(rows),
                "n_train_rows": sum(r["split"] == "train" for r in rows),
                "n_val_rows": sum(r["split"] == "val" for r in rows),
                "n_events": len(events),
                "targets": {},
            }
            for label, train_target in TARGETS.items():
                eval_target = "future_adopt" if label == "future_adopt" else "future_reach"
                out["targets"][label] = evaluate_setting(rows, train_target, eval_target) if rows else {}
                comp = out["targets"][label].get("comparisons", {}).get("learned_full_vs_early_reach", {})
                comp_pop = out["targets"][label].get("comparisons", {}).get("learned_full_vs_learned_early_pop", {})
                comp_sent = out["targets"][label].get("comparisons", {}).get("learned_early_pop_vs_sentiment_strength", {})
                if comp:
                    a = comp["ndcg@3"]["symbol_balanced_bootstrap"]
                    b = comp_pop.get("ndcg@3", {}).get("symbol_balanced_bootstrap", {})
                    c = comp_sent.get("ndcg@3", {}).get("symbol_balanced_bootstrap", {})
                    log(
                        f"  {label:<12} NDCG@3 delta: "
                        f"full-earlyReach={a.get('observed'):+.3f} CI[{a.get('ci05'):+.3f},{a.get('ci95'):+.3f}] | "
                        f"full-learnedPop={b.get('observed'):+.3f} CI[{b.get('ci05'):+.3f},{b.get('ci95'):+.3f}] | "
                        f"learnedPop-sent={c.get('observed'):+.3f} CI[{c.get('ci05'):+.3f},{c.get('ci95'):+.3f}]"
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
