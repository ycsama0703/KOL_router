"""Streaming agent triage with rolling KOL narrative memory.

Replacement for the deprecated first10 agent-routing setup.

Application point:
  A financial research/trading agent has a prefix tweet memory for a symbol-day.
  A new KOL tweet arrives. Before seeing any later tweets, the agent decides
  whether this arrival deserves research, retrieval, summarization, or trading
  attention.

Leakage controls:
  - Each row's features are extracted before the current tweet updates memory.
  - Labels use only later tweets assigned online to the same semantic frame.
  - OLtrait is estimated only from pre-2020 behavior through phase5.
  - Models are chronological linear ridge scorers.

The main comparison is:
  No-OL structured memory vs OL structured memory.
"""
from __future__ import annotations

import collections
import json
import math
import os
import pathlib
import time

import numpy as np

import phase5_sentiment_reconstruction as p5
import phase7_origin_alert as p7


OUT = pathlib.Path(__file__).with_name("phase19_streaming_agent_memory_result.json")

DEFAULT_THRESHOLDS = [0.55, 0.60, 0.65]
DEFAULT_PREFIX_MIN_PRIORS = [1, 3, 9]
MIN_EVENT_KOLS = 8
MAIN_PREFIX_MIN_PRIOR = 9
RNG = np.random.default_rng(1919)


def parse_float_list(env_name: str, default: list[float]) -> list[float]:
    raw = os.getenv(env_name)
    if not raw:
        return default
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(env_name: str, default: list[int]) -> list[int]:
    raw = os.getenv(env_name)
    if not raw:
        return default
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def selected_syms() -> list[str]:
    raw = os.getenv("STREAM_SYMS")
    if not raw:
        return list(p5.SYMS)
    wanted = {x.strip() for x in raw.split(",") if x.strip()}
    return [s for s in p5.SYMS if s in wanted]


THRESHOLDS = parse_float_list("STREAM_THRESHOLDS", DEFAULT_THRESHOLDS)
PREFIX_MIN_PRIORS = parse_int_list("STREAM_PREFIX_MIN_PRIORS", DEFAULT_PREFIX_MIN_PRIORS)
SYMS = selected_syms()
MAX_EVENTS_PER_SYMBOL = int(os.getenv("STREAM_MAX_EVENTS_PER_SYMBOL", "0"))


CURRENT_KOL = [
    "current_logfoll", "current_verified", "log_stream_rank", "elapsed_hours",
]

CURRENT_SENTIMENT = [
    "current_stance", "current_stance_abs",
]

HISTORY = [
    "hist_log_origin_count", "hist_mean_log_adopt", "hist_success_rate",
]

SEMANTIC_MEMORY = [
    "matched_sim", "is_new_frame", "novelty_to_memory",
    "prior_total_tweets", "prior_frame_count", "dominant_frame_share",
    "frame_entropy", "matched_prior_count", "matched_log_prior_reach",
    "matched_age_ranks", "matched_recency_ranks",
]

OL_MEMORY = [
    "current_ol", "matched_origin_ol", "matched_max_ol", "matched_mean_ol",
    "matched_ol_sum", "memory_prior_ol_mean", "memory_prior_max_ol",
    "ol_x_matched_sim", "ol_x_prior_reach",
]

FEATURE_SETS = {
    "follower_current": CURRENT_KOL,
    "sentiment_current": CURRENT_SENTIMENT,
    "semantic_memory": SEMANTIC_MEMORY,
    "no_ol_memory": CURRENT_KOL + CURRENT_SENTIMENT + HISTORY + SEMANTIC_MEMORY,
    "ol_current": CURRENT_KOL + ["current_ol"],
    "ol_memory": CURRENT_KOL + CURRENT_SENTIMENT + HISTORY + SEMANTIC_MEMORY + OL_MEMORY,
}

COMPARISONS = [
    ("ol_memory", "no_ol_memory"),
    ("ol_memory", "semantic_memory"),
    ("ol_memory", "follower_current"),
    ("ol_memory", "sentiment_current"),
    ("ol_memory", "ol_current"),
    ("no_ol_memory", "semantic_memory"),
]

TARGETS = ["log_future_adopt", "log_future_reach"]
METRICS = ["ndcg3", "hit1", "mass3", "js"]


def log(message: str) -> None:
    print(message, flush=True)


def safe_entropy(counts: list[float]) -> float:
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    p = np.asarray(counts, dtype=float) / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def memory_summary(frames: list[dict]) -> dict:
    if not frames:
        return {
            "prior_total_tweets": 0.0,
            "prior_frame_count": 0.0,
            "dominant_frame_share": 0.0,
            "frame_entropy": 0.0,
            "memory_prior_ol_mean": 0.0,
            "memory_prior_max_ol": 0.0,
        }
    counts = [float(f["count"]) for f in frames]
    total = float(sum(counts))
    ol_values = []
    for f in frames:
        ol_values.extend(f["ol_values"])
    return {
        "prior_total_tweets": total,
        "prior_frame_count": float(len(frames)),
        "dominant_frame_share": max(counts) / max(total, 1.0),
        "frame_entropy": safe_entropy(counts),
        "memory_prior_ol_mean": float(np.mean(ol_values)) if ol_values else 0.0,
        "memory_prior_max_ol": float(np.max(ol_values)) if ol_values else 0.0,
    }


def matched_frame_features(frames: list[dict], best_j: int, best_sim: float, rank: int) -> dict:
    if best_j < 0:
        return {
            "matched_sim": 0.0,
            "is_new_frame": 1.0,
            "matched_prior_count": 0.0,
            "matched_log_prior_reach": 0.0,
            "matched_age_ranks": 0.0,
            "matched_recency_ranks": 0.0,
            "matched_origin_ol": 0.0,
            "matched_max_ol": 0.0,
            "matched_mean_ol": 0.0,
            "matched_ol_sum": 0.0,
        }
    f = frames[best_j]
    ol_values = f["ol_values"]
    return {
        "matched_sim": float(best_sim),
        "is_new_frame": 0.0,
        "matched_prior_count": float(f["count"]),
        "matched_log_prior_reach": math.log1p(float(f["follower_mass"])),
        "matched_age_ranks": float(rank - f["first_rank"]),
        "matched_recency_ranks": float(rank - f["last_rank"]),
        "matched_origin_ol": float(f["origin_ol"]),
        "matched_max_ol": float(max(ol_values)) if ol_values else 0.0,
        "matched_mean_ol": float(np.mean(ol_values)) if ol_values else 0.0,
        "matched_ol_sum": float(sum(ol_values)),
    }


def update_frame(frames: list[dict], frame_j: int, r: dict, v: np.ndarray, rank: int, current_ol: float) -> None:
    if frame_j == len(frames):
        frames.append({
            "centroid": v,
            "count": 1,
            "follower_mass": max(0.0, float(r["followers"])),
            "origin_ol": float(current_ol),
            "ol_values": [float(current_ol)],
            "first_rank": rank,
            "last_rank": rank,
            "members": [rank - 1],
        })
        return
    f = frames[frame_j]
    c = f["centroid"] * f["count"] + v
    f["count"] += 1
    f["centroid"] = c / (np.linalg.norm(c) + 1e-12)
    f["follower_mass"] += max(0.0, float(r["followers"]))
    f["ol_values"].append(float(current_ol))
    f["last_rank"] = rank
    f["members"].append(rank - 1)


def compute_origin_history(rows_by, emb_by, thr: float) -> dict:
    """Pre-2020 KOL origin history over the selected symbol set."""
    stats = collections.defaultdict(lambda: {
        "n_origin": 0.0,
        "sum_log_adopt": 0.0,
        "n_success": 0.0,
        "sum_log_reach": 0.0,
    })
    for sym in SYMS:
        ev = p5.first_by_event(rows_by[sym], end=p5.TRAIN_END)
        for (_, _day), d in ev.items():
            items = sorted(d.values(), key=lambda r: r["ts"])
            if len(items) < MIN_EVENT_KOLS:
                continue
            clusters, _cents, _assign, _new_flags, _sims = p7.online_cluster(items, emb_by[sym], thr)
            for cl in clusters:
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


def build_streaming_panel(rows_by, emb_by, meta, ol, hist, thr: float, min_prior: int) -> tuple[list[dict], list[dict]]:
    panel = []
    event_summaries = []
    for si, sym in enumerate(SYMS, 1):
        ev = p5.first_by_event(rows_by[sym], start=p5.TRAIN_END, end=p5.VAL_END)
        used_events = 0
        for (_, day), d in sorted(ev.items(), key=lambda kv: kv[0][1]):
            if MAX_EVENTS_PER_SYMBOL and used_events >= MAX_EVENTS_PER_SYMBOL:
                break
            items = sorted(d.values(), key=lambda r: r["ts"])
            items = [r for r in items if r["kol"] in ol]
            if len(items) < MIN_EVENT_KOLS:
                continue
            event_id = f"{sym}:{day}:thr{thr:.2f}:prefix{min_prior}"
            event_start = items[0]["ts"]
            frames: list[dict] = []
            stream_to_frame = {}
            stream_to_row = {}
            event_row_start = len(panel)
            for pos, r in enumerate(items):
                rank = pos + 1
                v = p5.norm_vec(emb_by[sym], r["idx"])
                best = -1.0
                best_j = -1
                for j, f in enumerate(frames):
                    sim = float(v @ f["centroid"])
                    if sim > best:
                        best = sim
                        best_j = j
                creates_new = not (best_j >= 0 and best >= thr)
                frame_j = len(frames) if creates_new else best_j
                current_ol = float(ol[r["kol"]])
                if frames:
                    novelty_to_memory = 1.0 - max(0.0, best)
                else:
                    novelty_to_memory = 1.0

                if pos >= min_prior:
                    m = meta.get(r["kol"], {})
                    h = hist.get(r["kol"], {})
                    row = {
                        "event_id": event_id,
                        "sym": sym,
                        "day": day,
                        "split": "train" if day < p5.MODEL_SPLIT else "val",
                        "thr": thr,
                        "min_prior": min_prior,
                        "stream_rank": rank,
                        "current_kol": r["kol"],
                        "current_idx": int(r["idx"]),
                        "current_text": r.get("text", ""),
                        "current_ol": current_ol,
                        "current_logfoll": float(m.get("log_followers", math.log1p(r["followers"]))),
                        "current_verified": float(m.get("verified", r["verified"])),
                        "log_stream_rank": math.log(rank),
                        "elapsed_hours": (r["ts"] - event_start) / 3600.0,
                        "current_stance": float(r["stance"]),
                        "current_stance_abs": abs(float(r["stance"])),
                        "novelty_to_memory": novelty_to_memory,
                        "hist_log_origin_count": float(h.get("hist_log_origin_count", 0.0)),
                        "hist_mean_log_adopt": float(h.get("hist_mean_log_adopt", 0.0)),
                        "hist_success_rate": float(h.get("hist_success_rate", 0.0)),
                        "frame_j": frame_j,
                        "future_adopt": 0.0,
                        "future_reach": 0.0,
                    }
                    row.update(memory_summary(frames))
                    row.update(matched_frame_features(frames, -1 if creates_new else best_j, best, rank))
                    row["ol_x_matched_sim"] = row["current_ol"] * row["matched_sim"]
                    row["ol_x_prior_reach"] = row["current_ol"] * row["matched_log_prior_reach"]
                    stream_to_row[pos] = len(panel)
                    panel.append(row)

                stream_to_frame[pos] = frame_j
                update_frame(frames, frame_j, r, v, rank, current_ol)

            if len(panel) == event_row_start:
                continue

            frame_members = collections.defaultdict(list)
            for stream_i, frame_j in stream_to_frame.items():
                frame_members[frame_j].append(stream_i)
            for members in frame_members.values():
                members.sort()
                suffix_reach = 0.0
                suffix_count = 0
                for stream_i in reversed(members):
                    row_i = stream_to_row.get(stream_i)
                    if row_i is not None:
                        panel[row_i]["future_adopt"] = float(suffix_count)
                        panel[row_i]["future_reach"] = float(suffix_reach)
                        panel[row_i]["log_future_adopt"] = math.log1p(float(suffix_count))
                        panel[row_i]["log_future_reach"] = math.log1p(float(suffix_reach))
                    suffix_count += 1
                    suffix_reach += max(0.0, float(items[stream_i]["followers"]))

            used_events += 1
            event_summaries.append({
                "event_id": event_id,
                "sym": sym,
                "day": day,
                "n_stream_items": len(items),
                "n_candidate_rows": len(panel) - event_row_start,
                "n_final_frames": len(frames),
            })
        log(
            f"  thr={thr:.2f} prefix>={min_prior:<2} {si:02d}/{len(SYMS)} "
            f"{sym:<5} events={used_events:>4} rows={len(panel):>7}"
        )
    return panel, event_summaries


def train_scores(rows: list[dict], target: str) -> dict[str, np.ndarray]:
    scores = {}
    for name, feats in FEATURE_SETS.items():
        sc = p5.fit_ridge_scores(rows, feats, target)
        if sc is not None:
            scores[name] = sc
    return scores


def run_setting(rows: list[dict], target: str) -> dict:
    scores = train_scores(rows, target)
    evs = {name: p7.event_rows(rows, sc, target) for name, sc in scores.items()}
    means = {}
    for name, ee in evs.items():
        means[name] = {
            "global_top10": p7.evaluate_global(rows, scores[name], target),
            "n_events": len(ee),
            "n_symbols": len(set(e["sym"] for e in ee)),
            "pooled": {m: p7.pooled_mean(ee, m) for m in METRICS},
            "symbol_balanced": {m: p7.symbal_mean(ee, m) for m in METRICS},
        }
    comps = {}
    for model, base in COMPARISONS:
        if model not in evs or base not in evs:
            continue
        pairs = p7.aligned_pairs(evs[model], evs[base])
        comps[f"{model}_vs_{base}"] = {
            m: {
                "pooled_bootstrap": p7.bootstrap_pooled(pairs, m),
                "symbol_balanced_bootstrap": p7.bootstrap_symbal(pairs, m),
            }
            for m in METRICS
        }
    return {"means": means, "comparisons": comps}


def aggregate_main_region(by_setting: dict) -> dict:
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for key, setting in by_setting.items():
        if f"prefix{MAIN_PREFIX_MIN_PRIOR}" not in key:
            continue
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
    t0 = time.time()
    log("[1/5] Loading tweets and MiniLM embeddings")
    rows_by = {}
    emb_by = {}
    all_rows = []
    for i, sym in enumerate(SYMS, 1):
        rows, emb = p5.load_symbol(sym)
        rows_by[sym] = rows
        emb_by[sym] = emb
        all_rows.extend(rows)
        log(f"  {i:02d}/{len(SYMS)} {sym:<5} rows={len(rows):>6}")
    meta = p5.compute_metadata(all_rows)
    ol = p5.compute_oltrait(all_rows, meta)

    result = {
        "task": "streaming_agent_memory_triage",
        "deprecated_replaces": "static first10 agent-routing Experiment 1",
        "thresholds": THRESHOLDS,
        "prefix_min_priors": PREFIX_MIN_PRIORS,
        "main_prefix_min_prior": MAIN_PREFIX_MIN_PRIOR,
        "symbols": SYMS,
        "max_events_per_symbol": MAX_EVENTS_PER_SYMBOL,
        "targets": TARGETS,
        "metrics": METRICS,
        "feature_sets": FEATURE_SETS,
        "comparisons": COMPARISONS,
        "n_ol_kols": len(ol),
        "by_setting": {},
    }

    log("[2/5] Building streaming memory panels")
    for thr in THRESHOLDS:
        log(f"\n[history] threshold={thr:.2f}")
        hist = compute_origin_history(rows_by, emb_by, thr)
        for min_prior in PREFIX_MIN_PRIORS:
            key = f"thr{thr:.2f}_prefix{min_prior}"
            log(f"\n--- {key} ---")
            panel, events = build_streaming_panel(rows_by, emb_by, meta, ol, hist, thr, min_prior)
            out = {
                "n_rows": len(panel),
                "n_train_rows": sum(r["split"] == "train" for r in panel),
                "n_val_rows": sum(r["split"] == "val" for r in panel),
                "n_events": len(events),
                "targets": {},
            }
            for target in TARGETS:
                out["targets"][target] = run_setting(panel, target) if panel else {}
                comp = out["targets"][target].get("comparisons", {}).get("ol_memory_vs_no_ol_memory", {})
                if comp:
                    nd = comp["ndcg3"]["symbol_balanced_bootstrap"]
                    hit = comp["hit1"]["symbol_balanced_bootstrap"]
                    log(
                        f"  {target:<16} OL-vs-noOL symbal "
                        f"NDCG={nd.get('observed'):+.3f} CI[{nd.get('ci05'):+.3f},{nd.get('ci95'):+.3f}] | "
                        f"Hit1={hit.get('observed'):+.3f} CI[{hit.get('ci05'):+.3f},{hit.get('ci95'):+.3f}]"
                    )
            result["by_setting"][key] = out

    log("\n[4/5] Aggregating main prefix region")
    result["main_region_mean"] = aggregate_main_region(result["by_setting"])
    result["elapsed_sec"] = time.time() - t0
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"[5/5] wrote {OUT}")
    log(f"elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
