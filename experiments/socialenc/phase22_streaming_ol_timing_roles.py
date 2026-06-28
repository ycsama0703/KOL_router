"""Phase22: OL timing-role upgrade for streaming agent memory.

This keeps the replacement Experiment 1 task fixed:
  score current tweet x_t using only prefix memory M_{t-1}.

The model remains linear. The only change is feature design: instead of a coarse
OL-memory block, we separate OL roles by timing and frame maturity.

Roles:
  - OL as new-frame originator;
  - OL as early participant in a young frame;
  - OL as late follower in a mature frame;
  - OL-originated frame memory.
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
import phase19_streaming_agent_memory as p19


OUT = pathlib.Path(__file__).with_name("phase22_streaming_ol_timing_roles_result.json")

THRESHOLDS = [0.55, 0.60, 0.65]
PREFIX_MIN_PRIOR = 9
EARLY_FRAME_K = 2


BASE_NO_OL = p19.CURRENT_KOL + p19.CURRENT_SENTIMENT + p19.HISTORY + p19.SEMANTIC_MEMORY

OL_CURRENT = [
    "current_ol",
]

OL_ORIGIN_ROLE = [
    "current_ol_x_new_frame",
    "matched_origin_ol",
    "matched_origin_ol_x_sim",
]

OL_EARLY_ROLE = [
    "matched_early_ol_mean",
    "matched_early_ol_max",
    "matched_early_ol_sum",
    "matched_first_ol_rank",
    "matched_first_ol_is_origin",
    "matched_has_early_ol",
    "current_ol_x_young_frame",
    "current_ol_x_recent_frame",
]

OL_LATE_ROLE = [
    "matched_late_ol_mean",
    "matched_late_ol_max",
    "matched_late_ol_sum",
    "matched_last_ol_rank",
    "matched_ol_arrival_span",
    "current_ol_x_existing_frame",
    "current_ol_x_frame_age",
    "current_ol_x_frame_recency",
    "current_ol_x_prior_count",
    "current_ol_x_prior_reach",
    "current_ol_x_mature_frame",
]

OL_MATURITY_CONTEXT = [
    "matched_max_ol",
    "matched_mean_ol",
    "matched_ol_sum",
    "memory_prior_ol_mean",
    "memory_prior_max_ol",
    "matched_origin_ol_x_prior_count",
    "matched_origin_ol_x_prior_reach",
    "matched_max_ol_x_prior_count",
    "matched_max_ol_x_prior_reach",
]

FEATURE_SETS = {
    "follower_current": p19.CURRENT_KOL,
    "sentiment_current": p19.CURRENT_SENTIMENT,
    "semantic_memory": p19.SEMANTIC_MEMORY,
    "no_ol_memory": BASE_NO_OL,
    "old_ol_memory": p19.FEATURE_SETS["ol_memory"],
    "ol_current_step": BASE_NO_OL + OL_CURRENT,
    "ol_origin_step": BASE_NO_OL + OL_CURRENT + OL_ORIGIN_ROLE,
    "ol_early_step": BASE_NO_OL + OL_CURRENT + OL_ORIGIN_ROLE + OL_EARLY_ROLE,
    "ol_timing_roles": (
        BASE_NO_OL
        + OL_CURRENT
        + OL_ORIGIN_ROLE
        + OL_EARLY_ROLE
        + OL_LATE_ROLE
        + OL_MATURITY_CONTEXT
    ),
}

COMPARISONS = [
    ("ol_timing_roles", "no_ol_memory"),
    ("ol_timing_roles", "old_ol_memory"),
    ("ol_timing_roles", "ol_early_step"),
    ("ol_early_step", "ol_origin_step"),
    ("ol_origin_step", "ol_current_step"),
    ("ol_current_step", "no_ol_memory"),
    ("old_ol_memory", "no_ol_memory"),
]


def log(message: str) -> None:
    print(message, flush=True)


def selected_ol_values(frame: dict, start: int = 0, stop: int | None = None) -> list[float]:
    return [float(x) for x in frame["ol_values"][start:stop]]


def ol_summary(values: list[float], prefix: str) -> dict:
    if not values:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_sum": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_sum": float(np.sum(values)),
    }


def timing_memory_summary(frames: list[dict]) -> dict:
    base = p19.memory_summary(frames)
    ol_values = []
    for frame in frames:
        ol_values.extend(frame["ol_values"])
    base["memory_prior_ol_mean"] = float(np.mean(ol_values)) if ol_values else 0.0
    base["memory_prior_max_ol"] = float(np.max(ol_values)) if ol_values else 0.0
    return base


def timing_matched_features(frames: list[dict], best_j: int, best_sim: float, rank: int) -> dict:
    out = p19.matched_frame_features(frames, best_j, best_sim, rank)
    extra_names = [
        "matched_early_ol_mean", "matched_early_ol_max", "matched_early_ol_sum",
        "matched_late_ol_mean", "matched_late_ol_max", "matched_late_ol_sum",
        "matched_first_ol_rank", "matched_last_ol_rank", "matched_ol_arrival_span",
        "matched_first_ol_is_origin", "matched_has_early_ol",
    ]
    if best_j < 0:
        out.update({name: 0.0 for name in extra_names})
        return out

    frame = frames[best_j]
    ol_values = [float(x) for x in frame["ol_values"]]
    early_values = selected_ol_values(frame, 0, min(EARLY_FRAME_K, len(ol_values)))
    late_values = selected_ol_values(frame, min(EARLY_FRAME_K, len(ol_values)), None)
    out.update(ol_summary(early_values, "matched_early_ol"))
    out.update(ol_summary(late_values, "matched_late_ol"))

    ranks = frame["ranks"]
    positive_positions = [i for i, value in enumerate(ol_values) if value > 0]
    if positive_positions:
        first_i = positive_positions[0]
        last_i = positive_positions[-1]
        first_rank = ranks[first_i]
        last_rank = ranks[last_i]
        out["matched_first_ol_rank"] = float(first_rank - frame["first_rank"])
        out["matched_last_ol_rank"] = float(last_rank - frame["first_rank"])
        out["matched_ol_arrival_span"] = float(last_rank - first_rank)
        out["matched_first_ol_is_origin"] = 1.0 if first_i == 0 else 0.0
        out["matched_has_early_ol"] = 1.0 if first_i < EARLY_FRAME_K else 0.0
    else:
        out["matched_first_ol_rank"] = 0.0
        out["matched_last_ol_rank"] = 0.0
        out["matched_ol_arrival_span"] = 0.0
        out["matched_first_ol_is_origin"] = 0.0
        out["matched_has_early_ol"] = 0.0
    return out


def update_timing_frame(frames: list[dict], frame_j: int, r: dict, v: np.ndarray, rank: int, current_ol: float) -> None:
    p19.update_frame(frames, frame_j, r, v, rank, current_ol)
    frames[frame_j].setdefault("ranks", [])
    if len(frames[frame_j]["ranks"]) < len(frames[frame_j]["ol_values"]):
        frames[frame_j]["ranks"].append(rank)


def add_interactions(row: dict) -> None:
    current_ol = row["current_ol"]
    is_new = row["is_new_frame"]
    is_existing = 1.0 - is_new
    age = row["matched_age_ranks"]
    recency = row["matched_recency_ranks"]
    prior_count = row["matched_prior_count"]
    prior_reach = row["matched_log_prior_reach"]
    sim = row["matched_sim"]

    row["current_ol_x_new_frame"] = current_ol * is_new
    row["current_ol_x_existing_frame"] = current_ol * is_existing
    row["current_ol_x_young_frame"] = current_ol * (1.0 if 0 < prior_count <= EARLY_FRAME_K else 0.0)
    row["current_ol_x_recent_frame"] = current_ol * (1.0 / (1.0 + max(recency, 0.0)))
    row["current_ol_x_frame_age"] = current_ol * math.log1p(max(age, 0.0))
    row["current_ol_x_frame_recency"] = current_ol * math.log1p(max(recency, 0.0))
    row["current_ol_x_prior_count"] = current_ol * math.log1p(max(prior_count, 0.0))
    row["current_ol_x_prior_reach"] = current_ol * prior_reach
    row["current_ol_x_mature_frame"] = current_ol * (1.0 if prior_count > EARLY_FRAME_K else 0.0)

    row["matched_origin_ol_x_sim"] = row["matched_origin_ol"] * sim
    row["matched_origin_ol_x_prior_count"] = row["matched_origin_ol"] * math.log1p(max(prior_count, 0.0))
    row["matched_origin_ol_x_prior_reach"] = row["matched_origin_ol"] * prior_reach
    row["matched_max_ol_x_prior_count"] = row["matched_max_ol"] * math.log1p(max(prior_count, 0.0))
    row["matched_max_ol_x_prior_reach"] = row["matched_max_ol"] * prior_reach

    # Keep old phase19 names so old_ol_memory is evaluated on the same rows.
    row["ol_x_matched_sim"] = current_ol * sim
    row["ol_x_prior_reach"] = current_ol * prior_reach


def build_timing_panel(rows_by, emb_by, meta, ol, hist, thr: float) -> tuple[list[dict], list[dict]]:
    panel = []
    event_summaries = []
    for si, sym in enumerate(p19.SYMS, 1):
        ev = p5.first_by_event(rows_by[sym], start=p5.TRAIN_END, end=p5.VAL_END)
        used_events = 0
        for (_, day), d in sorted(ev.items(), key=lambda kv: kv[0][1]):
            items = sorted(d.values(), key=lambda r: r["ts"])
            items = [r for r in items if r["kol"] in ol]
            if len(items) < p19.MIN_EVENT_KOLS:
                continue
            event_id = f"{sym}:{day}:thr{thr:.2f}:prefix{PREFIX_MIN_PRIOR}"
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
                for j, frame in enumerate(frames):
                    sim = float(v @ frame["centroid"])
                    if sim > best:
                        best = sim
                        best_j = j
                creates_new = not (best_j >= 0 and best >= thr)
                frame_j = len(frames) if creates_new else best_j
                current_ol = float(ol[r["kol"]])
                novelty_to_memory = 1.0 - max(0.0, best) if frames else 1.0

                if pos >= PREFIX_MIN_PRIOR:
                    m = meta.get(r["kol"], {})
                    h = hist.get(r["kol"], {})
                    row = {
                        "event_id": event_id,
                        "sym": sym,
                        "day": day,
                        "split": "train" if day < p5.MODEL_SPLIT else "val",
                        "thr": thr,
                        "min_prior": PREFIX_MIN_PRIOR,
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
                    row.update(timing_memory_summary(frames))
                    row.update(timing_matched_features(frames, -1 if creates_new else best_j, best, rank))
                    add_interactions(row)
                    stream_to_row[pos] = len(panel)
                    panel.append(row)

                stream_to_frame[pos] = frame_j
                update_timing_frame(frames, frame_j, r, v, rank, current_ol)

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
        log(f"  thr={thr:.2f} prefix>=9 {si:02d}/{len(p19.SYMS)} {sym:<5} events={used_events:>4} rows={len(panel):>7}")
    return panel, event_summaries


def train_scores(rows: list[dict], target: str) -> dict[str, np.ndarray]:
    scores = {}
    for name, features in FEATURE_SETS.items():
        score = p5.fit_ridge_scores(rows, features, target)
        if score is not None:
            scores[name] = score
    return scores


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
    for model, base in COMPARISONS:
        if model not in evs or base not in evs:
            continue
        pairs = p7.aligned_pairs(evs[model], evs[base])
        comparisons[f"{model}_vs_{base}"] = {
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


def main() -> None:
    t0 = time.time()
    log("[1/5] Loading tweets and MiniLM embeddings")
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

    result = {
        "task": "streaming_ol_timing_roles",
        "thresholds": THRESHOLDS,
        "prefix_min_prior": PREFIX_MIN_PRIOR,
        "early_frame_k": EARLY_FRAME_K,
        "targets": p19.TARGETS,
        "metrics": p19.METRICS,
        "feature_sets": FEATURE_SETS,
        "feature_blocks": {
            "base_no_ol": BASE_NO_OL,
            "ol_current": OL_CURRENT,
            "ol_origin_role": OL_ORIGIN_ROLE,
            "ol_early_role": OL_EARLY_ROLE,
            "ol_late_role": OL_LATE_ROLE,
            "ol_maturity_context": OL_MATURITY_CONTEXT,
        },
        "comparisons": COMPARISONS,
        "by_setting": {},
    }

    log("[2/5] Building timing-role panels")
    for threshold in THRESHOLDS:
        log(f"\n[history] threshold={threshold:.2f}")
        hist = p19.compute_origin_history(rows_by, emb_by, threshold)
        key = f"thr{threshold:.2f}_prefix{PREFIX_MIN_PRIOR}"
        panel, events = build_timing_panel(rows_by, emb_by, metadata, ol, hist, threshold)
        out = {
            "n_rows": len(panel),
            "n_train_rows": sum(row["split"] == "train" for row in panel),
            "n_val_rows": sum(row["split"] == "val" for row in panel),
            "n_events": len(events),
            "targets": {},
        }
        for target in p19.TARGETS:
            scores = train_scores(panel, target)
            out["targets"][target] = evaluate_scores(panel, scores, target)
            comp = out["targets"][target].get("comparisons", {}).get("ol_timing_roles_vs_no_ol_memory")
            if comp:
                nd = comp["ndcg3"]["symbol_balanced_bootstrap"]
                hit = comp["hit1"]["symbol_balanced_bootstrap"]
                log(
                    f"  {target:<16} timing-vs-noOL "
                    f"NDCG={nd.get('observed'):+.3f} CI[{nd.get('ci05'):+.3f},{nd.get('ci95'):+.3f}] | "
                    f"Hit1={hit.get('observed'):+.3f} CI[{hit.get('ci05'):+.3f},{hit.get('ci95'):+.3f}]"
                )
        result["by_setting"][key] = out

    log("[4/5] Aggregating main region")
    result["main_region_mean"] = aggregate_main_region(result["by_setting"])
    result["elapsed_sec"] = time.time() - t0
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"[5/5] wrote {OUT}")
    log(f"elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
