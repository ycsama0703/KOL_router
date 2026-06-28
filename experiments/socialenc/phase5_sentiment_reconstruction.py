"""Sentiment baselines and narrative reconstruction for KOL frame diffusion.

This is the agent-facing task: given only early, point-in-time KOL tweets inside
a symbol-day event, predict which already-seeded semantic frames will be adopted
later in the same event.

Leakage controls:
  - OLtrait is estimated only on pre-2020 events.
  - Model train is 2020-01-01..2021-06-01; validation is 2021-06-01..2022-06-01.
  - Seed frames are clustered from early tweets only.
  - Later tweets are mapped back to early seed frames only after the cutoff.
  - Novelty uses only prior events for the same symbol.

This phase extends phase4 with:
  1) explicit traditional-sentiment baselines;
  2) final narrative distribution reconstruction metrics.

The script intentionally avoids sklearn so it can run on the luyao4 environment.
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import re
import time
from datetime import datetime

import numpy as np

DATA = pathlib.Path(__file__).resolve().parents[2] / "data/socialenc"
OUT = pathlib.Path(__file__).with_name("phase5_sentiment_reconstruction_result.json")

SYMS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "AMD", "MSTR",
    "COIN", "HOOD", "PLTR", "SPY", "QQQ", "BTC", "ETH", "SOL",
]
TRAIN_END = "2020-01-01"
MODEL_SPLIT = "2021-06-01"
VAL_END = "2022-06-01"

THRESHOLDS = [0.45, 0.55, 0.65]
EARLY_MODES = [
    {"name": "first3", "kind": "count", "value": 3},
    {"name": "first5", "kind": "count", "value": 5},
    {"name": "first10", "kind": "count", "value": 10},
    {"name": "first2h", "kind": "hours", "value": 2.0},
    {"name": "first6h", "kind": "hours", "value": 6.0},
]
MIN_EVENT_KOLS = 8
MIN_EARLY_KOLS = 3
MIN_FUTURE_KOLS = 2

rng = np.random.default_rng(404)

BULL = {
    "buy", "bought", "long", "bull", "bullish", "breakout", "upside", "rally",
    "rip", "moon", "strong", "strength", "beat", "beats", "raise", "upgrade",
    "outperform", "undervalued", "support", "bounce", "squeeze", "positive",
    "add", "adding",
}
BEAR = {
    "sell", "sold", "short", "bear", "bearish", "breakdown", "downside",
    "crash", "crushed", "fraud", "bubble", "weak", "weakness", "miss",
    "misses", "downgrade", "underperform", "overvalued", "resistance",
    "lawsuit", "probe", "investigation", "warning", "caution", "risk",
    "negative", "avoid", "scam", "bankrupt",
}

FEATURE_SETS = {
    "sentiment_event": [
        "event_stance_mean", "event_stance_abs", "event_bull_share",
        "event_bear_share", "event_sentiment_disagreement",
        "log_event_early_kols",
    ],
    "sentiment_frame": [
        "origin_stance", "frame_stance_mean", "frame_stance_abs",
        "frame_bull_share", "frame_bear_share", "follower_weighted_stance",
        "sentiment_alignment", "log_event_early_kols",
    ],
    "sentiment_all": [
        "origin_stance", "frame_stance_mean", "frame_stance_abs",
        "frame_bull_share", "frame_bear_share", "follower_weighted_stance",
        "sentiment_alignment", "event_stance_mean", "event_stance_abs",
        "event_bull_share", "event_bear_share", "event_sentiment_disagreement",
        "log_event_early_kols",
    ],
    "early_pop": [
        "log_early_adopt", "log_early_reach", "early_frame_share",
        "origin_rank_frac", "log_event_early_kols",
    ],
    "early_pop_sentiment": [
        "log_early_adopt", "log_early_reach", "early_frame_share",
        "origin_rank_frac", "origin_stance", "frame_stance_mean",
        "frame_stance_abs", "frame_bull_share", "frame_bear_share",
        "follower_weighted_stance", "event_sentiment_disagreement",
        "log_event_early_kols",
    ],
    "origin_visibility": [
        "origin_logfoll", "origin_verified", "origin_rank_frac",
        "log_event_early_kols",
    ],
    "ol_only": ["origin_ol", "log_event_early_kols"],
    "ol_visibility": [
        "origin_ol", "origin_logfoll", "origin_verified", "origin_rank_frac",
        "log_event_early_kols",
    ],
    "semantic": [
        "novelty", "cohesion", "stance_abs", "stance_pos", "stance_neg",
        "log_event_early_kols",
    ],
    "semantic_no_kol": [
        "log_early_adopt", "log_early_reach", "early_frame_share",
        "origin_rank_frac", "novelty", "cohesion", "origin_stance",
        "frame_stance_mean", "frame_stance_abs", "event_sentiment_disagreement",
        "log_event_early_kols", "cutoff_elapsed_hours",
    ],
    "full": [
        "log_early_adopt", "log_early_reach", "early_frame_share",
        "origin_rank_frac", "origin_ol", "origin_logfoll", "origin_verified",
        "novelty", "cohesion", "stance_abs", "stance_pos", "stance_neg",
        "origin_stance", "frame_stance_mean", "frame_stance_abs",
        "frame_bull_share", "frame_bear_share", "follower_weighted_stance",
        "event_stance_mean", "event_stance_abs", "event_sentiment_disagreement",
        "ol_x_visibility", "ol_x_novelty", "log_event_early_kols",
        "cutoff_elapsed_hours",
    ],
}


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_ts(s: str) -> float:
    return datetime.strptime(s.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S%z").timestamp()


def stance(text: str) -> int:
    t = " " + re.sub(r"https?://\S+|@\w+", " ", (text or "").lower()) + " "
    b = sum(1 for w in BULL if w in t)
    a = sum(1 for w in BEAR if w in t)
    if b >= a + 1 and b > 0:
        return 1
    if a >= b + 1 and a > 0:
        return -1
    return 0


def load_symbol(sym: str):
    z = np.load(DATA / f"{sym}.npz", allow_pickle=False)
    emb = z["emb"].astype(np.float32)
    rows = []
    with open(DATA / f"{sym}.jsonl", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            ca = r.get("created_at")
            k = r.get("kol_username")
            if not ca or not k:
                continue
            try:
                ts = parse_ts(ca)
            except Exception:
                continue
            fo = r.get("author_followers")
            rows.append({
                "sym": sym,
                "idx": idx,
                "kol": k,
                "ts": ts,
                "day": ca[:10],
                "hour": int(ca[11:13]),
                "followers": float(fo) if isinstance(fo, (int, float)) else 0.0,
                "verified": 1.0 if r.get("author_verified") else 0.0,
                "stance": stance(r.get("text") or ""),
                "text": r.get("text") or "",
            })
    return rows, emb


def first_by_event(rows, start=None, end=None):
    ev = collections.defaultdict(dict)
    for r in rows:
        if start and r["day"] < start:
            continue
        if end and r["day"] >= end:
            continue
        d = ev[(r["sym"], r["day"])]
        if r["kol"] not in d or r["ts"] < d[r["kol"]]["ts"]:
            d[r["kol"]] = r
    return ev


def compute_metadata(all_rows):
    hours = collections.defaultdict(list)
    folls = collections.defaultdict(list)
    verified = collections.defaultdict(list)
    for r in all_rows:
        hours[r["kol"]].append(r["hour"])
        verified[r["kol"]].append(r["verified"])
        if r["followers"] > 0:
            folls[r["kol"]].append(r["followers"])
    meta = {}
    for k, hs in hours.items():
        mf = float(np.median(folls[k])) if folls[k] else 0.0
        meta[k] = {
            "medhour": float(np.median(hs)),
            "med_followers": mf,
            "log_followers": float(np.log1p(mf)),
            "verified": float(np.max(verified[k])) if verified[k] else 0.0,
        }
    return meta


def compute_oltrait(all_rows, meta):
    log("[2/5] Computing pre-2020 deconfounded OLtrait")
    ev = first_by_event(all_rows, end=TRAIN_END)
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
    lraw = {k: s[k] / n[k] for k in s if n[k] >= 4 and k in meta}
    ks = list(lraw)
    mh = np.array([meta[k]["medhour"] for k in ks], dtype=float)
    lv = np.array([lraw[k] for k in ks], dtype=float)
    A = np.vstack([np.ones_like(mh), mh, mh**2]).T
    c, *_ = np.linalg.lstsq(A, lv, rcond=None)
    fit = A @ c
    ol = {k: float(lv[i] - fit[i]) for i, k in enumerate(ks)}
    corr = np.corrcoef([ol[k] for k in ks], mh)[0, 1] if len(ks) > 2 else np.nan
    log(f"  OL KOLs={len(ol)} corr(OL,hour)={corr:+.3f}")
    return ol


def norm_vec(emb, idx):
    v = emb[idx].astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def greedy_seed_frames(items, emb, thr):
    clusters = []
    cents = []
    sims = []
    for r in items:
        v = norm_vec(emb, r["idx"])
        best = -1.0
        best_j = -1
        for j, c in enumerate(cents):
            sim = float(v @ c)
            if sim > best:
                best = sim
                best_j = j
        if best_j >= 0 and best >= thr:
            clusters[best_j].append(r)
            sims[best_j].append(best)
            c = cents[best_j] * (len(clusters[best_j]) - 1) + v
            cents[best_j] = c / (np.linalg.norm(c) + 1e-12)
        else:
            clusters.append([r])
            cents.append(v)
            sims.append([])
    return clusters, cents, sims


def split_early_future(items, mode):
    if mode["kind"] == "count":
        n = int(mode["value"])
        if len(items) <= n:
            return [], []
        return items[:n], items[n:]
    cutoff = items[0]["ts"] + float(mode["value"]) * 3600.0
    early = [r for r in items if r["ts"] <= cutoff]
    future = [r for r in items if r["ts"] > cutoff]
    return early, future


def assign_future_to_seeds(future, emb, cents, thr):
    counts = np.zeros(len(cents), dtype=float)
    reach = np.zeros(len(cents), dtype=float)
    for r in future:
        v = norm_vec(emb, r["idx"])
        sims = [float(v @ c) for c in cents]
        if not sims:
            continue
        j = int(np.argmax(sims))
        if sims[j] >= thr:
            counts[j] += 1.0
            reach[j] += max(0.0, float(r["followers"]))
    return counts, reach


def build_panel_for(rows_by, emb_by, meta, ol, thr, mode):
    panel = []
    event_summaries = []
    for si, sym in enumerate(SYMS, 1):
        ev = first_by_event(rows_by[sym], start=TRAIN_END, end=VAL_END)
        past_cents = []
        used_events = 0
        for (_, day), d in sorted(ev.items(), key=lambda kv: kv[0][1]):
            items = sorted(d.values(), key=lambda r: r["ts"])
            items = [r for r in items if r["kol"] in ol]
            if len(items) < MIN_EVENT_KOLS:
                continue
            early, future = split_early_future(items, mode)
            if len(early) < MIN_EARLY_KOLS or len(future) < MIN_FUTURE_KOLS:
                continue
            clusters, cents, intra_sims = greedy_seed_frames(early, emb_by[sym], thr)
            if len(clusters) < 2:
                continue
            future_counts, future_reach = assign_future_to_seeds(future, emb_by[sym], cents, thr)
            if future_counts.sum() <= 0:
                continue
            used_events += 1
            event_id = f"{sym}:{day}:{mode['name']}:{thr:.2f}"
            early_start = early[0]["ts"]
            cutoff_elapsed = (early[-1]["ts"] - early_start) / 3600.0
            early_total_reach = sum(max(0.0, float(r["followers"])) for r in early)
            early_stances = np.array([r["stance"] for r in early], dtype=float)
            event_stance_mean = float(np.mean(early_stances)) if len(early_stances) else 0.0
            event_bull_share = float(np.mean(early_stances > 0)) if len(early_stances) else 0.0
            event_bear_share = float(np.mean(early_stances < 0)) if len(early_stances) else 0.0
            event_disagreement = 1.0 if event_bull_share > 0 and event_bear_share > 0 else 0.0
            for j, cl in enumerate(clusters):
                origin = min(cl, key=lambda r: r["ts"])
                stances = [r["stance"] for r in cl]
                stance_arr = np.array(stances, dtype=float)
                early_reach = sum(max(0.0, float(r["followers"])) for r in cl)
                stance_weights = np.array([max(0.0, float(r["followers"])) for r in cl], dtype=float)
                if stance_weights.sum() > 0:
                    follower_weighted_stance = float((stance_arr * stance_weights).sum() / stance_weights.sum())
                else:
                    follower_weighted_stance = float(stance_arr.mean()) if len(stance_arr) else 0.0
                if past_cents:
                    P = np.vstack(past_cents[-1000:])
                    novelty = 1.0 - float(np.max(cents[j] @ P.T))
                else:
                    novelty = np.nan
                cohesion = float(np.mean(intra_sims[j])) if intra_sims[j] else 1.0
                origin_m = meta.get(origin["kol"], {})
                origin_ol = float(ol[origin["kol"]])
                origin_logfoll = float(origin_m.get("log_followers", np.log1p(origin["followers"])))
                row = {
                    "event_id": event_id,
                    "sym": sym,
                    "day": day,
                    "split": "train" if day < MODEL_SPLIT else "val",
                    "thr": thr,
                    "mode": mode["name"],
                    "early_kols": len(early),
                    "future_kols": len(future),
                    "n_seed_frames": len(clusters),
                    "log_event_early_kols": math.log(len(early)),
                    "cutoff_elapsed_hours": cutoff_elapsed,
                    "origin_kol": origin["kol"],
                    "origin_text": origin.get("text", ""),
                    "frame_texts": [x.get("text", "") for x in sorted(cl, key=lambda r: r["ts"])[:3]],
                    "frame_all_texts": [x.get("text", "") for x in sorted(cl, key=lambda r: r["ts"])],
                    "frame_kols": [x["kol"] for x in sorted(cl, key=lambda r: r["ts"])],
                    "frame_embedding": cents[j].astype(np.float32, copy=True),
                    "origin_ol": origin_ol,
                    "origin_logfoll": origin_logfoll,
                    "origin_verified": float(origin_m.get("verified", origin["verified"])),
                    "origin_rank_frac": (early.index(origin) + 1) / max(1, len(early)),
                    "early_size": len(cl),
                    "log_early_adopt": math.log1p(max(0, len(cl) - 1)),
                    "log_early_reach": math.log1p(early_reach),
                    "early_frame_share": len(cl) / max(1, len(early)),
                    "early_reach_share": early_reach / max(1.0, early_total_reach),
                    "novelty": novelty,
                    "cohesion": cohesion,
                    "origin_stance": float(origin["stance"]),
                    "frame_stance_mean": float(stance_arr.mean()) if len(stance_arr) else 0.0,
                    "frame_stance_abs": abs(float(stance_arr.mean())) if len(stance_arr) else 0.0,
                    "frame_bull_share": float(np.mean(stance_arr > 0)) if len(stance_arr) else 0.0,
                    "frame_bear_share": float(np.mean(stance_arr < 0)) if len(stance_arr) else 0.0,
                    "follower_weighted_stance": follower_weighted_stance,
                    "event_stance_mean": event_stance_mean,
                    "event_stance_abs": abs(event_stance_mean),
                    "event_bull_share": event_bull_share,
                    "event_bear_share": event_bear_share,
                    "event_sentiment_disagreement": event_disagreement,
                    "stance_abs": 1.0 if any(x != 0 for x in stances) else 0.0,
                    "stance_pos": 1.0 if sum(x > 0 for x in stances) > sum(x < 0 for x in stances) else 0.0,
                    "stance_neg": 1.0 if sum(x < 0 for x in stances) > sum(x > 0 for x in stances) else 0.0,
                    "future_adopt": float(future_counts[j]),
                    "future_reach": float(future_reach[j]),
                    "log_future_adopt": math.log1p(float(future_counts[j])),
                    "log_future_reach": math.log1p(float(future_reach[j])),
                }
                row["ol_x_visibility"] = row["origin_ol"] * row["origin_logfoll"]
                row["ol_x_novelty"] = row["origin_ol"] * (0.0 if not np.isfinite(row["novelty"]) else row["novelty"])
                row["sentiment_alignment"] = row["origin_stance"] * row["frame_stance_mean"]
                panel.append(row)
            event_summaries.append({
                "event_id": event_id,
                "sym": sym,
                "day": day,
                "n_seed_frames": len(clusters),
                "early_kols": len(early),
                "future_kols": len(future),
                "future_assigned": float(future_counts.sum()),
                "future_reach": float(future_reach.sum()),
            })
            past_cents.extend(cents)
        log(f"  thr={thr:.2f} {mode['name']:<7} {si:02d}/{len(SYMS)} {sym:<5} events={used_events:>4} rows={len(panel):>6}")
    return panel, event_summaries


def train_standardizer(X, train_mask):
    med = np.nanmedian(X[train_mask], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X2 = np.where(np.isfinite(X), X, med)
    mu = X2[train_mask].mean(axis=0)
    sd = X2[train_mask].std(axis=0) + 1e-12
    return med, mu, sd


def apply_standardizer(X, med, mu, sd):
    X2 = np.where(np.isfinite(X), X, med)
    return (X2 - mu) / sd


def fit_ridge_scores(rows, features, target, alpha=3.0):
    X = np.array([[r.get(f, np.nan) for f in features] for r in rows], dtype=float)
    y = np.array([r[target] for r in rows], dtype=float)
    train = np.array([r["split"] == "train" for r in rows], dtype=bool)
    good_train = train & np.isfinite(y)
    if good_train.sum() < len(features) + 30 or (~train).sum() < 30:
        return None
    med, mu, sd = train_standardizer(X, good_train)
    Xs = apply_standardizer(X, med, mu, sd)
    Xtr = np.column_stack([np.ones(good_train.sum()), Xs[good_train]])
    ytr = y[good_train]
    lam = np.eye(Xtr.shape[1]) * alpha
    lam[0, 0] = 0.0
    beta = np.linalg.solve(Xtr.T @ Xtr + lam, Xtr.T @ ytr)
    scores = np.column_stack([np.ones(len(rows)), Xs]) @ beta
    return scores


def rank_average(x):
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def auc_score(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    good = np.isfinite(score)
    y = y[good]
    score = score[good]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rank_average(score)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    good = np.isfinite(score)
    y = y[good]
    score = score[good]
    n_pos = int(y.sum())
    if n_pos == 0:
        return np.nan
    order = np.argsort(-score)
    yy = y[order]
    hits = np.cumsum(yy)
    precision = hits / (np.arange(len(yy)) + 1)
    return float((precision * yy).sum() / n_pos)


def dcg(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0
    denom = np.log2(np.arange(2, len(values) + 2))
    return float((values / denom).sum())


def event_ranking_metrics(rows, scores, target, split="val", k=3):
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r["split"] == split and np.isfinite(scores[i]):
            groups[r["event_id"]].append((i, float(r[target]), float(scores[i])))
    ndcgs = []
    hit1 = []
    recallk = []
    events = 0
    for vals in groups.values():
        if len(vals) < 2:
            continue
        y = np.array([v[1] for v in vals], dtype=float)
        s = np.array([v[2] for v in vals], dtype=float)
        if np.nanmax(y) <= 0:
            continue
        events += 1
        pred_order = np.argsort(-s)
        ideal_order = np.argsort(-y)
        kk = min(k, len(vals))
        nd = dcg(y[pred_order[:kk]]) / max(dcg(y[ideal_order[:kk]]), 1e-12)
        ndcgs.append(nd)
        best = set(np.where(y == np.nanmax(y))[0].tolist())
        hit1.append(1.0 if pred_order[0] in best else 0.0)
        recallk.append(1.0 if any(idx in best for idx in pred_order[:kk]) else 0.0)
    return {
        "n_events": int(events),
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else np.nan,
        "top1_hits_actual_best": float(np.mean(hit1)) if hit1 else np.nan,
        f"recall_actual_best@{k}": float(np.mean(recallk)) if recallk else np.nan,
    }


def softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - np.nanmax(x)
    e = np.exp(np.clip(x, -50, 50))
    return e / max(float(e.sum()), 1e-12)


def pearson_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2:
        return np.nan
    aa = a - np.mean(a)
    bb = b - np.mean(b)
    den = np.sqrt((aa * aa).sum() * (bb * bb).sum())
    if den <= 1e-12:
        return np.nan
    return float((aa * bb).sum() / den)


def spearman_corr(a, b):
    if len(a) < 2:
        return np.nan
    return pearson_corr(rank_average(np.asarray(a, dtype=float)), rank_average(np.asarray(b, dtype=float)))


def js_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(float(p.sum()), 1e-12)
    q = q / max(float(q.sum()), 1e-12)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log(a[mask] / np.maximum(b[mask], 1e-12))).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def event_distribution_metrics(rows, scores, target, split="val"):
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r["split"] == split and np.isfinite(scores[i]):
            groups[r["event_id"]].append((float(r[target]), float(scores[i])))
    spears = []
    pears = []
    jsds = []
    mass3 = []
    events = 0
    for vals in groups.values():
        if len(vals) < 2:
            continue
        y = np.array([v[0] for v in vals], dtype=float)
        if np.nanmax(y) <= 0:
            continue
        s = np.array([v[1] for v in vals], dtype=float)
        p = y / max(float(y.sum()), 1e-12)
        q = softmax(s)
        events += 1
        spears.append(spearman_corr(y, s))
        pears.append(pearson_corr(p, q))
        jsds.append(js_divergence(p, q))
        kk = min(3, len(y))
        mass3.append(float(p[np.argsort(-s)[:kk]].sum()))
    return {
        "n_events": int(events),
        "spearman_mean": float(np.nanmean(spears)) if spears else np.nan,
        "pearson_dist_mean": float(np.nanmean(pears)) if pears else np.nan,
        "js_divergence_mean": float(np.nanmean(jsds)) if jsds else np.nan,
        "actual_mass_in_pred_top3": float(np.nanmean(mass3)) if mass3 else np.nan,
    }


def evaluate_scores(rows, scores, target):
    y = np.array([r[target] for r in rows], dtype=float)
    train = np.array([r["split"] == "train" for r in rows], dtype=bool)
    val = ~train
    if train.sum() < 30 or val.sum() < 30:
        return None
    q90 = float(np.nanquantile(y[train], 0.90))
    ybin = (y >= q90).astype(int)
    val_y = ybin[val]
    val_s = np.asarray(scores, dtype=float)[val]
    return {
        "target": target,
        "train_q90": q90,
        "n_train": int(train.sum()),
        "n_val": int(val.sum()),
        "val_positive_rate": float(val_y.mean()) if len(val_y) else np.nan,
        "val_auc_top10": auc_score(val_y, val_s),
        "val_ap_top10": average_precision(val_y, val_s),
        "event_rank": event_ranking_metrics(rows, scores, target, split="val", k=3),
        "distribution_reconstruction": event_distribution_metrics(rows, scores, target, split="val"),
    }


def raw_score(rows, expr):
    vals = []
    for r in rows:
        if expr == "early_reach":
            vals.append(r["log_early_reach"])
        elif expr == "early_adopt":
            vals.append(r["log_early_adopt"])
        elif expr == "origin_followers":
            vals.append(r["origin_logfoll"])
        elif expr == "origin_ol":
            vals.append(r["origin_ol"])
        elif expr == "frame_sentiment_abs":
            vals.append(r["frame_stance_abs"])
        elif expr == "frame_bullish":
            vals.append(r["frame_stance_mean"])
        elif expr == "follower_weighted_sentiment":
            vals.append(r["follower_weighted_stance"])
        else:
            vals.append(r["log_early_reach"])
    return np.asarray(vals, dtype=float)


def run_one(rows):
    out = {
        "n_rows": len(rows),
        "n_train_rows": sum(r["split"] == "train" for r in rows),
        "n_val_rows": sum(r["split"] == "val" for r in rows),
        "n_events": len(set(r["event_id"] for r in rows)),
        "models": {},
    }
    for target in ["log_future_adopt", "log_future_reach"]:
        out["models"][target] = {}
        for name in [
            "early_reach", "early_adopt", "origin_followers", "origin_ol",
            "frame_sentiment_abs", "frame_bullish", "follower_weighted_sentiment",
        ]:
            scores = raw_score(rows, name)
            out["models"][target][f"raw_{name}"] = evaluate_scores(rows, scores, target)
        for name, feats in FEATURE_SETS.items():
            scores = fit_ridge_scores(rows, feats, target)
            if scores is None:
                out["models"][target][f"ridge_{name}"] = None
            else:
                out["models"][target][f"ridge_{name}"] = evaluate_scores(rows, scores, target)
    return out


def main():
    t0 = time.time()
    log("[1/5] Loading tweets and MiniLM embeddings")
    rows_by = {}
    emb_by = {}
    all_rows = []
    for i, sym in enumerate(SYMS, 1):
        rows, emb = load_symbol(sym)
        rows_by[sym] = rows
        emb_by[sym] = emb
        all_rows.extend(rows)
        log(f"  {i:02d}/{len(SYMS)} {sym:<5} rows={len(rows):>6} emb={emb.shape}")
    meta = compute_metadata(all_rows)
    ol = compute_oltrait(all_rows, meta)

    result = {
        "task": "sentiment_baselines_and_narrative_reconstruction",
        "train_end_for_oltrait": TRAIN_END,
        "model_train_period": [TRAIN_END, MODEL_SPLIT],
        "validation_period": [MODEL_SPLIT, VAL_END],
        "thresholds": THRESHOLDS,
        "early_modes": EARLY_MODES,
        "min_event_kols": MIN_EVENT_KOLS,
        "min_early_kols": MIN_EARLY_KOLS,
        "min_future_kols": MIN_FUTURE_KOLS,
        "feature_sets": FEATURE_SETS,
        "n_tweets": len(all_rows),
        "n_ol_kols": len(ol),
        "by_setting": {},
    }

    log("[3/5] Building early-frame panels and evaluating models")
    for thr in THRESHOLDS:
        for mode in EARLY_MODES:
            key = f"thr{thr:.2f}_{mode['name']}"
            log(f"\n--- {key} ---")
            rows, events = build_panel_for(rows_by, emb_by, meta, ol, thr, mode)
            if not rows:
                result["by_setting"][key] = {"n_rows": 0}
                continue
            out = run_one(rows)
            out["n_event_summaries"] = len(events)
            out["event_future_assigned_mean"] = float(np.mean([e["future_assigned"] for e in events])) if events else np.nan
            result["by_setting"][key] = out
            # Compact progress summary.
            for target in ["log_future_adopt", "log_future_reach"]:
                base = out["models"][target].get("ridge_early_pop") or {}
                sent = out["models"][target].get("ridge_sentiment_all") or {}
                pop_sent = out["models"][target].get("ridge_early_pop_sentiment") or {}
                full = out["models"][target].get("ridge_full") or {}
                b_rank = (base.get("event_rank") or {}).get("ndcg@3", np.nan)
                s_rank = (sent.get("event_rank") or {}).get("ndcg@3", np.nan)
                ps_rank = (pop_sent.get("event_rank") or {}).get("ndcg@3", np.nan)
                f_rank = (full.get("event_rank") or {}).get("ndcg@3", np.nan)
                b_ap = base.get("val_ap_top10", np.nan)
                s_ap = sent.get("val_ap_top10", np.nan)
                ps_ap = pop_sent.get("val_ap_top10", np.nan)
                f_ap = full.get("val_ap_top10", np.nan)
                f_js = (full.get("distribution_reconstruction") or {}).get("js_divergence_mean", np.nan)
                b_js = (base.get("distribution_reconstruction") or {}).get("js_divergence_mean", np.nan)
                log(
                    f"  {target:<16} sent AP={s_ap:.3f} NDCG3={s_rank:.3f} | "
                    f"pop AP={b_ap:.3f} NDCG3={b_rank:.3f} | "
                    f"pop+sent AP={ps_ap:.3f} NDCG3={ps_rank:.3f} | "
                    f"full AP={f_ap:.3f} NDCG3={f_rank:.3f} JS={b_js:.3f}->{f_js:.3f}"
                )

    log("\n[4/5] Summarizing best settings")
    summary = []
    for key, setting in result["by_setting"].items():
        for target, models in setting.get("models", {}).items():
            for model, metrics in models.items():
                if not metrics:
                    continue
                rank = metrics.get("event_rank") or {}
                summary.append({
                    "setting": key,
                    "target": target,
                    "model": model,
                    "ap": metrics.get("val_ap_top10"),
                    "auc": metrics.get("val_auc_top10"),
                    "ndcg3": rank.get("ndcg@3"),
                    "hit1": rank.get("top1_hits_actual_best"),
                    "events": rank.get("n_events"),
                })
    summary_sorted = sorted(
        summary,
        key=lambda x: (
            -1 if x["target"] == "log_future_reach" else 0,
            -(x["ndcg3"] if np.isfinite(x["ndcg3"]) else -1),
            -(x["ap"] if np.isfinite(x["ap"]) else -1),
        ),
    )
    result["top_summary"] = summary_sorted[:40]
    result["elapsed_sec"] = time.time() - t0
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"[5/5] wrote {OUT}")
    log(f"elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
