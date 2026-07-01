"""Phase96: PERFORMANCE test of relational originator features (single main window 25.6-26.6).
Variants under LambdaMART (inner-CV): {context} / {context+Ok} / {context+Ok+relational} / {context+relational}.
Gate: does relational beat current best {context+Ok}=~0.811 ? thr=0.50 first10 reach.
"""
from __future__ import annotations
import collections, json, math, pathlib, time
import numpy as np
import phase5_sentiment_reconstruction as p5
import phase7_origin_alert as p7
import phase33_origin_alert_ablation as p33
import phase65_pit_lightweight_2025_2026 as p65
import lightgbm as lgb

THR = 0.50; MEK = 8; TARGET = "log_future_reach"
TR_S, TE_S, TE_E = "2024-06-01", "2025-06-01", "2026-06-01"; INNER = "2025-01-01"; N_GRID = [50, 100, 200, 400]
def log(m): print(m, flush=True)
p65.split_for_block = lambda b: b; p65.THR = THR
def block_for_day(day):
    if TR_S <= day < TE_S: return "train"
    if TE_S <= day < TE_E: return "test"
    return None
p65.block_for_day = block_for_day
def states_for(rows_by, emb_by, all_rows):
    states = {}
    for block, cutoff in [("train", TR_S), ("test", TE_S)]:
        hist_rows = p65.rows_before(all_rows, cutoff); meta = p5.compute_metadata(hist_rows)
        ol = p65.with_train_end(cutoff, p5.compute_oltrait, hist_rows, meta)
        hist = p65.with_train_end(cutoff, p7.compute_origin_history, rows_by, emb_by, THR)
        raw_ol = p65.with_train_end(cutoff, p33.compute_raw_oltrait, hist_rows, meta)
        states[block] = {"cutoff": cutoff, "meta": meta, "ol": ol, "hist": hist, "raw_ol": raw_ol, "shuffled_ol": p65.make_shuffled_ol(ol, block)}
    return states
def add_relational(panel):
    by = collections.defaultdict(list)
    for i, r in enumerate(panel): by[r["event_id"]].append(i)
    for ev, idxs in by.items():
        ols = np.array([float(panel[i]["origin_ol"]) for i in idxs])
        mx = float(ols.max()); mn = float(ols.mean()); n = len(idxs)
        ranks = ols.argsort().argsort()
        for j, i in enumerate(idxs):
            o = float(panel[i]["origin_ol"])
            panel[i]["ol_rank_in_event"] = float(ranks[j]) / (n - 1) if n > 1 else 0.5
            panel[i]["ol_gap_to_max"] = o - mx
            panel[i]["ol_minus_mean"] = o - mn
            panel[i]["ol_is_top"] = 1.0 if o >= mx else 0.0
def grouped(idx, panel):
    by = collections.OrderedDict()
    for i in idx: by.setdefault(panel[i]["event_id"], []).append(i)
    order, sizes = [], []
    for ev, lst in by.items(): order.extend(lst); sizes.append(len(lst))
    return order, sizes
def symbal_ndcg_mask(panel, pred, mask):
    g = collections.defaultdict(list)
    for i, r in enumerate(panel):
        if mask[i] and np.isfinite(pred[i]):
            g[r["event_id"]].append((r["sym"], float(r[TARGET]), float(pred[i])))
    return p7.symbal_mean(p65.event_metrics_from_groups(g), "ndcg3")
def fit_lgbm(X, y, order, sizes, n):
    m = lgb.LGBMRanker(objective="rank_xendcg", n_estimators=n, learning_rate=0.05, num_leaves=31,
                       min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=8, verbosity=-1)
    m.fit(X[order], y[order], group=sizes); return m
def run(panel, feats, yg, tr, infit, indev):
    X = p65.matrix(panel, feats)
    o_inf, s_inf = grouped([i for i in range(len(panel)) if infit[i]], panel)
    best = (-1.0, None)
    for n in N_GRID:
        nd = symbal_ndcg_mask(panel, fit_lgbm(X, yg, o_inf, s_inf, n).predict(X), indev)
        if nd > best[0]: best = (nd, n)
    n = best[1] or 100
    o_tr, s_tr = grouped([i for i in range(len(panel)) if tr[i]], panel)
    pred = fit_lgbm(X, yg, o_tr, s_tr, n).predict(X)
    evs = p65.event_rows_for_split(panel, pred, TARGET, "test")
    return {k: p7.symbal_mean(evs, k) for k in p7.METRICS}, n
def main():
    t0 = time.time()
    rows_by, emb_by, all_rows = {}, {}, []
    for sym in p5.SYMS:
        r, e = p5.load_symbol(sym); rows_by[sym] = r; emb_by[sym] = e; all_rows.extend(r)
    p7.MIN_EVENT_KOLS = MEK; p65.TRAIN_START = TR_S; p65.TEST_END = TE_E
    states = states_for(rows_by, emb_by, all_rows)
    p65.ORIGIN_WINDOW = {"name": "first10", "max_rank": 10}
    panel, _ = p65.build_pit_origin_panel(rows_by, emb_by, states)
    add_relational(panel)
    day = np.array([r["day"] for r in panel]); tr = np.array([r["split"] == "train" for r in panel])
    infit = tr & (day < INNER); indev = tr & (day >= INNER)
    yc = np.array([float(r.get(TARGET, float("nan"))) for r in panel], dtype=float)
    fin = np.isfinite(yc); yg = np.zeros(len(panel), dtype=int)
    qs = np.unique(np.quantile(yc[fin], np.linspace(0.0, 1.0, 33)))
    yg[fin] = np.clip(np.digitize(yc[fin], qs[1:-1]), 0, max(1, len(qs) - 2)).astype(int)
    REL = ["ol_rank_in_event", "ol_gap_to_max", "ol_minus_mean", "ol_is_top"]
    base = p7.FEATURE_SETS["no_ol_strong"]
    variants = {"context": base, "context+Ok (current best)": base + ["origin_ol"],
                "context+Ok+relational": base + ["origin_ol"] + REL, "context+relational (no abs Ok)": base + REL}
    log("=== main-window 25.6-26.6 performance (test events=%d) ===" % (~tr).sum())
    res = {}
    for name, feats in variants.items():
        m, n = run(panel, feats, yg, tr, infit, indev); res[name] = m
        log("  %-32s n=%d NDCG=%.4f Hit=%.4f Mass=%.4f JS=%.4f" % (name, n, m["ndcg3"], m["hit1"], m["mass3"], m["js"]))
    pathlib.Path(__file__).with_name("phase96_relational_perf_2566_result.json").write_text(json.dumps(res, indent=2))
    log("done %.1fs" % (time.time() - t0))
if __name__ == "__main__":
    main()
