"""Phase93: redesigned ablation (core) for current model = LambdaMART on {context + O_k}.
A: ranker (ridge pointwise / XGBoost pairwise / LambdaMART listwise) on {context+O_k}.
B: features under LambdaMART (context-only / O_k-only / full).
C1: full vs shuffled-O_k (identity). 5 rolling windows pooled; symbol-balanced bootstrap CI.
thr=0.50 first10 reach.
"""
from __future__ import annotations
import collections, json, math, pathlib, time
import numpy as np
import phase5_sentiment_reconstruction as p5
import phase7_origin_alert as p7
import phase33_origin_alert_ablation as p33
import phase65_pit_lightweight_2025_2026 as p65
import lightgbm as lgb
import xgboost as xgb

THR = 0.50; MEK = 8; TARGET = "log_future_reach"; N_GRID = [50, 100, 200, 400]
ALPHAS = p65.RIDGE_ALPHAS
p65.BOOTSTRAP_B = 4000; p65.RNG = np.random.default_rng(93)
WINDOWS = [
    ("25.6-26.6", "2024-06-01", "2025-06-01", "2026-06-01", "2024-12-01"),
    ("24.6-25.6", "2023-06-01", "2024-06-01", "2025-06-01", "2023-12-01"),
    ("23.6-24.6", "2022-06-01", "2023-06-01", "2024-06-01", "2022-12-01"),
    ("22.6-23.6", "2021-06-01", "2022-06-01", "2023-06-01", "2021-12-01"),
    ("21.6-22.6", "2020-06-01", "2021-06-01", "2022-06-01", "2020-12-01"),
]
def log(m): print(m, flush=True)
p65.split_for_block = lambda b: b; p65.THR = THR
def make_bfd(tr, te, ee):
    def f(day):
        if tr <= day < te: return "train"
        if te <= day < ee: return "test"
        return None
    return f
def states_for(tr_s, te_s, rows_by, emb_by, all_rows):
    states = {}
    for block, cutoff in [("train", tr_s), ("test", te_s)]:
        hist_rows = p65.rows_before(all_rows, cutoff)
        meta = p5.compute_metadata(hist_rows)
        ol = p65.with_train_end(cutoff, p5.compute_oltrait, hist_rows, meta)
        hist = p65.with_train_end(cutoff, p7.compute_origin_history, rows_by, emb_by, THR)
        raw_ol = p65.with_train_end(cutoff, p33.compute_raw_oltrait, hist_rows, meta)
        states[block] = {"cutoff": cutoff, "meta": meta, "ol": ol, "hist": hist,
                         "raw_ol": raw_ol, "shuffled_ol": p65.make_shuffled_ol(ol, block)}
    return states
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
def lgbm_fit(X, y, order, sizes, n):
    m = lgb.LGBMRanker(objective="rank_xendcg", n_estimators=n, learning_rate=0.05, num_leaves=31,
                       min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=8, verbosity=-1)
    m.fit(X[order], y[order], group=sizes); return m
def xgb_fit(X, y, order, sizes, n):
    qid = np.concatenate([[gi] * s for gi, s in enumerate(sizes)])
    m = xgb.XGBRanker(objective="rank:pairwise", n_estimators=n, learning_rate=0.05, max_depth=6,
                      tree_method="hist", subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=8, verbosity=0)
    m.fit(X[order], y[order], qid=qid); return m
def tree_predict(panel, feats, ygrade, tr, infit, indev, fitter):
    X = p65.matrix(panel, feats)
    o_inf, s_inf = grouped([i for i in range(len(panel)) if infit[i]], panel)
    best = (-1.0, None)
    for n in N_GRID:
        nd = symbal_ndcg_mask(panel, fitter(X, ygrade, o_inf, s_inf, n).predict(X), indev)
        if nd > best[0]: best = (nd, n)
    n = best[1] or 100
    o_tr, s_tr = grouped([i for i in range(len(panel)) if tr[i]], panel)
    return fitter(X, ygrade, o_tr, s_tr, n).predict(X)
def ridge_predict(panel, feats, tr, infit, indev):
    best = (-1.0, None)
    for a in ALPHAS:
        pred = p65.ridge_predict_from_mask(panel, feats, TARGET, infit, a)
        if pred is None: continue
        nd = symbal_ndcg_mask(panel, pred, indev)
        if nd > best[0]: best = (nd, a)
    a = best[1] or 10.0
    return p65.ridge_predict_from_mask(panel, feats, TARGET, tr, a)

def main():
    t0 = time.time()
    rows_by, emb_by, all_rows = {}, {}, []
    for sym in p5.SYMS:
        r, e = p5.load_symbol(sym); rows_by[sym] = r; emb_by[sym] = e; all_rows.extend(r)
    p7.MIN_EVENT_KOLS = MEK
    base = p7.FEATURE_SETS["no_ol_strong"]
    full = base + ["origin_ol"]; okonly = ["origin_ol"]; shuf = base + ["origin_ol_shuffled"]
    configs = {"A1_ridge_full": ("ridge", full), "A2_xgb_pairwise_full": ("xgb", full),
               "A3_lgbm_full": ("lgbm", full), "B1_lgbm_context": ("lgbm", base),
               "B2_lgbm_Okonly": ("lgbm", okonly), "C1_lgbm_shuffOk": ("lgbm", shuf)}
    ev = {k: [] for k in configs}
    for lab, tr_s, te_s, te_e, inner in WINDOWS:
        p65.TRAIN_START = tr_s; p65.TEST_END = te_e; p65.block_for_day = make_bfd(tr_s, te_s, te_e)
        states = states_for(tr_s, te_s, rows_by, emb_by, all_rows)
        p65.ORIGIN_WINDOW = {"name": "first10", "max_rank": 10}
        panel, _ = p65.build_pit_origin_panel(rows_by, emb_by, states)
        day = np.array([r["day"] for r in panel]); tr = np.array([r["split"] == "train" for r in panel])
        infit = tr & (day < inner); indev = tr & (day >= inner)
        yc = np.array([float(r.get(TARGET, float("nan"))) for r in panel], dtype=float)
        finm = np.isfinite(yc); yg = np.zeros(len(panel), dtype=int)
        qs = np.unique(np.quantile(yc[finm], np.linspace(0.0, 1.0, 33)))
        yg[finm] = np.clip(np.digitize(yc[finm], qs[1:-1]), 0, max(1, len(qs) - 2)).astype(int)
        for k, (kind, feats) in configs.items():
            if kind == "ridge": pred = ridge_predict(panel, feats, tr, infit, indev)
            elif kind == "xgb": pred = tree_predict(panel, feats, yg, tr, infit, indev, xgb_fit)
            else: pred = tree_predict(panel, feats, yg, tr, infit, indev, lgbm_fit)
            ev[k] += p65.event_rows_for_split(panel, pred, TARGET, "test")
        log("  window %s done" % lab)
    log("pooled events per config: %d" % len(ev["A3_lgbm_full"]))
    means = {k: {m: p7.symbal_mean(ev[k], m) for m in p7.METRICS} for k in configs}
    log("=== pooled symbol-balanced means ===")
    for k in configs:
        mm = means[k]; log("  %-22s NDCG=%.4f Hit=%.4f Mass=%.4f JS=%.4f" % (k, mm["ndcg3"], mm["hit1"], mm["mass3"], mm["js"]))
    pairs = [("A3_lgbm_full", "A1_ridge_full"), ("A3_lgbm_full", "A2_xgb_pairwise_full"),
             ("A3_lgbm_full", "B1_lgbm_context"), ("A3_lgbm_full", "C1_lgbm_shuffOk"),
             ("A3_lgbm_full", "B2_lgbm_Okonly")]
    comps = p65.comparisons(ev, pairs)
    log("=== pooled bootstrap CI (B=4000) ===")
    for mdl, b in pairs:
        for metric in ["ndcg3", "hit1"]:
            s = comps["%s_vs_%s" % (mdl, b)][metric]["symbol_balanced_bootstrap"]
            sig = "SIG>0" if (s["ci05"] is not None and s["ci05"] > 0) else "ns"
            log("  %-38s %-6s %+0.4f [%+.4f, %+.4f] %s" % ("%s - %s" % (mdl, b), metric, s["observed"], s["ci05"], s["ci95"], sig))
    pathlib.Path(__file__).with_name("phase93_ablation_redesign_core_result.json").write_text(json.dumps({"means": means, "comparisons": comps, "elapsed_sec": time.time() - t0}, indent=2))
    log("done %.1fs" % (time.time() - t0))

if __name__ == "__main__":
    main()
