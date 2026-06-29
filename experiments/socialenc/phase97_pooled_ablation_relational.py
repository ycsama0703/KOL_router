"""Phase97: pooled (5-window) ablation WITH relational features.
A{context} / B{context+Ok} / B2{context+Ok+relational} / C{context+shuffOk+shuff-relational}.
LambdaMART; pooled symbol-balanced bootstrap CI on B2-A, B2-B, B2-C. thr=0.50 first10 reach.
"""
from __future__ import annotations
import collections, json, math, pathlib, time
import numpy as np
import phase5_sentiment_reconstruction as p5
import phase7_origin_alert as p7
import phase33_origin_alert_ablation as p33
import phase65_pit_lightweight_2025_2026 as p65
import lightgbm as lgb
THR = 0.50; MEK = 8; TARGET = "log_future_reach"; N_GRID = [50, 100, 200, 400]
p65.BOOTSTRAP_B = 4000; p65.RNG = np.random.default_rng(97)
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
        hr = p65.rows_before(all_rows, cutoff); meta = p5.compute_metadata(hr)
        ol = p65.with_train_end(cutoff, p5.compute_oltrait, hr, meta)
        hist = p65.with_train_end(cutoff, p7.compute_origin_history, rows_by, emb_by, THR)
        raw = p65.with_train_end(cutoff, p33.compute_raw_oltrait, hr, meta)
        states[block] = {"cutoff": cutoff, "meta": meta, "ol": ol, "hist": hist, "raw_ol": raw, "shuffled_ol": p65.make_shuffled_ol(ol, block)}
    return states
def add_relational(panel, src, prefix):
    by = collections.defaultdict(list)
    for i, r in enumerate(panel): by[r["event_id"]].append(i)
    for ev, idxs in by.items():
        vals = np.array([float(panel[i][src]) for i in idxs]); mx = float(vals.max()); mn = float(vals.mean()); n = len(idxs)
        ranks = vals.argsort().argsort()
        for j, i in enumerate(idxs):
            v = float(panel[i][src])
            panel[i][prefix + "_rank"] = float(ranks[j]) / (n - 1) if n > 1 else 0.5
            panel[i][prefix + "_gapmax"] = v - mx
            panel[i][prefix + "_minusmean"] = v - mn
            panel[i][prefix + "_istop"] = 1.0 if v >= mx else 0.0
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
def fit_predict(panel, feats, yg, tr, infit, indev):
    X = p65.matrix(panel, feats)
    o_inf, s_inf = grouped([i for i in range(len(panel)) if infit[i]], panel)
    best = (-1.0, None)
    for n in N_GRID:
        nd = symbal_ndcg_mask(panel, fit_lgbm(X, yg, o_inf, s_inf, n).predict(X), indev)
        if nd > best[0]: best = (nd, n)
    n = best[1] or 100
    o_tr, s_tr = grouped([i for i in range(len(panel)) if tr[i]], panel)
    return fit_lgbm(X, yg, o_tr, s_tr, n).predict(X)
def main():
    t0 = time.time()
    rows_by, emb_by, all_rows = {}, {}, []
    for sym in p5.SYMS:
        r, e = p5.load_symbol(sym); rows_by[sym] = r; emb_by[sym] = e; all_rows.extend(r)
    p7.MIN_EVENT_KOLS = MEK
    base = p7.FEATURE_SETS["no_ol_strong"]
    OLR = ["ol_rank", "ol_gapmax", "ol_minusmean", "ol_istop"]
    SHR = ["sh_rank", "sh_gapmax", "sh_minusmean", "sh_istop"]
    sets = {"A_context": base, "B_ctx+Ok": base + ["origin_ol"],
            "B2_ctx+Ok+rel": base + ["origin_ol"] + OLR, "C_ctx+shuf+shufrel": base + ["origin_ol_shuffled"] + SHR}
    ev = {k: [] for k in sets}
    for lab, tr_s, te_s, te_e, inner in WINDOWS:
        p65.TRAIN_START = tr_s; p65.TEST_END = te_e; p65.block_for_day = make_bfd(tr_s, te_s, te_e)
        states = states_for(tr_s, te_s, rows_by, emb_by, all_rows)
        p65.ORIGIN_WINDOW = {"name": "first10", "max_rank": 10}
        panel, _ = p65.build_pit_origin_panel(rows_by, emb_by, states)
        add_relational(panel, "origin_ol", "ol"); add_relational(panel, "origin_ol_shuffled", "sh")
        day = np.array([r["day"] for r in panel]); tr = np.array([r["split"] == "train" for r in panel])
        infit = tr & (day < inner); indev = tr & (day >= inner)
        yc = np.array([float(r.get(TARGET, float("nan"))) for r in panel], dtype=float)
        fin = np.isfinite(yc); yg = np.zeros(len(panel), dtype=int)
        qs = np.unique(np.quantile(yc[fin], np.linspace(0.0, 1.0, 33)))
        yg[fin] = np.clip(np.digitize(yc[fin], qs[1:-1]), 0, max(1, len(qs) - 2)).astype(int)
        for k, feats in sets.items():
            ev[k] += p65.event_rows_for_split(panel, fit_predict(panel, feats, yg, tr, infit, indev), TARGET, "test")
        log("  window %s done" % lab)
    means = {k: {m: p7.symbal_mean(ev[k], m) for m in p7.METRICS} for k in sets}
    log("pooled events=%d" % len(ev["B2_ctx+Ok+rel"]))
    log("=== pooled means ===")
    for k in sets:
        mm = means[k]; log("  %-22s NDCG=%.4f Hit=%.4f Mass=%.4f JS=%.4f" % (k, mm["ndcg3"], mm["hit1"], mm["mass3"], mm["js"]))
    pairs = [("B2_ctx+Ok+rel", "A_context"), ("B2_ctx+Ok+rel", "B_ctx+Ok"), ("B2_ctx+Ok+rel", "C_ctx+shuf+shufrel")]
    comps = p65.comparisons(ev, pairs)
    log("=== pooled bootstrap CI (B=4000) ===")
    for mdl, b in pairs:
        for metric in ["ndcg3", "hit1"]:
            s = comps["%s_vs_%s" % (mdl, b)][metric]["symbol_balanced_bootstrap"]
            sig = "SIG>0" if (s["ci05"] is not None and s["ci05"] > 0) else "ns"
            log("  %-38s %-6s %+0.4f [%+.4f, %+.4f] %s" % ("%s - %s" % (mdl, b), metric, s["observed"], s["ci05"], s["ci95"], sig))
    pathlib.Path(__file__).with_name("phase97_pooled_ablation_relational_result.json").write_text(json.dumps({"means": means, "comparisons": comps, "elapsed_sec": time.time() - t0}, indent=2))
    log("done %.1fs" % (time.time() - t0))
if __name__ == "__main__":
    main()
