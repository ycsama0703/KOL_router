"""Phase98: structure-as-GRAPH. Build lead-lag DiGraph from PIT history; per-KOL centralities
(out/in/net strength, reversed-PageRank originator-authority, HITS hub). Pooled 5-window
performance + bootstrap vs context. thr=0.50 first10 reach.
"""
from __future__ import annotations
import collections, json, math, pathlib, time
import numpy as np
import networkx as nx
import phase5_sentiment_reconstruction as p5
import phase7_origin_alert as p7
import phase33_origin_alert_ablation as p33
import phase65_pit_lightweight_2025_2026 as p65
import lightgbm as lgb
THR = 0.50; MEK = 8; TARGET = "log_future_reach"; N_GRID = [50, 100, 200, 400]
p65.BOOTSTRAP_B = 4000; p65.RNG = np.random.default_rng(98)
WINDOWS = [
    ("25.6-26.6", "2024-06-01", "2025-06-01", "2026-06-01", "2024-12-01"),
    ("24.6-25.6", "2023-06-01", "2024-06-01", "2025-06-01", "2023-12-01"),
    ("23.6-24.6", "2022-06-01", "2023-06-01", "2024-06-01", "2022-12-01"),
    ("22.6-23.6", "2021-06-01", "2022-06-01", "2023-06-01", "2021-12-01"),
    ("21.6-22.6", "2020-06-01", "2021-06-01", "2022-06-01", "2020-12-01"),
]
GRAPH = ["g_out", "g_in", "g_net", "g_pr", "g_hub"]
def _summarize95(vals):
    vals = np.asarray(vals, dtype=float); vals = vals[np.isfinite(vals)]
    if len(vals) == 0: return {"observed": None, "ci05": None, "ci95": None}
    return {"observed": float(np.mean(vals)), "ci05": float(np.quantile(vals, 0.025)), "ci95": float(np.quantile(vals, 0.975))}
p65.summarize = _summarize95
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
def graph_feats(all_rows, cutoff):
    ev = p5.first_by_event(all_rows, end=cutoff)
    lead = collections.defaultdict(float); nodes = set()
    for d in ev.values():
        if len(d) < 5: continue
        parts = sorted(d.items(), key=lambda kv: kv[1]["ts"]); ks = [k for k, _ in parts]
        for a in range(len(ks)):
            nodes.add(ks[a])
            for b in range(a + 1, len(ks)):
                lead[(ks[a], ks[b])] += 1.0
    G = nx.DiGraph(); G.add_nodes_from(nodes)
    for (a, b), w in lead.items(): G.add_edge(a, b, weight=w)
    out_s = dict(G.out_degree(weight="weight")); in_s = dict(G.in_degree(weight="weight"))
    try: pr = nx.pagerank(G.reverse(copy=True), weight="weight") if len(nodes) > 1 else {}
    except Exception: pr = {}
    try: hubs, _ = nx.hits(G, max_iter=500, normalized=True) if len(nodes) > 1 else ({}, {})
    except Exception: hubs = {}
    feats = {}
    for k in nodes:
        o = out_s.get(k, 0.0); i = in_s.get(k, 0.0)
        feats[k] = {"g_out": math.log1p(o), "g_in": math.log1p(i), "g_net": o - i, "g_pr": pr.get(k, 0.0), "g_hub": hubs.get(k, 0.0)}
    return feats
def attach_graph(panel, gf_by_block):
    for r in panel:
        gf = gf_by_block[r["split"]].get(r["origin_kol"], {"g_out": 0.0, "g_in": 0.0, "g_net": 0.0, "g_pr": 0.0, "g_hub": 0.0})
        for k, v in gf.items(): r[k] = v
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
    sets = {"A": base, "B": base + ["origin_ol"], "BG": base + ["origin_ol"] + GRAPH,
            "BG_pr": base + ["origin_ol", "g_pr"], "BG_hub": base + ["origin_ol", "g_hub"],
            "BG_net": base + ["origin_ol", "g_net"], "BG_io": base + ["origin_ol", "g_out", "g_in"]}
    ev = {k: [] for k in sets}
    for lab, tr_s, te_s, te_e, inner in WINDOWS:
        p65.TRAIN_START = tr_s; p65.TEST_END = te_e; p65.block_for_day = make_bfd(tr_s, te_s, te_e)
        states = states_for(tr_s, te_s, rows_by, emb_by, all_rows)
        gf = {"train": graph_feats(all_rows, tr_s), "test": graph_feats(all_rows, te_s)}
        p65.ORIGIN_WINDOW = {"name": "first10", "max_rank": 10}
        panel, _ = p65.build_pit_origin_panel(rows_by, emb_by, states)
        attach_graph(panel, gf)
        day = np.array([r["day"] for r in panel]); tr = np.array([r["split"] == "train" for r in panel])
        infit = tr & (day < inner); indev = tr & (day >= inner)
        yc = np.array([float(r.get(TARGET, float("nan"))) for r in panel], dtype=float)
        fin = np.isfinite(yc); yg = np.zeros(len(panel), dtype=int)
        qs = np.unique(np.quantile(yc[fin], np.linspace(0.0, 1.0, 33)))
        yg[fin] = np.clip(np.digitize(yc[fin], qs[1:-1]), 0, max(1, len(qs) - 2)).astype(int)
        for k, feats in sets.items():
            ev[k] += p65.event_rows_for_split(panel, fit_predict(panel, feats, yg, tr, infit, indev), TARGET, "test")
        log("  window %s done (graph nodes train=%d test=%d)" % (lab, len(gf["train"]), len(gf["test"])))
    means = {k: {m: p7.symbal_mean(ev[k], m) for m in p7.METRICS} for k in sets}
    log("pooled events=%d" % len(ev["A"]))
    log("=== pooled means ===")
    for k in sets:
        mm = means[k]; log("  %-20s NDCG=%.4f Hit=%.4f Mass=%.4f JS=%.4f" % (k, mm["ndcg3"], mm["hit1"], mm["mass3"], mm["js"]))
    pairs = [("BG", "A"), ("BG", "B"), ("BG_pr", "B"), ("BG_hub", "B"), ("BG_net", "B"), ("BG_io", "B")]
    comps = p65.comparisons(ev, pairs)
    log("=== pooled bootstrap CI (B=4000, 95% CI) ===")
    for mdl, b in pairs:
        for metric in ["ndcg3", "hit1"]:
            s = comps["%s_vs_%s" % (mdl, b)][metric]["symbol_balanced_bootstrap"]
            sig = "SIG>0" if (s["ci05"] is not None and s["ci05"] > 0) else "ns"
            log("  %-34s %-6s %+0.4f [%+.4f, %+.4f] %s" % ("%s - %s" % (mdl, b), metric, s["observed"], s["ci05"], s["ci95"], sig))
    pathlib.Path(__file__).with_name("phase100_graph_consolidate_result.json").write_text(json.dumps({"means": means, "comparisons": comps, "elapsed_sec": time.time() - t0}, indent=2))
    log("done %.1fs" % (time.time() - t0))
if __name__ == "__main__":
    main()
