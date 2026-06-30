"""Phase104: prior-art account-signal baselines vs our originator structure.
Romero IP-influence (computed on lead-lag graph), Yamada source-spreader (=hist_mean_log_adopt),
Zhou track-record (=hist_success_rate), vs our O_k / g_net.
Part1: single-window 25.6-26.6 standalone-score NDCG. Part2: pooled 5-window LambdaMART swap-signal bootstrap.
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
THR = 0.50; MEK = 8; TARGET = "log_future_reach"; N_GRID = [50, 100, 200, 400]
p65.BOOTSTRAP_B = 4000; p65.RNG = np.random.default_rng(104)
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
def graph_signals(all_rows, cutoff):
    # build lead-lag directed weighted graph; return g_net dict and Romero IP-influence dict
    ev = p5.first_by_event(all_rows, end=cutoff)
    lead = collections.defaultdict(float); nodes = set()
    for d in ev.values():
        if len(d) < 5: continue
        ks = [k for k, _ in sorted(d.items(), key=lambda kv: kv[1]["ts"])]
        for a in range(len(ks)):
            nodes.add(ks[a])
            for b in range(a + 1, len(ks)): lead[(ks[a], ks[b])] += 1.0
    nodes = sorted(nodes); idx = {k: i for i, k in enumerate(nodes)}; n = len(nodes)
    g_net = {}
    out = collections.defaultdict(float); inn = collections.defaultdict(float)
    for (a, b), w in lead.items(): out[a] += w; inn[b] += w
    for k in nodes: g_net[k] = out.get(k, 0.0) - inn.get(k, 0.0)
    # Romero IP fixed point
    ip = {k: 0.0 for k in nodes}
    if n > 1:
        W = np.zeros((n, n), dtype=float)
        for (a, b), w in lead.items(): W[idx[a], idx[b]] = w
        rs = W.sum(axis=1); cs = W.sum(axis=0)
        U = W / np.where(rs[:, None] > 0, rs[:, None], 1.0)   # U[i,j]=frac of i leads to j
        V = W / np.where(cs[None, :] > 0, cs[None, :], 1.0)   # V[i,j]=frac of j in-leads from i
        I = np.ones(n); P = np.ones(n)
        for _ in range(100):
            In = U @ P; Pn = V.T @ I
            s1 = In.sum(); s2 = Pn.sum()
            if s1 > 0: In = In / s1 * n
            if s2 > 0: Pn = Pn / s2 * n
            I, P = In, Pn
        for k in nodes: ip[k] = float(I[idx[k]])
    return g_net, ip
def attach(panel, sig):
    for r in panel:
        b = r["split"]; gn, ip = sig[b]
        r["g_net"] = gn.get(r["origin_kol"], 0.0); r["romero_ip"] = ip.get(r["origin_kol"], 0.0)
        # aliases for prior-art (already-present features)
        r["yamada_src"] = float(r.get("hist_mean_log_adopt", 0.0))
        r["zhou_track"] = float(r.get("hist_success_rate", 0.0))
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
    sets = {"A_context": base, "ours_Ok_gnet": base + ["origin_ol", "g_net"], "ctx_romero": base + ["romero_ip"]}
    ev = {k: [] for k in sets}
    part1 = None
    for lab, tr_s, te_s, te_e, inner in WINDOWS:
        p65.TRAIN_START = tr_s; p65.TEST_END = te_e; p65.block_for_day = make_bfd(tr_s, te_s, te_e)
        states = states_for(tr_s, te_s, rows_by, emb_by, all_rows)
        sig = {"train": graph_signals(all_rows, tr_s), "test": graph_signals(all_rows, te_s)}
        p65.ORIGIN_WINDOW = {"name": "first10", "max_rank": 10}
        panel, _ = p65.build_pit_origin_panel(rows_by, emb_by, states)
        attach(panel, sig)
        day = np.array([r["day"] for r in panel]); tr = np.array([r["split"] == "train" for r in panel])
        infit = tr & (day < inner); indev = tr & (day >= inner)
        yc = np.array([float(r.get(TARGET, float("nan"))) for r in panel], dtype=float)
        fin = np.isfinite(yc); yg = np.zeros(len(panel), dtype=int)
        qs = np.unique(np.quantile(yc[fin], np.linspace(0.0, 1.0, 33)))
        yg[fin] = np.clip(np.digitize(yc[fin], qs[1:-1]), 0, max(1, len(qs) - 2)).astype(int)
        # Part2 pooled
        for k, feats in sets.items():
            ev[k] += p65.event_rows_for_split(panel, fit_predict(panel, feats, yg, tr, infit, indev), TARGET, "test")
        # Part1 standalone (only recent window)
        if lab == "25.6-26.6":
            part1 = {}
            for sname in ["romero_ip", "yamada_src", "zhou_track", "origin_ol", "g_net"]:
                pred = np.array([float(r.get(sname, 0.0)) for r in panel])
                evs = p65.event_rows_for_split(panel, pred, TARGET, "test")
                part1[sname] = {m: p7.symbal_mean(evs, m) for m in p7.METRICS}
        log("  window %s done" % lab)
    log("=== PART1 standalone single-feature ranking (25.6-26.6) ===")
    for sname, mm in part1.items():
        log("  %-12s NDCG=%.4f Hit=%.4f" % (sname, mm["ndcg3"], mm["hit1"]))
    log("=== PART2 pooled means (5 win, %d events) ===" % len(ev["A_context"]))
    means = {k: {m: p7.symbal_mean(ev[k], m) for m in p7.METRICS} for k in sets}
    for k in sets: log("  %-14s NDCG=%.4f Hit=%.4f" % (k, means[k]["ndcg3"], means[k]["hit1"]))
    comps = p65.comparisons(ev, [("ours_Ok_gnet", "A_context"), ("ctx_romero", "A_context"), ("ours_Ok_gnet", "ctx_romero")])
    log("=== PART2 bootstrap 90% CI ===")
    for mdl, b in [("ours_Ok_gnet", "A_context"), ("ctx_romero", "A_context"), ("ours_Ok_gnet", "ctx_romero")]:
        for metric in ["ndcg3", "hit1"]:
            s = comps["%s_vs_%s" % (mdl, b)][metric]["symbol_balanced_bootstrap"]
            sig = "SIG>0" if (s["ci05"] is not None and s["ci05"] > 0) else "ns"
            log("  %-30s %-6s %+0.4f [%+.4f,%+.4f] %s" % ("%s-%s" % (mdl, b), metric, s["observed"], s["ci05"], s["ci95"], sig))
    pathlib.Path(__file__).with_name("phase104_priorart_baselines_result.json").write_text(json.dumps({"part1": part1, "part2_means": means, "comparisons": comps}, indent=2))
    log("done %.1fs" % (time.time() - t0))
if __name__ == "__main__":
    main()
