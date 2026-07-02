"""Phase106: is g_net just a proxy for posting ACTIVITY/VOLUME?

Referees (and Precursors&Laggards 2010, which rate-normalizes) will ask whether
g_net = out-in is merely "who posts/participates more". We test it four ways,
reusing the phase98 graph panel + phase65 listwise-GBDT eval + pooled bootstrap:

  (1) CORRELATION  Spearman(g_net, volume) where volume = graph matchups
      g_total=out+in and events-participated. Low corr => not a scale proxy.
  (2) INCREMENT OVER EXPLICIT ACTIVITY CONTROL (money test)
      does g_net still add NDCG on top of context + log_g_total (the pure
      volume analog of g_net)?  We control g_total ONLY, never (g_out,g_in)
      jointly, since those two determine g_net by construction.
  (3) P&L-STYLE RATE NORMALIZATION
      g_net_rate = (out-in)/(out+in) in [-1,1]; does the volume-free rate
      predict as well as raw g_net? If yes, directionality (not scale) drives it.
  (4) reported alongside: activity-alone increment, both-over-context.

thr=0.50 first10 reach; 5 rolling windows; symbol-balanced bootstrap B=4000.
"""
from __future__ import annotations
import collections, json, math, pathlib, time
import numpy as np
import networkx as nx
import phase5_sentiment_reconstruction as p5
import phase7_origin_alert as p7
import phase33_origin_alert_ablation as p33
import phase65_pit_lightweight_2025_2026 as p65
import phase98_graph_struct as p98
import lightgbm as lgb

THR = 0.50; MEK = 8; TARGET = "log_future_reach"; N_GRID = [50, 100, 200, 400]
p65.BOOTSTRAP_B = 4000; p65.RNG = np.random.default_rng(106)
WINDOWS = p98.WINDOWS
def log(m): print(m, flush=True)
p65.split_for_block = lambda b: b; p65.THR = THR


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3: return float("nan"), int(len(x))
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1]), int(len(x))


def graph_feats2(all_rows, cutoff):
    """Like p98.graph_feats but also emits volume (g_total, part_count) and
    the rate-normalized g_net_rate."""
    ev = p5.first_by_event(all_rows, end=cutoff)
    lead = collections.defaultdict(float); nodes = set()
    part = collections.Counter()
    for d in ev.values():
        if len(d) < 5: continue
        parts = sorted(d.items(), key=lambda kv: kv[1]["ts"]); ks = [k for k, _ in parts]
        for k in ks: part[k] += 1
        for a in range(len(ks)):
            nodes.add(ks[a])
            for b in range(a + 1, len(ks)):
                lead[(ks[a], ks[b])] += 1.0
    G = nx.DiGraph(); G.add_nodes_from(nodes)
    for (a, b), w in lead.items(): G.add_edge(a, b, weight=w)
    out_s = dict(G.out_degree(weight="weight")); in_s = dict(G.in_degree(weight="weight"))
    feats = {}
    for k in nodes:
        o = out_s.get(k, 0.0); i = in_s.get(k, 0.0); tot = o + i
        feats[k] = {
            "g_out": math.log1p(o), "g_in": math.log1p(i), "g_net": o - i,
            "g_total": tot, "log_g_total": math.log1p(tot),
            "g_net_rate": (o - i) / tot if tot > 0 else 0.0,
            "part_count": float(part.get(k, 0)),
        }
    return feats


def attach(panel, gf_by_block):
    default = {"g_out": 0.0, "g_in": 0.0, "g_net": 0.0, "g_total": 0.0,
               "log_g_total": 0.0, "g_net_rate": 0.0, "part_count": 0.0}
    for r in panel:
        gf = gf_by_block[r["split"]].get(r["origin_kol"], default)
        for k, v in gf.items(): r[k] = v


def main():
    t0 = time.time()
    rows_by, emb_by, all_rows = {}, {}, []
    for sym in p5.SYMS:
        r, e = p5.load_symbol(sym); rows_by[sym] = r; emb_by[sym] = e; all_rows.extend(r)
    p7.MIN_EVENT_KOLS = MEK
    base = p7.FEATURE_SETS["no_ol_strong"]
    sets = {
        "A_context":        base,
        "C_ctx+act":        base + ["log_g_total"],
        "G_ctx+gnet":       base + ["g_net"],
        "CG_ctx+act+gnet":  base + ["log_g_total", "g_net"],
        "R_ctx+gnetrate":   base + ["g_net_rate"],
    }
    ev = {k: [] for k in sets}
    # correlation accumulators (test-block graph, pooled over windows)
    corr = {"g_net": [], "g_total": [], "part_count": [], "g_net_rate": [], "g_out": [], "g_in": []}
    per_window_corr = []

    for lab, tr_s, te_s, te_e, inner in WINDOWS:
        p65.TRAIN_START = tr_s; p65.TEST_END = te_e; p65.block_for_day = p98.make_bfd(tr_s, te_s, te_e)
        states = p98.states_for(tr_s, te_s, rows_by, emb_by, all_rows)
        gf = {"train": graph_feats2(all_rows, tr_s), "test": graph_feats2(all_rows, te_s)}
        p65.ORIGIN_WINDOW = {"name": "first10", "max_rank": 10}
        panel, _ = p65.build_pit_origin_panel(rows_by, emb_by, states)
        attach(panel, gf)
        day = np.array([r["day"] for r in panel]); tr = np.array([r["split"] == "train" for r in panel])
        infit = tr & (day < inner); indev = tr & (day >= inner)
        yc = np.array([float(r.get(TARGET, float("nan"))) for r in panel], dtype=float)
        fin = np.isfinite(yc); yg = np.zeros(len(panel), dtype=int)
        qs = np.unique(np.quantile(yc[fin], np.linspace(0.0, 1.0, 33)))
        yg[fin] = np.clip(np.digitize(yc[fin], qs[1:-1]), 0, max(1, len(qs) - 2)).astype(int)
        for k, feats in sets.items():
            ev[k] += p65.event_rows_for_split(panel, p98.fit_predict(panel, feats, yg, tr, infit, indev), TARGET, "test")
        # per-account correlation on the test-block graph
        g = gf["test"]
        gn = [v["g_net"] for v in g.values()]
        gt = [v["g_total"] for v in g.values()]
        pc = [v["part_count"] for v in g.values()]
        gr = [v["g_net_rate"] for v in g.values()]
        go = [v["g_out"] for v in g.values()]
        gi = [v["g_in"] for v in g.values()]
        corr["g_net"] += gn; corr["g_total"] += gt; corr["part_count"] += pc
        corr["g_net_rate"] += gr; corr["g_out"] += go; corr["g_in"] += gi
        rho_tot, n_tot = spearman(gn, gt)
        rho_part, _ = spearman(gn, pc)
        per_window_corr.append({"window": lab, "n_accounts": n_tot,
                                "spearman_gnet_gtotal": rho_tot, "spearman_gnet_partcount": rho_part})
        log("  window %s done (graph nodes train=%d test=%d, |g_net,g_total| rho=%.3f)"
            % (lab, len(gf["train"]), len(gf["test"]), rho_tot))

    means = {k: {m: p7.symbal_mean(ev[k], m) for m in p7.METRICS} for k in sets}
    log("pooled events=%d" % len(ev["A_context"]))
    log("=== pooled means ===")
    for k in sets:
        mm = means[k]; log("  %-20s NDCG=%.4f Hit=%.4f" % (k, mm["ndcg3"], mm["hit1"]))

    # (1) CORRELATION
    r_gt, n_gt = spearman(corr["g_net"], corr["g_total"])
    r_pc, _ = spearman(corr["g_net"], corr["part_count"])
    r_go, _ = spearman(corr["g_net"], corr["g_out"])
    r_gi, _ = spearman(corr["g_net"], corr["g_in"])
    r_rate_gt, _ = spearman(corr["g_net_rate"], corr["g_total"])
    correlation = {
        "n_account_window": n_gt,
        "spearman_gnet_vs_gtotal": r_gt,
        "spearman_gnet_vs_partcount": r_pc,
        "spearman_gnet_vs_gout": r_go,
        "spearman_gnet_vs_gin": r_gi,
        "spearman_gnetrate_vs_gtotal": r_rate_gt,
        "per_window": per_window_corr,
    }
    log("=== (1) correlation g_net vs volume (pooled acct-windows n=%d) ===" % n_gt)
    log("  Spearman(g_net, g_total=out+in)   = %+.3f" % r_gt)
    log("  Spearman(g_net, part_count)       = %+.3f" % r_pc)
    log("  Spearman(g_net, g_out)            = %+.3f" % r_go)
    log("  Spearman(g_net, g_in)             = %+.3f" % r_gi)
    log("  Spearman(g_net_rate, g_total)     = %+.3f  (rate should be ~0 vs volume)" % r_rate_gt)

    # (2)-(4) bootstrap comparisons
    pairs = [
        ("G_ctx+gnet", "A_context"),        # baseline: g_net over context
        ("C_ctx+act", "A_context"),         # activity alone over context
        ("CG_ctx+act+gnet", "C_ctx+act"),   # *** money test: g_net over context+activity ***
        ("CG_ctx+act+gnet", "A_context"),   # both over context
        ("R_ctx+gnetrate", "A_context"),    # rate-normalized over context
    ]
    comps = p65.comparisons(ev, pairs)
    log("=== (2)-(4) pooled bootstrap CI (B=4000) ===")
    for mdl, b in pairs:
        for metric in ["ndcg3", "hit1"]:
            s = comps["%s_vs_%s" % (mdl, b)][metric]["symbol_balanced_bootstrap"]
            sig = "SIG>0" if (s["ci05"] is not None and s["ci05"] > 0) else "ns"
            log("  %-36s %-6s %+0.4f [%+.4f, %+.4f] %s"
                % ("%s - %s" % (mdl, b), metric, s["observed"], s["ci05"], s["ci95"], sig))

    # verdict helper
    money = comps["CG_ctx+act+gnet_vs_C_ctx+act"]["ndcg3"]["symbol_balanced_bootstrap"]
    rate = comps["R_ctx+gnetrate_vs_A_context"]["ndcg3"]["symbol_balanced_bootstrap"]
    verdict = {
        "gnet_adds_over_activity_control": bool(money["ci05"] is not None and money["ci05"] > 0),
        "gnet_over_activity_delta": money["observed"], "gnet_over_activity_ci": [money["ci05"], money["ci95"]],
        "rate_normalized_works": bool(rate["ci05"] is not None and rate["ci05"] > 0),
        "rate_delta": rate["observed"], "rate_ci": [rate["ci05"], rate["ci95"]],
        "spearman_gnet_gtotal": r_gt,
    }
    log("=== VERDICT ===")
    log("  g_net adds over context+activity?  %s (dNDCG=%+.4f)"
        % (verdict["gnet_adds_over_activity_control"], money["observed"]))
    log("  rate-normalized g_net works?       %s (dNDCG=%+.4f)"
        % (verdict["rate_normalized_works"], rate["observed"]))
    log("  => g_net is %s"
        % ("a DIRECTIONAL lead-lag signal, NOT a volume proxy"
           if verdict["gnet_adds_over_activity_control"] else "possibly a VOLUME PROXY — investigate"))

    pathlib.Path(__file__).with_name("phase106_gnet_activity_confound_result.json").write_text(
        json.dumps({"means": means, "correlation": correlation, "comparisons": comps,
                    "verdict": verdict, "elapsed_sec": time.time() - t0}, indent=2))
    log("done %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
