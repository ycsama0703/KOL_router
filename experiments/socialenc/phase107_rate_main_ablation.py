"""Phase107: g_net_rate vs raw g_net for the MAIN model, same 口径 as phase98/100.

Reports both:
  - MAIN experiment 口径: single main window (25.6-26.6) pooled means.
  - ABLATION 口径: 5-window pooled means + symbol-balanced bootstrap CI (B=4000).

Feature sets mirror the real deployed model { context + O_k + graph-feature }:
  A_context        = no_ol_strong
  B_ctx+Ok         = + origin_ol
  BG_gnet          = + origin_ol + g_net        (current final model)
  BG_rate          = + origin_ol + g_net_rate   (rate-normalized candidate)

We do NOT modify any doc/model here — this is a read-only measurement to compare
the raw vs rate encoding under the exact main/ablation protocols.
"""
from __future__ import annotations
import collections, json, pathlib, time
import numpy as np
import phase5_sentiment_reconstruction as p5
import phase7_origin_alert as p7
import phase65_pit_lightweight_2025_2026 as p65
import phase98_graph_struct as p98
import phase106_gnet_activity_confound as p106

THR = 0.50; MEK = 8; TARGET = "log_future_reach"
p65.BOOTSTRAP_B = 4000; p65.RNG = np.random.default_rng(107)
WINDOWS = p98.WINDOWS  # index 0 = "25.6-26.6" = main window
MAIN_LAB = WINDOWS[0][0]
def log(m): print(m, flush=True)
p65.split_for_block = lambda b: b; p65.THR = THR


def main():
    t0 = time.time()
    rows_by, emb_by, all_rows = {}, {}, []
    for sym in p5.SYMS:
        r, e = p5.load_symbol(sym); rows_by[sym] = r; emb_by[sym] = e; all_rows.extend(r)
    p7.MIN_EVENT_KOLS = MEK
    base = p7.FEATURE_SETS["no_ol_strong"]
    sets = {
        "A_context":  base,
        "B_ctx+Ok":   base + ["origin_ol"],
        "BG_gnet":    base + ["origin_ol", "g_net"],
        "BG_rate":    base + ["origin_ol", "g_net_rate"],
    }
    ev = {k: [] for k in sets}        # pooled (all 5 windows)
    ev_main = {k: [] for k in sets}   # main window only

    for lab, tr_s, te_s, te_e, inner in WINDOWS:
        p65.TRAIN_START = tr_s; p65.TEST_END = te_e; p65.block_for_day = p98.make_bfd(tr_s, te_s, te_e)
        states = p98.states_for(tr_s, te_s, rows_by, emb_by, all_rows)
        gf = {"train": p106.graph_feats2(all_rows, tr_s), "test": p106.graph_feats2(all_rows, te_s)}
        p65.ORIGIN_WINDOW = {"name": "first10", "max_rank": 10}
        panel, _ = p65.build_pit_origin_panel(rows_by, emb_by, states)
        p106.attach(panel, gf)
        day = np.array([r["day"] for r in panel]); tr = np.array([r["split"] == "train" for r in panel])
        infit = tr & (day < inner); indev = tr & (day >= inner)
        yc = np.array([float(r.get(TARGET, float("nan"))) for r in panel], dtype=float)
        fin = np.isfinite(yc); yg = np.zeros(len(panel), dtype=int)
        qs = np.unique(np.quantile(yc[fin], np.linspace(0.0, 1.0, 33)))
        yg[fin] = np.clip(np.digitize(yc[fin], qs[1:-1]), 0, max(1, len(qs) - 2)).astype(int)
        for k, feats in sets.items():
            e = p65.event_rows_for_split(panel, p98.fit_predict(panel, feats, yg, tr, infit, indev), TARGET, "test")
            ev[k] += e
            if lab == MAIN_LAB:
                ev_main[k] += e
        log("  window %s done" % lab)

    main_means = {k: {m: p7.symbal_mean(ev_main[k], m) for m in p7.METRICS} for k in sets}
    pooled_means = {k: {m: p7.symbal_mean(ev[k], m) for m in p7.METRICS} for k in sets}

    log("=== MAIN experiment 口径 (single window %s, n=%d events) ===" % (MAIN_LAB, len(ev_main["A_context"])))
    for k in sets:
        mm = main_means[k]; log("  %-14s NDCG=%.4f Hit=%.4f" % (k, mm["ndcg3"], mm["hit1"]))
    log("=== ABLATION 口径 (pooled 5 windows, n=%d events) ===" % len(ev["A_context"]))
    for k in sets:
        mm = pooled_means[k]; log("  %-14s NDCG=%.4f Hit=%.4f" % (k, mm["ndcg3"], mm["hit1"]))

    pairs = [
        ("BG_gnet", "B_ctx+Ok"),   # current structure increment over context+Ok
        ("BG_rate", "B_ctx+Ok"),   # rate structure increment over context+Ok
        ("BG_rate", "BG_gnet"),    # rate vs raw head-to-head
        ("BG_gnet", "A_context"),
        ("BG_rate", "A_context"),
    ]
    comps = p65.comparisons(ev, pairs)
    log("=== pooled bootstrap CI (B=4000) ===")
    for mdl, b in pairs:
        for metric in ["ndcg3", "hit1"]:
            s = comps["%s_vs_%s" % (mdl, b)][metric]["symbol_balanced_bootstrap"]
            sig = "SIG>0" if (s["ci05"] is not None and s["ci05"] > 0) else "ns"
            log("  %-28s %-6s %+0.4f [%+.4f, %+.4f] %s"
                % ("%s - %s" % (mdl, b), metric, s["observed"], s["ci05"], s["ci95"], sig))

    pathlib.Path(__file__).with_name("phase107_rate_main_ablation_result.json").write_text(
        json.dumps({"main_window": MAIN_LAB, "main_means": main_means, "pooled_means": pooled_means,
                    "comparisons": comps, "elapsed_sec": time.time() - t0}, indent=2))
    log("done %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
