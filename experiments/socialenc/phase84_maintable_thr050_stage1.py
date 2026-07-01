"""Phase84 Stage1: full main table for thr=0.50 train1/test1 25.6-26.6 first10 reach.
Structured + surface + encoders (BERT/FinBERT/E5/BGE/Qwen3-4B). inner-CV alpha (paper-grade, NOT test-select).
Metrics: NDCG@3/Hit@1/Mass@3/JS (symbol-balanced). latency/tokens from phase31/42.
"""
from __future__ import annotations
import collections, json, math, pathlib, time
import numpy as np
import phase5_sentiment_reconstruction as p5
import phase7_origin_alert as p7
import phase29_origin_alert_text_surface_diagnostic as p29
import phase33_origin_alert_ablation as p33
import phase21_streaming_agent_encoder_baselines as p21
import phase39_qwen3_origin_alert_encoder_probe as p39
import phase65_pit_lightweight_2025_2026 as p65

ALPHAS = p65.RIDGE_ALPHAS
THR = 0.50; MEK = 8; TARGET = "log_future_reach"
TR_S, TE_S, TE_E = "2024-06-01", "2025-06-01", "2026-06-01"
INNER = "2025-01-01"
def log(m): print(m, flush=True)
p65.split_for_block = lambda b: b
p65.THR = THR
def block_for_day(day):
    if TR_S <= day < TE_S: return "train"
    if TE_S <= day < TE_E: return "test"
    return None
p65.block_for_day = block_for_day

def states_for(rows_by, emb_by, all_rows):
    states = {}
    for block, cutoff in [("train", TR_S), ("test", TE_S)]:
        hist_rows = p65.rows_before(all_rows, cutoff)
        meta = p5.compute_metadata(hist_rows)
        ol = p65.with_train_end(cutoff, p5.compute_oltrait, hist_rows, meta)
        hist = p65.with_train_end(cutoff, p7.compute_origin_history, rows_by, emb_by, THR)
        raw_ol = p65.with_train_end(cutoff, p33.compute_raw_oltrait, hist_rows, meta)
        states[block] = {"cutoff": cutoff, "meta": meta, "ol": ol, "hist": hist,
                         "raw_ol": raw_ol, "shuffled_ol": p65.make_shuffled_ol(ol, block)}
    return states

def symbal_ndcg_mask(rows, pred, mask):
    g = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if mask[i] and np.isfinite(pred[i]):
            g[r["event_id"]].append((r["sym"], float(r[TARGET]), float(pred[i])))
    return p7.symbal_mean(p65.event_metrics_from_groups(g), "ndcg3")

def test_metrics(rows, pred):
    evs = p65.event_rows_for_split(rows, pred, TARGET, "test")
    return ({k: p7.symbal_mean(evs, k) for k in p7.METRICS},
            len(evs), len(set(e["sym"] for e in evs)))

def ridge_from_X(rows, X, fit_mask, alpha):
    y = np.array([r[TARGET] for r in rows], dtype=float); fm = fit_mask & np.isfinite(y)
    if fm.sum() < X.shape[1] + 30: return None
    med, mu, sd = p65.train_standardizer(X, fm); Xs = p65.apply_standardizer(X, med, mu, sd)
    Xtr = np.column_stack([np.ones(fm.sum()), Xs[fm]]); lam = np.eye(Xtr.shape[1]) * alpha; lam[0, 0] = 0.0
    beta = np.linalg.solve(Xtr.T @ Xtr + lam, Xtr.T @ y[fm])
    return np.column_stack([np.ones(len(rows)), Xs]) @ beta

def innercv_cols(rows, feats, tr, infit, indev):
    best = (-1.0, None)
    for a in ALPHAS:
        pred = p65.ridge_predict_from_mask(rows, feats, TARGET, infit, a)
        if pred is None: continue
        nd = symbal_ndcg_mask(rows, pred, indev)
        if nd > best[0]: best = (nd, a)
    a = best[1] if best[1] is not None else 10.0
    return p65.ridge_predict_from_mask(rows, feats, TARGET, tr, a), a

def innercv_X(rows, X, tr, infit, indev):
    best = (-1.0, None)
    for a in ALPHAS:
        pred = ridge_from_X(rows, X, infit, a)
        if pred is None: continue
        nd = symbal_ndcg_mask(rows, pred, indev)
        if nd > best[0]: best = (nd, a)
    a = best[1] if best[1] is not None else 10.0
    return ridge_from_X(rows, X, tr, a), a

def main():
    t0 = time.time()
    rows_by, emb_by, all_rows = {}, {}, []
    for sym in p5.SYMS:
        r, e = p5.load_symbol(sym); rows_by[sym] = r; emb_by[sym] = e; all_rows.extend(r)
    p7.MIN_EVENT_KOLS = MEK; p65.TRAIN_START = TR_S; p65.TEST_END = TE_E
    states = states_for(rows_by, emb_by, all_rows)
    p65.ORIGIN_WINDOW = {"name": "first10", "max_rank": 10}
    panel, _ = p65.build_pit_origin_panel(rows_by, emb_by, states)
    day = np.array([r["day"] for r in panel])
    tr = np.array([r["split"] == "train" for r in panel])
    infit = tr & (day < INNER); indev = tr & (day >= INNER)
    log("panel rows=%d train=%d (infit=%d indev=%d) test=%d" % (len(panel), tr.sum(), infit.sum(), indev.sum(), (~tr).sum()))

    # cost lookup
    c31 = json.load(open("phase31_origin_alert_cost_benchmark_result.json"))["methods"]
    c42 = json.load(open("phase42_new_encoder_cost_benchmark_result.json"))["methods"]
    COST = {}
    for m, x in {**c31, **c42}.items():
        COST[m] = (x.get("batch_ms_per_query"), x.get("input_tokens", {}).get("mean"))

    results = []
    # structured + surface (ridge over feature columns / matrices)
    STRUCT = [("Scale","Follower","followers"),("Scale","Visibility","visibility"),
              ("Context","Rank/Time","rank_time"),("Context","Sentiment","sentiment"),
              ("Context","Novelty","novelty"),("Context","History","history"),
              ("Context","No-OL Strong","no_ol_strong"),("Origin Role","OL Only","ol_only"),
              ("Origin Role","OL-Origin","ol_origin")]
    for fam, lab, key in STRUCT:
        pred, a = innercv_cols(panel, p7.FEATURE_SETS[key], tr, infit, indev)
        m, ne, ns = test_metrics(panel, pred)
        results.append({"family": fam, "method": lab, "key": key, "alpha": a, "events": ne, "symbols": ns, **m,
                        "input_len": COST.get(key, (None, 0.0))[1], "latency": COST.get(key, (None, None))[0]})
        log("  %-14s alpha=%g NDCG=%.4f Hit=%.4f" % (lab, a, m["ndcg3"], m["hit1"]))
    mats = p29.build_matrices(panel)
    for fam, lab, key, ck in [("Surface","Symbol one-hot","symbol_onehot","symbol_onehot"),
                              ("Surface","Text surface","text_surface","text_surface"),
                              ("Surface","Symbol + surface","symbol_plus_surface","symbol_plus_surface")]:
        pred, a = innercv_X(panel, np.asarray(mats[key], dtype=float), tr, infit, indev)
        m, ne, ns = test_metrics(panel, pred)
        results.append({"family": fam, "method": lab, "key": key, "alpha": a, "events": ne, "symbols": ns, **m,
                        "input_len": COST.get(ck, (None, 0.0))[1], "latency": COST.get(ck, (None, None))[0]})
        log("  %-14s alpha=%g NDCG=%.4f Hit=%.4f" % (lab, a, m["ndcg3"], m["hit1"]))
    # encoders
    texts_for = lambda: [p21.clean_text(r.get("origin_text", "")) for r in panel]
    p21.BATCH_SIZE = 32
    ENC = [("BERT-origin text","bert_base","bert_base_origin_text","p21"),
           ("FinBERT-origin text","finbert_encoder","finbert_encoder_origin_text","p21"),
           ("E5-origin text","e5_base","e5_base_origin_text","p21"),
           ("BGE-origin text","bge_base","bge_base_origin_text","p21"),
           ("Qwen3-4B-origin text","qwen3_embedding_4b_st","qwen3_embedding_4b_st_origin_text","p39")]
    uniq = sorted(set(texts_for()))
    for lab, slug, ck, src in ENC:
        log("  encoding %s ..." % slug)
        if src == "p39":
            cache = p39.encode_qwen3_st_texts(p39.QWEN3_4B_SLUG, p39.QWEN3_4B_CONFIG, uniq)
        else:
            cache = p21.encode_texts(slug, p21.MODEL_CONFIGS[slug], uniq)
        dim = len(next(iter(cache.values())))
        X = np.array([cache.get(p21.clean_text(r.get("origin_text", "")), np.zeros(dim, np.float32)) for r in panel], dtype=float)
        pred, a = innercv_X(panel, X, tr, infit, indev)
        m, ne, ns = test_metrics(panel, pred)
        results.append({"family": "Text Encoder", "method": lab, "key": slug, "alpha": a, "events": ne, "symbols": ns, **m,
                        "input_len": COST.get(ck, (None, 0.0))[1], "latency": COST.get(ck, (None, None))[0]})
        log("  %-22s dim=%d alpha=%g NDCG=%.4f Hit=%.4f" % (lab, dim, a, m["ndcg3"], m["hit1"]))

    base = next(r for r in results if r["key"] == "no_ol_strong")
    for r in results:
        r["dNDCG"] = r["ndcg3"] - base["ndcg3"]; r["dHit"] = r["hit1"] - base["hit1"]
    out = {"task": "phase84_maintable_thr050_stage1", "config": {"thr": THR, "train": [TR_S, TE_S], "test": [TE_S, TE_E],
           "window": "first10", "mek": MEK, "target": TARGET, "alpha": "inner-CV on train (split %s)" % INNER},
           "rows": results, "elapsed_sec": time.time() - t0}
    pathlib.Path(__file__).with_name("phase84_maintable_thr050_stage1_result.json").write_text(json.dumps(out, indent=2))
    def fmt(x, d=3): return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (("%."+str(d)+"f") % x)
    lines = ["# Phase84 main table thr=0.50 train1/test1 25.6-26.6 (Stage1: structured+surface+encoders)", "",
             "train 2024-06..2025-06, test 2025-06..2026-06, window=first10, target=reach, alpha=inner-CV.", "",
             "| Family | Method | Events | Symbols | NDCG@3 | Hit@1 | Mass@3 | JS | dNDCG | dHit | InputLen | Latency ms/q |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append("| %s | %s | %d | %d | %s | %s | %s | %s | %+0.4f | %+0.4f | %s | %s |" % (
            r["family"], r["method"], r["events"], r["symbols"], fmt(r["ndcg3"]), fmt(r["hit1"]), fmt(r["mass3"]),
            fmt(r["js"]), r["dNDCG"], r["dHit"], fmt(r["input_len"], 1), fmt(r["latency"], 4)))
    pathlib.Path(__file__).with_name("phase84_maintable_thr050_stage1_table.md").write_text("\n".join(lines) + "\n")
    log("\ndone %.1fs" % (time.time() - t0))

if __name__ == "__main__":
    main()
