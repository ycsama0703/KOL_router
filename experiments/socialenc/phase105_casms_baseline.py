"""Phase105: best-effort CasMS generation-stage baseline.
CasMS zero-observation inputs = message text (BERT/Qwen) + originator static-graph node embedding (node2vec).
We feed both into the SAME listwise LambdaMART on our frame-reach ranking task (fair adaptation;
we lack a follow graph -> use PIT co-occurrence graph for node2vec; personalized-retweet term approximated
by the tree learning text x graph interactions). Single window 25.6-26.6, thr=0.50 first10 reach.
"""
from __future__ import annotations
import collections, json, math, pathlib, time
import numpy as np
import networkx as nx
from node2vec import Node2Vec
import phase5_sentiment_reconstruction as p5
import phase7_origin_alert as p7
import phase33_origin_alert_ablation as p33
import phase21_streaming_agent_encoder_baselines as p21
import phase39_qwen3_origin_alert_encoder_probe as p39
import phase65_pit_lightweight_2025_2026 as p65
import lightgbm as lgb
THR = 0.50; MEK = 8; TARGET = "log_future_reach"; N_GRID = [50, 100, 200, 400]
TR_S, TE_S, TE_E = "2024-06-01", "2025-06-01", "2026-06-01"; INNER = "2025-01-01"; NVDIM = 64
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
        hr = p65.rows_before(all_rows, cutoff); meta = p5.compute_metadata(hr)
        ol = p65.with_train_end(cutoff, p5.compute_oltrait, hr, meta)
        hist = p65.with_train_end(cutoff, p7.compute_origin_history, rows_by, emb_by, THR)
        raw = p65.with_train_end(cutoff, p33.compute_raw_oltrait, hr, meta)
        states[block] = {"cutoff": cutoff, "meta": meta, "ol": ol, "hist": hist, "raw_ol": raw, "shuffled_ol": p65.make_shuffled_ol(ol, block)}
    return states
def node2vec_emb(all_rows, cutoff, dim=NVDIM):
    ev = p5.first_by_event(all_rows, end=cutoff)
    co = collections.defaultdict(float); nodes = set()
    for d in ev.values():
        if len(d) < 5: continue
        ks = list(d.keys())
        for i in range(len(ks)):
            nodes.add(ks[i])
            for j in range(i + 1, len(ks)):
                a, b = sorted((ks[i], ks[j])); co[(a, b)] += 1.0
    G = nx.Graph(); G.add_nodes_from(nodes)
    for (a, b), w in co.items(): G.add_edge(a, b, weight=w)
    if G.number_of_nodes() < 2: return {}, dim
    n2v = Node2Vec(G, dimensions=dim, walk_length=20, num_walks=10, workers=8, weight_key="weight", quiet=True)
    model = n2v.fit(window=5, min_count=1, workers=8)
    return {k: np.asarray(model.wv[str(k)], dtype=float) for k in nodes if str(k) in model.wv}, dim
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
def run_X(panel, X, yg, tr, infit, indev):
    o_inf, s_inf = grouped([i for i in range(len(panel)) if infit[i]], panel)
    best = (-1.0, None)
    for n in N_GRID:
        nd = symbal_ndcg_mask(panel, fit_lgbm(X, yg, o_inf, s_inf, n).predict(X), indev)
        if nd > best[0]: best = (nd, n)
    n = best[1] or 100
    o_tr, s_tr = grouped([i for i in range(len(panel)) if tr[i]], panel)
    pred = fit_lgbm(X, yg, o_tr, s_tr, n).predict(X)
    evs = p65.event_rows_for_split(panel, pred, TARGET, "test")
    return {m: p7.symbal_mean(evs, m) for m in p7.METRICS}, n
def main():
    t0 = time.time()
    rows_by, emb_by, all_rows = {}, {}, []
    for sym in p5.SYMS:
        r, e = p5.load_symbol(sym); rows_by[sym] = r; emb_by[sym] = e; all_rows.extend(r)
    p7.MIN_EVENT_KOLS = MEK; p65.TRAIN_START = TR_S; p65.TEST_END = TE_E
    states = states_for(rows_by, emb_by, all_rows)
    log("node2vec PIT graphs ...")
    nv_tr, _ = node2vec_emb(all_rows, TR_S); nv_te, _ = node2vec_emb(all_rows, TE_S)
    log("node2vec nodes train=%d test=%d" % (len(nv_tr), len(nv_te)))
    p65.ORIGIN_WINDOW = {"name": "first10", "max_rank": 10}
    panel, _ = p65.build_pit_origin_panel(rows_by, emb_by, states)
    day = np.array([r["day"] for r in panel]); tr = np.array([r["split"] == "train" for r in panel])
    infit = tr & (day < INNER); indev = tr & (day >= INNER)
    yc = np.array([float(r.get(TARGET, float("nan"))) for r in panel], dtype=float)
    fin = np.isfinite(yc); yg = np.zeros(len(panel), dtype=int)
    qs = np.unique(np.quantile(yc[fin], np.linspace(0.0, 1.0, 33)))
    yg[fin] = np.clip(np.digitize(yc[fin], qs[1:-1]), 0, max(1, len(qs) - 2)).astype(int)
    # Qwen text emb
    cache = p21.load_embedding_cache(p39.QWEN3_4B_SLUG); qdim = len(next(iter(cache.values())))
    Xq = np.array([cache.get(p21.clean_text(r.get("origin_text", "")), np.zeros(qdim, np.float32)) for r in panel], dtype=float)
    # node2vec per row (by block)
    Xnv = np.zeros((len(panel), NVDIM), dtype=float)
    for i, r in enumerate(panel):
        nv = nv_tr if r["split"] == "train" else nv_te
        v = nv.get(r["origin_kol"]);
        if v is not None: Xnv[i] = v
    Xcasms = np.hstack([Xq, Xnv])
    Xours = p65.matrix(panel, p7.FEATURE_SETS["no_ol_strong"] + ["origin_ol"])  # +g_net below
    # add g_net to ours via graph net-degree (reuse simple degree)
    ev = p5.first_by_event(all_rows, end=TE_S); out = collections.defaultdict(float); inn = collections.defaultdict(float)
    for d in ev.values():
        if len(d) < 5: continue
        ks = [k for k, _ in sorted(d.items(), key=lambda kv: kv[1]["ts"])]
        for a in range(len(ks)):
            for b in range(a + 1, len(ks)): out[ks[a]] += 1.0; inn[ks[b]] += 1.0
    evtr = p5.first_by_event(all_rows, end=TR_S); outr = collections.defaultdict(float); intr = collections.defaultdict(float)
    for d in evtr.values():
        if len(d) < 5: continue
        ks = [k for k, _ in sorted(d.items(), key=lambda kv: kv[1]["ts"])]
        for a in range(len(ks)):
            for b in range(a + 1, len(ks)): outr[ks[a]] += 1.0; intr[ks[b]] += 1.0
    gnet = np.array([((out.get(r["origin_kol"],0)-inn.get(r["origin_kol"],0)) if r["split"]=="test" else (outr.get(r["origin_kol"],0)-intr.get(r["origin_kol"],0))) for r in panel]).reshape(-1,1)
    Xours = np.hstack([Xours, gnet])
    log("dims: Qwen=%d node2vec=%d casms=%d ours=%d  test=%d" % (Xq.shape[1], Xnv.shape[1], Xcasms.shape[1], Xours.shape[1], int((~tr).sum())))
    res = {}
    for name, X in [("Qwen-only (text)", Xq), ("node2vec-only (graph pos)", Xnv), ("CasMS-style (text+node2vec)", Xcasms), ("ours {context+O_k+g_net}", Xours)]:
        m, n = run_X(panel, X, yg, tr, infit, indev); res[name] = m
        log("  %-30s n=%d NDCG=%.4f Hit=%.4f Mass=%.4f JS=%.4f" % (name, n, m["ndcg3"], m["hit1"], m["mass3"], m["js"]))
    pathlib.Path(__file__).with_name("phase105_casms_baseline_result.json").write_text(json.dumps(res, indent=2))
    log("done %.1fs" % (time.time() - t0))
if __name__ == "__main__":
    main()
