"""Phase51: GRAPH-router listwise dilution (upgrade of phase50).

Same downstream LLM (DeepSeek), same listwise attention-routing / capture metric
as phase50, but the shortlister is now our REAL model:
  ol  = LambdaMART on {context(no_ol_strong) + origin_ol + g_net}   <- graph router
  no_ol = LambdaMART on {context}                                    <- context baseline
  follower = origin_logfoll scalar                                   <- follower baseline
  random   = random shortlist

Panel/window/graph machinery is phase98's (main window 25.6-26.6, thr=0.50,
first10, reach). Eligible decision days are TEST-split days with >= max(K)
candidates; we take the densest LIMIT_DAYS of them to bound LLM cost.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import time

import numpy as np

import phase5_sentiment_reconstruction as p5
import phase7_origin_alert as p7
import phase21_streaming_agent_encoder_baselines as p21
import phase28_origin_alert_encoder_baselines as p28
import phase65_pit_lightweight_2025_2026 as p65
import phase98_graph_struct as g
import phase104_priorart_baselines as p104
import phase50_deepseek_listwise_dilution as p50

ENCODER_SLUGS = ["bert_base", "finbert_encoder", "e5_base", "bge_base", "e5_mistral_7b_instruct"]

TARGET = "log_future_reach"
K_VALUES = [10, 20, 30]
SHORTLISTS = [10, 20]
SELECT_R = 3
RNG_SEED = 20260630
LIMIT_DAYS_DEFAULT = 40

# Route phase50's LLM cache to a fresh, separate store (panel/uids differ).
p50.SLUG = "deepseek_official__deepseek-v4-flash__graph_listwise_v1"
p50.CACHE_DIR = pathlib.Path.home() / "workspace/projects/alphagap/experiments/socialenc/phase51_graph_listwise_cache"

OUT = pathlib.Path.home() / "workspace/projects/alphagap/experiments/socialenc/phase51_graph_listwise_dilution_result.json"
TABLE_OUT = pathlib.Path.home() / "workspace/projects/alphagap/experiments/socialenc/phase51_graph_listwise_dilution_table.md"


def log(m):
    print(m, flush=True)


def build_graph_panel():
    rows_by, emb_by, all_rows = {}, {}, []
    for sym in p5.SYMS:
        r, e = p5.load_symbol(sym)
        rows_by[sym] = r
        emb_by[sym] = e
        all_rows.extend(r)
    p7.MIN_EVENT_KOLS = g.MEK
    lab, tr_s, te_s, te_e, inner = g.WINDOWS[0]  # 25.6-26.6 main window
    p65.TRAIN_START = tr_s
    p65.TEST_END = te_e
    p65.block_for_day = g.make_bfd(tr_s, te_s, te_e)
    states = g.states_for(tr_s, te_s, rows_by, emb_by, all_rows)
    gf = {"train": g.graph_feats(all_rows, tr_s), "test": g.graph_feats(all_rows, te_s)}
    p65.ORIGIN_WINDOW = {"name": "first10", "max_rank": 10}
    panel, _ = p65.build_pit_origin_panel(rows_by, emb_by, states)
    g.attach_graph(panel, gf)
    # prior-art: Romero IP-influence (computed on lead-lag graph), plus Yamada/Zhou aliases
    ip = {"train": p104.graph_signals(all_rows, tr_s)[1], "test": p104.graph_signals(all_rows, te_s)[1]}
    for r in panel:
        r["romero_ip"] = ip[r["split"]].get(r["origin_kol"], 0.0)
        r["yamada_src"] = float(r.get("hist_mean_log_adopt", 0.0))
        r["zhou_track"] = float(r.get("hist_success_rate", 0.0))
    log(f"  panel rows={len(panel)} window={lab} graph_nodes train={len(gf['train'])} test={len(gf['test'])}")
    return panel, inner, all_rows, tr_s, te_s


def encoder_score(panel, inner, slug):
    p28.INNER_SPLIT = inner
    p28.ORIGIN_WINDOW = {"name": "first10", "max_rank": 10}
    cache = p21.load_embedding_cache(slug)
    if not cache:
        log(f"  WARN no emb cache for {slug} -- skip")
        return None
    matrix = p28.origin_text_matrix(panel, cache)
    pred = p28.fit_matrix_score(panel, matrix, TARGET, f"{slug}_origin_text", {})
    return None if pred is None else np.asarray(pred, dtype=float)


def fit_predict_matrix(panel, X, yg, tr, infit, indev):
    o_inf, s_inf = g.grouped([i for i in range(len(panel)) if infit[i]], panel)
    best = (-1.0, None)
    for n in g.N_GRID:
        nd = g.symbal_ndcg_mask(panel, g.fit_lgbm(X, yg, o_inf, s_inf, n).predict(X), indev)
        if nd > best[0]:
            best = (nd, n)
    n = best[1] or 100
    o_tr, s_tr = g.grouped([i for i in range(len(panel)) if tr[i]], panel)
    return g.fit_lgbm(X, yg, o_tr, s_tr, n).predict(X)


def casms_score(panel, all_rows, tr_s, te_s, yg, tr, infit, indev):
    import phase39_qwen3_origin_alert_encoder_probe as p39
    import phase105_casms_baseline as p105
    log("  building CasMS-style shortlister (qwen text + node2vec PIT graph -> LambdaMART) ...")
    nv_tr, _ = p105.node2vec_emb(all_rows, tr_s)
    nv_te, _ = p105.node2vec_emb(all_rows, te_s)
    cache = p21.load_embedding_cache(p39.QWEN3_4B_SLUG)
    if not cache:
        log("  WARN no qwen cache for CasMS -- skip")
        return None
    qdim = len(next(iter(cache.values())))
    Xq = np.array([cache.get(p21.clean_text(r.get("origin_text", "")), np.zeros(qdim, np.float32)) for r in panel], dtype=float)
    Xnv = np.zeros((len(panel), p105.NVDIM), dtype=float)
    for i, r in enumerate(panel):
        nv = nv_tr if r["split"] == "train" else nv_te
        v = nv.get(r["origin_kol"])
        if v is not None:
            Xnv[i] = v
    Xcasms = np.hstack([Xq, Xnv])
    return np.asarray(fit_predict_matrix(panel, Xcasms, yg, tr, infit, indev), dtype=float)


def routing_scores(panel, inner, include_qwen=False, include_encoders=False, include_priorart=False,
                   include_casms=False, all_rows=None, tr_s=None, te_s=None):
    base = p7.FEATURE_SETS["no_ol_strong"]
    feat_ol = base + ["origin_ol", "g_net"]
    feat_noool = base
    day = np.array([r["day"] for r in panel])
    tr = np.array([r["split"] == "train" for r in panel])
    infit = tr & (day < inner)
    indev = tr & (day >= inner)
    yc = np.array([float(r.get(TARGET, float("nan"))) for r in panel], dtype=float)
    fin = np.isfinite(yc)
    yg = np.zeros(len(panel), dtype=int)
    qs = np.unique(np.quantile(yc[fin], np.linspace(0.0, 1.0, 33)))
    yg[fin] = np.clip(np.digitize(yc[fin], qs[1:-1]), 0, max(1, len(qs) - 2)).astype(int)
    log("  fitting graph router (context+origin_ol+g_net) ...")
    ol_pred = g.fit_predict(panel, feat_ol, yg, tr, infit, indev)
    log("  fitting context baseline (no_ol_strong) ...")
    noool_pred = g.fit_predict(panel, feat_noool, yg, tr, infit, indev)
    log("  fitting Ranking-Algo family rep: LambdaMART {context+O_k} (no graph) ...")
    lmart_ctxok_pred = g.fit_predict(panel, base + ["origin_ol"], yg, tr, infit, indev)
    foll = np.array([float(r.get("origin_logfoll", float("nan"))) for r in panel], dtype=float)
    out = {
        "ol": {i: float(ol_pred[i]) for i in range(len(panel)) if np.isfinite(ol_pred[i])},
        "no_ol": {i: float(noool_pred[i]) for i in range(len(panel)) if np.isfinite(noool_pred[i])},
        "lmart_ctxOk": {i: float(lmart_ctxok_pred[i]) for i in range(len(panel)) if np.isfinite(lmart_ctxok_pred[i])},
        "follower": {i: float(foll[i]) for i in range(len(panel)) if np.isfinite(foll[i])},
    }
    def add_vec(name, vec):
        if vec is not None:
            out[name] = {i: float(vec[i]) for i in range(len(panel)) if np.isfinite(vec[i])}

    if include_qwen:
        log("  fitting SOTA text shortlister (qwen3-embedding-4b ridge readout) ...")
        add_vec("qwen3_4b", encoder_score(panel, inner, "qwen3_embedding_4b_st"))
    if include_encoders:
        for slug in ENCODER_SLUGS:
            log(f"  fitting encoder shortlister ({slug} ridge readout) ...")
            add_vec(slug, encoder_score(panel, inner, slug))
    if include_priorart:
        log("  building prior-art scalar shortlisters (romero_ip / yamada_src / zhou_track) ...")
        add_vec("romero_ip", np.array([float(r.get("romero_ip", float("nan"))) for r in panel], dtype=float))
        add_vec("yamada_src", np.array([float(r.get("yamada_src", float("nan"))) for r in panel], dtype=float))
        add_vec("zhou_track", np.array([float(r.get("zhou_track", float("nan"))) for r in panel], dtype=float))
    if include_casms:
        add_vec("casms", casms_score(panel, all_rows, tr_s, te_s, yg, tr, infit, indev))
    return out


def build_tasks(panel, scores, limit_days):
    by_day = collections.defaultdict(list)
    for index, row in enumerate(panel):
        if row["split"] == "test" and np.isfinite(float(row.get(TARGET, float("nan")))):
            by_day[str(row["day"])].append(index)
    eligible_days = [d for d, idx in by_day.items() if len(idx) >= max(K_VALUES)]
    eligible_days.sort(key=lambda d: (-len(by_day[d]), d))  # densest first
    if limit_days is not None:
        eligible_days = eligible_days[:limit_days]
    rng = random.Random(RNG_SEED)
    tasks = []
    for day in eligible_days:
        indices = sorted(by_day[day], key=lambda i: (panel[i]["sym"], panel[i]["event_id"], panel[i]["frame_j"], p50.row_uid_for_cache[i]))
        rng.shuffle(indices)
        full_pool = indices[:max(K_VALUES)]
        for k in K_VALUES:
            tasks.append({"day": day, "policy": f"full_k{k}", "policy_family": "full_llm",
                          "k": k, "shortlist": None, "candidate_indices": full_pool[:k]})
        for b in SHORTLISTS:
            rnd = full_pool[:]
            rng.shuffle(rnd)
            tasks.append({"day": day, "policy": f"random_b{b}", "policy_family": "random_to_llm",
                          "k": max(K_VALUES), "shortlist": b, "candidate_indices": rnd[:b]})
            for name in scores.keys():
                tasks.append({"day": day, "policy": f"{name}_b{b}", "policy_family": f"{name}_to_llm",
                              "k": max(K_VALUES), "shortlist": b,
                              "candidate_indices": p50.select_top(full_pool, scores[name], b)})
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-days", type=int, default=LIMIT_DAYS_DEFAULT)
    parser.add_argument("--limit-calls", type=int, default=None)
    parser.add_argument("--include-qwen", action="store_true")
    parser.add_argument("--include-encoders", action="store_true")
    parser.add_argument("--include-priorart", action="store_true")
    parser.add_argument("--include-casms", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    started = time.time()

    log("[1/5] Building GRAPH panel (main window 25.6-26.6, thr=0.50)")
    panel, inner, all_rows, tr_s, te_s = build_graph_panel()
    p50.row_uid_for_cache = {index: p50.row_uid(row) for index, row in enumerate(panel)}

    log("[2/5] Fitting routing scores")
    scores = routing_scores(panel, inner, args.include_qwen, args.include_encoders, args.include_priorart,
                            args.include_casms, all_rows, tr_s, te_s)
    log(f"  shortlister policies: {sorted(scores.keys())}")

    tasks = build_tasks(panel, scores, args.limit_days)
    full_pool_by_day = {}
    for task in tasks:
        if task["policy"] == f"full_k{max(K_VALUES)}":
            full_pool_by_day[task["day"]] = task["candidate_indices"]
    for task in tasks:
        task["full_pool_indices"] = full_pool_by_day[task["day"]]
    log(f"[3/5] tasks={len(tasks)} eligible_days={len(set(t['day'] for t in tasks))} policies={len(scores)}")
    if args.dry_run:
        print(json.dumps({"tasks": len(tasks), "by_policy": dict(collections.Counter(t["policy"] for t in tasks))}, indent=2))
        return

    api_key = p50.get_api_key()
    cache = p50.load_cache()
    log(f"[4/5] cache={len(cache)}")
    records = []
    calls = 0
    for i, task in enumerate(tasks, 1):
        key = p50.cache_key(task["day"], task["policy"], task["candidate_indices"], SELECT_R)
        if key in cache:
            record = cache[key]
        else:
            if args.limit_calls is not None and calls >= args.limit_calls:
                log(f"  reached limit_calls={args.limit_calls}")
                break
            prompt = p50.build_prompt(panel, task["candidate_indices"], SELECT_R)
            max_tokens = 512 + 160 * SELECT_R
            content, obj, latency_ms, error = p50.call_deepseek(api_key, prompt, max_tokens)
            calls += 1
            usage = obj.get("usage") if isinstance(obj, dict) else {}
            selected, parse_error = p50.parse_selection(content, set(range(len(task["candidate_indices"]))), SELECT_R)
            record = {
                "cache_key": key, "model": p50.MODEL, "policy": task["policy"],
                "policy_family": task["policy_family"], "day": task["day"], "k": task["k"],
                "shortlist": task["shortlist"], "r": SELECT_R,
                "candidate_uids": [p50.row_uid_for_cache[index] for index in task["candidate_indices"]],
                "response": content, "selected": selected, "parse_error": parse_error or error,
                "latency_ms": latency_ms, "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"), "total_tokens": usage.get("total_tokens"),
            }
            p50.append_cache([record])
            cache[key] = record
            if calls % 10 == 0:
                log(f"  calls={calls} tasks={i}/{len(tasks)}")
        selected = record.get("selected") or []
        metrics = p50.evaluate_task(panel, task, selected) if len(selected) >= SELECT_R else {}
        records.append({**task, **record, **metrics})

    log("[5/5] Summarizing")
    by_policy_records = collections.defaultdict(list)
    for record in records:
        by_policy_records[record["policy"]].append(record)
    by_policy = {policy: p50.summarize_metrics(vals) for policy, vals in by_policy_records.items()}
    result = {
        "task": "graph_listwise_dilution", "model": p50.MODEL, "threshold": 0.50,
        "origin_window": {"name": "first10", "max_rank": 10}, "window": "25.6-26.6", "split": "test",
        "router": "LambdaMART {context+origin_ol+g_net}", "target": TARGET,
        "k_values": K_VALUES, "shortlists": SHORTLISTS, "select_r": SELECT_R,
        "n_tasks_planned": len(tasks), "n_records_evaluated": len(records), "n_new_api_calls": calls,
        "policies": sorted(scores.keys()), "limit_days": args.limit_days,
        "by_policy": by_policy, "records": records, "elapsed_sec": time.time() - started,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    TABLE_OUT.write_text(p50.markdown_table({"by_policy": by_policy}), encoding="utf-8")
    log(p50.markdown_table({"by_policy": by_policy}))
    log(f"done {time.time()-started:.1f}s  new_calls={calls}  out={OUT}")


if __name__ == "__main__":
    main()
