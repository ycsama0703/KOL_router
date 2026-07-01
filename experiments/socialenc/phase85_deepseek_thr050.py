"""Phase85 (concurrent): DeepSeek V4 Flash pointwise main-table row, thr=0.50 train1/test1 25.6-26.6.
ThreadPoolExecutor for openrouter requests. Resumable cache.
"""
from __future__ import annotations
import collections, json, math, pathlib, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import phase5_sentiment_reconstruction as p5
import phase7_origin_alert as p7
import phase33_origin_alert_ablation as p33
import phase18_origin_alert_llm_baselines as p18
import phase32_openrouter_origin_alert_baselines as p32
import phase65_pit_lightweight_2025_2026 as p65

THR = 0.50; MEK = 8; TARGET = "log_future_reach"
TR_S, TE_S, TE_E = "2024-06-01", "2025-06-01", "2026-06-01"
MODEL_ID = "deepseek/deepseek-v4-flash"
BATCH = 2; WORKERS = 12
KEY_FILE = pathlib.Path.home() / ".config/alphagap/openrouter_api_key"
CACHE = pathlib.Path(__file__).with_name("phase85_deepseek_thr050_cache.jsonl")
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
        hist_rows = p65.rows_before(all_rows, cutoff)
        meta = p5.compute_metadata(hist_rows)
        ol = p65.with_train_end(cutoff, p5.compute_oltrait, hist_rows, meta)
        hist = p65.with_train_end(cutoff, p7.compute_origin_history, rows_by, emb_by, THR)
        raw_ol = p65.with_train_end(cutoff, p33.compute_raw_oltrait, hist_rows, meta)
        states[block] = {"cutoff": cutoff, "meta": meta, "ol": ol, "hist": hist,
                         "raw_ol": raw_ol, "shuffled_ol": p65.make_shuffled_ol(ol, block)}
    return states

def load_cache():
    c = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                if o.get("reach_score") is not None: c[o["key"]] = o
    return c

def main():
    t0 = time.time()
    api_key = KEY_FILE.read_text().strip()
    rows_by, emb_by, all_rows = {}, {}, []
    for sym in p5.SYMS:
        r, e = p5.load_symbol(sym); rows_by[sym] = r; emb_by[sym] = e; all_rows.extend(r)
    p7.MIN_EVENT_KOLS = MEK; p65.TRAIN_START = TR_S; p65.TEST_END = TE_E
    states = states_for(rows_by, emb_by, all_rows)
    p65.ORIGIN_WINDOW = {"name": "first10", "max_rank": 10}
    panel, _ = p65.build_pit_origin_panel(rows_by, emb_by, states)
    for r in panel:
        pl = p18.item_payload(r); r["_key"] = p18.item_key(pl); r["_payload"] = pl
    test_rows = [r for r in panel if r["split"] == "test"]
    uniq = {}
    for r in test_rows:
        if r["_key"] not in uniq: uniq[r["_key"]] = r["_payload"]
    cache = load_cache()
    missing = [(k, v) for k, v in uniq.items() if k not in cache]
    log("test rows=%d unique=%d cached=%d missing=%d (workers=%d)" % (len(test_rows), len(uniq), len(uniq) - len(missing), len(missing), WORKERS))

    batches = [missing[s:s + BATCH] for s in range(0, len(missing), BATCH)]
    lock = threading.Lock()
    fh = open(CACHE, "a")
    lat, ptok = [], []
    done = {"n": 0}

    def work(batch):
        prompt = p32.prompt_for_batch(batch)
        max_tokens = max(512, 220 * len(batch) + 200)
        content, obj, latency_ms, err = p32.call_openrouter(api_key, MODEL_ID, prompt, max_tokens, None, False)
        if err:
            return None
        scores = p32.parse_scores(content, len(batch))
        usage = obj.get("usage") if isinstance(obj, dict) else {}
        pt = float(usage["prompt_tokens"]) / len(batch) if isinstance(usage, dict) and usage.get("prompt_tokens") else None
        recs = []
        for idx, (k, _v) in enumerate(batch):
            sc = scores.get(idx, {})
            recs.append({"key": k, "adoption_score": sc.get("adoption_score"), "reach_score": sc.get("reach_score")})
        return recs, latency_ms / len(batch) if np.isfinite(latency_ms) else None, pt

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(work, b): b for b in batches}
        for fut in as_completed(futs):
            res = fut.result()
            with lock:
                done["n"] += 1
                if res is not None:
                    recs, lq, pt = res
                    for rec in recs:
                        cache[rec["key"]] = rec; fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    if lq is not None: lat.append(lq)
                    if pt is not None: ptok.append(pt)
                if done["n"] % 20 == 0:
                    log("  %d/%d batches  lat=%.0fms/q  elapsed=%.0fs" % (done["n"], len(batches), np.mean(lat) if lat else 0, time.time() - t0))
    fh.close()

    field = "reach_score" if TARGET == "log_future_reach" else "adoption_score"
    pred = np.full(len(panel), np.nan)
    for i, r in enumerate(panel):
        if r["split"] == "test" and r["_key"] in cache:
            v = cache[r["_key"]].get(field)
            if v is not None: pred[i] = float(v)
    testmask = np.array([r["split"] == "test" for r in panel])
    valid = testmask & np.isfinite(pred)
    med = float(np.median(pred[valid])) if valid.sum() else 0.0
    imp = testmask & ~np.isfinite(pred); pred[imp] = med
    log("imputed %d/%d test candidates to neutral=%.3f" % (int(imp.sum()), int(testmask.sum()), med))
    evs = p65.event_rows_for_split(panel, pred, TARGET, "test")
    m = {k: p7.symbal_mean(evs, k) for k in p7.METRICS}
    out = {"task": "phase85_deepseek_thr050", "model": MODEL_ID, "events": len(evs),
           "symbols": len(set(e["sym"] for e in evs)), **m,
           "latency_ms_per_query": float(np.mean(lat)) if lat else None,
           "prompt_tokens_per_query": float(np.mean(ptok)) if ptok else None,
           "n_unique_items": len(uniq), "n_scored": int(np.isfinite(pred).sum()), "elapsed_sec": time.time() - t0}
    pathlib.Path(__file__).with_name("phase85_deepseek_thr050_result.json").write_text(json.dumps(out, indent=2))
    log("=== DeepSeek V4 Flash | events=%d NDCG=%.4f Hit=%.4f Mass=%.4f JS=%.4f | lat=%.1fms/q ptok=%.1f ===" % (
        out["events"], m["ndcg3"], m["hit1"], m["mass3"], m["js"],
        out["latency_ms_per_query"] or float("nan"), out["prompt_tokens_per_query"] or float("nan")))
    log("done %.1fs" % (time.time() - t0))

if __name__ == "__main__":
    main()
