"""Phase86: local LLM (Llama3.1-8B, Qwen2.5-7B) pointwise rows for thr=0.50 main table.
Reuses p18 item_payload/prompt_for_batch/call_ollama/parse_scores. Cache per model (resumable).
Metrics: NDCG@3/Hit@1/Mass@3/JS symbol-balanced over test events.
"""
from __future__ import annotations
import collections, json, math, pathlib, time
import numpy as np
import phase5_sentiment_reconstruction as p5
import phase7_origin_alert as p7
import phase33_origin_alert_ablation as p33
import phase18_origin_alert_llm_baselines as p18
import phase65_pit_lightweight_2025_2026 as p65

THR = 0.50; MEK = 8; TARGET = "log_future_reach"
TR_S, TE_S, TE_E = "2024-06-01", "2025-06-01", "2026-06-01"
MODELS = [("Gemma3-12B", "gemma3:12b"), ("Qwen2.5-7B", "qwen2.5:7b-instruct"), ("Llama3.1-8B", "llama3.1:8b")]
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

def load_cache(path):
    c = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                if o.get("reach_score") is not None: c[o["key"]] = o
    return c

def main():
    t0 = time.time()
    rows_by, emb_by, all_rows = {}, {}, []
    for sym in p5.SYMS:
        r, e = p5.load_symbol(sym); rows_by[sym] = r; emb_by[sym] = e; all_rows.extend(r)
    p18.BATCH_SIZE = 8
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
    log("test rows=%d unique items=%d" % (len(test_rows), len(uniq)))

    out_rows = []
    for label, model in MODELS:
        cache_path = pathlib.Path(__file__).with_name("phase86_%s_cache.jsonl" % model.replace(":", "_").replace(".", "_"))
        cache = load_cache(cache_path)
        missing = [(k, v) for k, v in uniq.items() if k not in cache]
        log("[%s] cached=%d missing=%d" % (label, len(uniq) - len(missing), len(missing)))
        with open(cache_path, "a") as fh:
            for s in range(0, len(missing), p18.BATCH_SIZE):
                batch = missing[s:s + p18.BATCH_SIZE]
                try:
                    resp = p18.call_ollama(model, p18.prompt_for_batch(batch))
                    scores = p18.parse_scores(resp, len(batch))
                except Exception as ex:
                    log("  batch %d err: %s" % (s, str(ex)[:120])); scores = {}
                for idx, (k, _v) in enumerate(batch):
                    sc = scores.get(idx, {})
                    rec = {"key": k, "adoption_score": sc.get("adoption_score"), "reach_score": sc.get("reach_score")}
                    cache[k] = rec; fh.write(json.dumps(rec) + "\n")
                fh.flush()
                if (s // p18.BATCH_SIZE) % 10 == 0:
                    log("  [%s] %d/%d" % (label, min(s + p18.BATCH_SIZE, len(missing)), len(missing)))
        field = "reach_score" if TARGET == "log_future_reach" else "adoption_score"
        pred = np.full(len(panel), np.nan)
        for i, r in enumerate(panel):
            if r["split"] == "test" and r["_key"] in cache:
                v = cache[r["_key"]].get(field)
                if v is not None: pred[i] = float(v)
        evs = p65.event_rows_for_split(panel, pred, TARGET, "test")
        m = {k: p7.symbal_mean(evs, k) for k in p7.METRICS}
        rec = {"family": "Local LLM", "method": label, "model": model, "events": len(evs),
               "symbols": len(set(e["sym"] for e in evs)), **m, "n_scored": int(np.isfinite(pred).sum())}
        out_rows.append(rec)
        log("=== %s | events=%d NDCG=%.4f Hit=%.4f Mass=%.4f JS=%.4f ===" % (label, rec["events"], m["ndcg3"], m["hit1"], m["mass3"], m["js"]))
    pathlib.Path(__file__).with_name("phase86_localllm_thr050_result.json").write_text(json.dumps({"rows": out_rows, "elapsed_sec": time.time() - t0}, indent=2))
    log("done %.1fs" % (time.time() - t0))

if __name__ == "__main__":
    main()
