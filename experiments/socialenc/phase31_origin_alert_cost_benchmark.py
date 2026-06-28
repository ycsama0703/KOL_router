"""Phase31: online cost benchmark for Experiment 3 origin-alert methods.

This benchmark does not re-run the main performance experiment. It measures
input length and online scoring latency on a fixed sample from the main
threshold-0.55, first10 origin-alert validation panel.

Reported latency is intended as operational evidence for large-scale agent
triage, not as a universal hardware-independent complexity theorem.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import time
import urllib.error
import urllib.request

import numpy as np

import phase5_sentiment_reconstruction as p5
import phase7_origin_alert as p7
import phase18_origin_alert_llm_baselines as p18
import phase21_streaming_agent_encoder_baselines as p21
import phase29_origin_alert_text_surface_diagnostic as p29


OUT = pathlib.Path(__file__).with_name("phase31_origin_alert_cost_benchmark_result.json")

THRESHOLD = 0.55
ORIGIN_WINDOW = {"name": "first10", "max_rank": 10}
ENCODER_SLUGS = ["bert_base", "finbert_encoder", "e5_base", "bge_base"]
LLM_SLUGS = ["qwen2.5_7b_origin", "llama3.1_8b_origin"]
LLM_TOKENIZER_MODELS = {
    "qwen2.5_7b_origin": "Qwen/Qwen2.5-7B-Instruct",
    # Tokenizer-compatible public mirror for the local Ollama llama3.1:8b model.
    "llama3.1_8b_origin": "NousResearch/Meta-Llama-3.1-8B-Instruct",
}

NO_OL_FEATURES = p7.FEATURE_SETS["no_ol_strong"]
OL_ORIGIN_FEATURES = p7.FEATURE_SETS["ol_origin"]


def log(message: str) -> None:
    print(message, flush=True)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def summarize(values: list[float]) -> dict:
    vals = [float(v) for v in values if np.isfinite(v)]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": float(statistics.mean(vals)),
        "median": float(statistics.median(vals)),
        "p90": percentile(vals, 90),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def load_main_panel() -> list[dict]:
    rows_by = {}
    emb_by = {}
    all_rows = []
    for sym in p5.SYMS:
        rows, emb = p5.load_symbol(sym)
        rows_by[sym] = rows
        emb_by[sym] = emb
        all_rows.extend(rows)
    metadata = p5.compute_metadata(all_rows)
    ol = p5.compute_oltrait(all_rows, metadata)
    hist = p7.compute_origin_history(rows_by, emb_by, THRESHOLD)
    rows, _events = p7.build_origin_panel(rows_by, emb_by, metadata, ol, hist, THRESHOLD, ORIGIN_WINDOW)
    return rows


def eval_candidates(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        if row["split"] == "val":
            groups.setdefault(row["event_id"], []).append(row)
    selected = []
    for values in groups.values():
        if len(values) < 2:
            continue
        y = np.asarray([row["log_future_reach"] for row in values], dtype=float)
        if np.isfinite(y).all() and np.nanmax(y) > 0:
            selected.extend(values)
    return selected


def deterministic_sample(rows: list[dict], n: int) -> list[dict]:
    if len(rows) <= n:
        return list(rows)
    # Stable, spread-out sample over the chronological validation panel.
    idx = np.linspace(0, len(rows) - 1, n, dtype=int)
    return [rows[int(i)] for i in idx]


def linear_score_latency(rows: list[dict], features: list[str], repeats: int) -> dict:
    beta = np.ones(len(features), dtype=float)
    single_ms = []
    for row in rows:
        t0 = time.perf_counter()
        for _ in range(repeats):
            x = np.asarray([row.get(feature, 0.0) for feature in features], dtype=float)
            _ = float(x @ beta)
        single_ms.append((time.perf_counter() - t0) * 1000.0 / repeats)
    t0 = time.perf_counter()
    for _ in range(repeats):
        X = np.asarray([[row.get(feature, 0.0) for feature in features] for row in rows], dtype=float)
        _ = X @ beta
    batch_ms = (time.perf_counter() - t0) * 1000.0 / repeats / max(1, len(rows))
    return {
        "single_ms_per_query": summarize(single_ms),
        "batch_ms_per_query": batch_ms,
        "feature_dim": len(features),
        "input_tokens": summarize([0.0 for _ in rows]),
    }


def symbol_onehot_latency(rows: list[dict], repeats: int) -> dict:
    syms = list(p5.SYMS)
    sym_index = {sym: i for i, sym in enumerate(syms)}
    beta = np.ones(len(syms), dtype=float)
    single_ms = []
    for row in rows:
        t0 = time.perf_counter()
        for _ in range(repeats):
            x = np.zeros(len(syms), dtype=float)
            x[sym_index[row["sym"]]] = 1.0
            _ = float(x @ beta)
        single_ms.append((time.perf_counter() - t0) * 1000.0 / repeats)
    t0 = time.perf_counter()
    for _ in range(repeats):
        X = np.zeros((len(rows), len(syms)), dtype=float)
        for i, row in enumerate(rows):
            X[i, sym_index[row["sym"]]] = 1.0
        _ = X @ beta
    batch_ms = (time.perf_counter() - t0) * 1000.0 / repeats / max(1, len(rows))
    return {
        "single_ms_per_query": summarize(single_ms),
        "batch_ms_per_query": batch_ms,
        "feature_dim": len(syms),
        "input_tokens": summarize([0.0 for _ in rows]),
    }


def surface_latency(rows: list[dict], repeats: int) -> dict:
    beta = np.ones(20, dtype=float)
    word_counts = []
    single_ms = []
    for row in rows:
        text = row.get("origin_text", "")
        word_counts.append(len((text or "").split()))
        t0 = time.perf_counter()
        for _ in range(repeats):
            x = np.asarray(p29.text_surface_features(text, row["sym"]), dtype=float)
            _ = float(x @ beta)
        single_ms.append((time.perf_counter() - t0) * 1000.0 / repeats)
    t0 = time.perf_counter()
    for _ in range(repeats):
        X = np.asarray([p29.text_surface_features(row.get("origin_text", ""), row["sym"]) for row in rows], dtype=float)
        _ = X @ beta
    batch_ms = (time.perf_counter() - t0) * 1000.0 / repeats / max(1, len(rows))
    return {
        "single_ms_per_query": summarize(single_ms),
        "batch_ms_per_query": batch_ms,
        "feature_dim": 20,
        "input_words": summarize(word_counts),
    }


def symbol_plus_surface_latency(rows: list[dict], repeats: int) -> dict:
    syms = list(p5.SYMS)
    sym_index = {sym: i for i, sym in enumerate(syms)}
    dim = len(syms) + 20
    beta = np.ones(dim, dtype=float)
    word_counts = []
    single_ms = []
    for row in rows:
        text = row.get("origin_text", "")
        word_counts.append(len((text or "").split()))
        t0 = time.perf_counter()
        for _ in range(repeats):
            x = np.zeros(dim, dtype=float)
            x[sym_index[row["sym"]]] = 1.0
            x[len(syms):] = np.asarray(p29.text_surface_features(text, row["sym"]), dtype=float)
            _ = float(x @ beta)
        single_ms.append((time.perf_counter() - t0) * 1000.0 / repeats)
    t0 = time.perf_counter()
    for _ in range(repeats):
        X = np.zeros((len(rows), dim), dtype=float)
        for i, row in enumerate(rows):
            X[i, sym_index[row["sym"]]] = 1.0
            X[i, len(syms):] = np.asarray(p29.text_surface_features(row.get("origin_text", ""), row["sym"]), dtype=float)
        _ = X @ beta
    batch_ms = (time.perf_counter() - t0) * 1000.0 / repeats / max(1, len(rows))
    return {
        "single_ms_per_query": summarize(single_ms),
        "batch_ms_per_query": batch_ms,
        "feature_dim": dim,
        "input_words": summarize(word_counts),
    }


def encoder_latency(rows: list[dict], slug: str, single_n: int, batch_size: int) -> dict:
    import torch
    from transformers import AutoModel, AutoTokenizer

    config = p21.MODEL_CONFIGS[slug]
    texts = [config["prefix"] + p21.clean_text(row.get("origin_text", "")) for row in rows]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    model = AutoModel.from_pretrained(config["model"]).eval().to(device)
    if device == "cuda":
        model.half()

    token_counts = []
    for text in texts:
        encoded = tokenizer(text, truncation=True, max_length=128, add_special_tokens=True)
        token_counts.append(len(encoded["input_ids"]))

    def forward(batch_texts: list[str]) -> None:
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state.float()
            if config["pooling"] == "cls":
                pooled = hidden[:, 0]
            else:
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            _ = torch.nn.functional.normalize(pooled, p=2, dim=1)
        if device == "cuda":
            torch.cuda.synchronize()

    # Warmup.
    forward(texts[: min(batch_size, len(texts))])

    single_ms = []
    for text in texts[: min(single_n, len(texts))]:
        t0 = time.perf_counter()
        forward([text])
        single_ms.append((time.perf_counter() - t0) * 1000.0)

    batch_times = []
    n_scored = 0
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        t0 = time.perf_counter()
        forward(batch)
        elapsed = (time.perf_counter() - t0) * 1000.0
        batch_times.append(elapsed / max(1, len(batch)))
        n_scored += len(batch)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": config["model"],
        "device": device,
        "feature_dim": 768,
        "input_tokens": summarize(token_counts),
        "single_ms_per_query": summarize(single_ms),
        "batch_ms_per_query": float(np.mean(batch_times)) if batch_times else None,
        "batch_size": batch_size,
        "n_batch_scored": n_scored,
    }


def ollama_token_count(model: str, prompt: str) -> int | None:
    body = {"model": model, "prompt": prompt}
    request = urllib.request.Request(
        p18.OLLAMA_URL.replace("/api/generate", "/api/tokenize"),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            obj = json.loads(response.read().decode("utf-8"))
        tokens = obj.get("tokens")
        if isinstance(tokens, list):
            return len(tokens)
        count = obj.get("count")
        if isinstance(count, int):
            return count
    except Exception:
        return None
    return None


def load_hf_tokenizer(slug: str):
    model = LLM_TOKENIZER_MODELS.get(slug)
    if not model:
        return None, None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model), model
    except Exception:
        return None, model


def llm_latency(rows: list[dict], slug: str, model: str, single_n: int, batch_n: int, batch_size: int) -> dict:
    items = []
    for row in rows[: max(single_n, batch_n)]:
        payload = p18.item_payload(row)
        items.append((p18.item_key(payload), payload))

    single_prompts = [p18.prompt_for_batch([item]) for item in items[:single_n]]
    batch_prompts = [
        p18.prompt_for_batch(items[start:start + batch_size])
        for start in range(0, min(batch_n, len(items)), batch_size)
    ]

    token_counts = []
    token_source = "ollama_api"
    hf_tokenizer, hf_tokenizer_model = load_hf_tokenizer(slug)
    for prompt in single_prompts:
        count = ollama_token_count(model, prompt)
        if count is None and hf_tokenizer is not None:
            token_source = "hf_tokenizer"
            count = len(hf_tokenizer(prompt, add_special_tokens=True)["input_ids"])
        elif count is None:
            token_source = "whitespace_fallback"
            count = len(prompt.split())
        token_counts.append(count)

    single_ms = []
    parse_success = 0
    for prompt in single_prompts:
        t0 = time.perf_counter()
        response = p18.call_ollama(model, prompt)
        elapsed = (time.perf_counter() - t0) * 1000.0
        parsed = p18.parse_scores(response, 1)
        parse_success += int(0 in parsed)
        single_ms.append(elapsed)

    batch_ms_per_query = []
    batch_parse_success = 0
    batch_items = 0
    for prompt in batch_prompts:
        n_items = prompt.count("\"text\"")
        t0 = time.perf_counter()
        response = p18.call_ollama(model, prompt)
        elapsed = (time.perf_counter() - t0) * 1000.0
        parsed = p18.parse_scores(response, n_items)
        batch_parse_success += len(parsed)
        batch_items += n_items
        if n_items:
            batch_ms_per_query.append(elapsed / n_items)

    return {
        "model": model,
        "input_tokens": summarize(token_counts),
        "token_source": token_source,
        "hf_tokenizer_model": hf_tokenizer_model,
        "single_ms_per_query": summarize(single_ms),
        "single_parse_success": f"{parse_success}/{len(single_prompts)}",
        "batch_ms_per_query": float(np.mean(batch_ms_per_query)) if batch_ms_per_query else None,
        "batch_size": batch_size,
        "batch_parse_success": f"{batch_parse_success}/{batch_items}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--encoder-single-n", type=int, default=32)
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument("--llm-single-n", type=int, default=5)
    parser.add_argument("--llm-batch-n", type=int, default=20)
    parser.add_argument("--llm-batch-size", type=int, default=10)
    parser.add_argument("--linear-repeats", type=int, default=200)
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    started = time.time()
    log("[1/5] Loading main panel")
    rows = load_main_panel()
    candidates = eval_candidates(rows)
    sample = deterministic_sample(candidates, args.sample_size)
    log(f"  panel_rows={len(rows)} eval_candidate_rows={len(candidates)} sample={len(sample)}")

    result = {
        "task": "origin_alert_cost_benchmark",
        "threshold": THRESHOLD,
        "origin_window": ORIGIN_WINDOW,
        "sample": {
            "requested_sample_size": args.sample_size,
            "actual_sample_size": len(sample),
            "eval_candidate_rows": len(candidates),
        },
        "methods": {},
    }

    log("[2/5] Scalar and surface methods")
    for method, features in p7.FEATURE_SETS.items():
        result["methods"][method] = linear_score_latency(sample, features, args.linear_repeats)
    result["methods"]["symbol_onehot"] = symbol_onehot_latency(sample, args.linear_repeats)
    result["methods"]["text_surface"] = surface_latency(sample, args.linear_repeats)
    result["methods"]["symbol_plus_surface"] = symbol_plus_surface_latency(sample, args.linear_repeats)

    log("[3/5] Encoder methods")
    for slug in ENCODER_SLUGS:
        log(f"  {slug}")
        result["methods"][f"{slug}_origin_text"] = encoder_latency(
            sample,
            slug,
            args.encoder_single_n,
            args.encoder_batch_size,
        )

    if args.skip_llm:
        log("[4/5] Skipping LLM methods")
    else:
        log("[4/5] LLM methods")
        for slug in LLM_SLUGS:
            log(f"  {slug}")
            result["methods"][slug] = llm_latency(
                sample,
                slug,
                p18.MODELS[slug],
                args.llm_single_n,
                args.llm_batch_n,
                args.llm_batch_size,
            )

    result["elapsed_sec"] = time.time() - started
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"[5/5] wrote {OUT}")
    log(f"elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
