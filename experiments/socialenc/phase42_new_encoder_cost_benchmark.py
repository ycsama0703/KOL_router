"""Phase42: cost benchmark for newly added Experiment 3 text encoders.

This refreshes the main-table Input Len and online latency columns for encoder
rows added after Phase31:
  - Qwen/Qwen3-Embedding-4B
  - intfloat/e5-mistral-7b-instruct

It reuses Phase31's threshold-0.55, first10 validation sample construction, but
does not rerun the quality experiment.
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import time

import numpy as np

import phase21_streaming_agent_encoder_baselines as p21
import phase31_origin_alert_cost_benchmark as p31


OUT = pathlib.Path(__file__).with_name("phase42_new_encoder_cost_benchmark_result.json")

ENCODER_CONFIGS = {
    "qwen3_embedding_4b_st_origin_text": {
        "model": "Qwen/Qwen3-Embedding-4B",
        "prefix": "",
        "batch_size": 4,
        "dtype": "float16",
    },
    "e5_mistral_7b_instruct_origin_text": {
        "model": "intfloat/e5-mistral-7b-instruct",
        "prefix": "",
        "batch_size": 1,
        "dtype": "float16",
    },
}


def benchmark_sentence_transformer(
    rows: list[dict],
    method: str,
    config: dict,
    single_n: int,
) -> dict:
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    model_name = config["model"]
    prefix = config.get("prefix", "")
    batch_size = int(config["batch_size"])
    texts = [prefix + p21.clean_text(row.get("origin_text", "")) for row in rows]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    token_counts = [
        len(tokenizer(text, truncation=True, max_length=128, add_special_tokens=True)["input_ids"])
        for text in texts
    ]

    kwargs = {"model_kwargs": {"dtype": torch.float16}} if device == "cuda" else {}
    model = SentenceTransformer(model_name, **kwargs)
    feature_dim = int(model.get_sentence_embedding_dimension() or 0)

    def encode(batch_texts: list[str]) -> None:
        _ = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if device == "cuda":
            torch.cuda.synchronize()

    # Warmup.
    encode(texts[: min(batch_size, len(texts))])

    single_ms = []
    for text in texts[: min(single_n, len(texts))]:
        t0 = time.perf_counter()
        encode([text])
        single_ms.append((time.perf_counter() - t0) * 1000.0)

    batch_times = []
    n_scored = 0
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        t0 = time.perf_counter()
        encode(batch)
        elapsed = (time.perf_counter() - t0) * 1000.0
        batch_times.append(elapsed / max(1, len(batch)))
        n_scored += len(batch)

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "method": method,
        "device": device,
        "feature_dim": feature_dim,
        "input_tokens": p31.summarize(token_counts),
        "single_ms_per_query": p31.summarize(single_ms),
        "batch_ms_per_query": float(np.mean(batch_times)) if batch_times else None,
        "batch_size": batch_size,
        "n_batch_scored": n_scored,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--single-n", type=int, default=32)
    parser.add_argument("--methods", nargs="*", default=list(ENCODER_CONFIGS))
    args = parser.parse_args()

    started = time.time()
    p31.log("[1/4] Loading Phase31-compatible validation sample")
    rows = p31.load_main_panel()
    candidates = p31.eval_candidates(rows)
    sample = p31.deterministic_sample(candidates, args.sample_size)
    p31.log(f"  panel_rows={len(rows)} eval_candidate_rows={len(candidates)} sample={len(sample)}")

    result = {
        "task": "new_encoder_cost_benchmark",
        "threshold": p31.THRESHOLD,
        "origin_window": p31.ORIGIN_WINDOW,
        "sample": {
            "requested_sample_size": args.sample_size,
            "actual_sample_size": len(sample),
            "eval_candidate_rows": len(candidates),
        },
        "methods": {},
    }
    if OUT.exists():
        previous = json.loads(OUT.read_text(encoding="utf-8"))
        if previous.get("task") == result["task"]:
            result["methods"].update(previous.get("methods", {}))

    p31.log("[2/4] Benchmarking new encoder rows")
    for method in args.methods:
        if method not in ENCODER_CONFIGS:
            raise SystemExit(f"Unknown method: {method}")
        config = ENCODER_CONFIGS[method]
        p31.log(f"  {method}")
        result["methods"][method] = benchmark_sentence_transformer(sample, method, config, args.single_n)
        OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    result["elapsed_sec"] = time.time() - started
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    p31.log(f"[3/4] wrote {OUT}")
    p31.log(f"[4/4] elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
