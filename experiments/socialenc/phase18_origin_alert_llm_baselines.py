"""Experiment 3 zero-shot local-LLM origin-time baselines.

The linear OL-Origin method is frozen. Each LLM receives one anonymized origin
tweet plus the contemporaneously available Strong Non-OL context and predicts
two absolute scores: future KOL adoption and follower-weighted reach. It never
sees another tweet from the frame, early popularity, future text, OLtrait, KOL
identity, symbol, or date.

Candidates from unrelated events are batched only for inference throughput.
The prompt requires each item to be scored independently.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import pathlib
import re
import time
import urllib.error
import urllib.request

import numpy as np

import phase5_sentiment_reconstruction as p5
import phase7_origin_alert as p7


OUT = pathlib.Path(__file__).with_name("phase18_origin_alert_llm_baselines_result.json")
CACHE_DIR = pathlib.Path(__file__).with_name("phase18_origin_llm_cache")
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
BATCH_SIZE = int(os.getenv("ORIGIN_LLM_BATCH_SIZE", "40"))
MAIN_THRESHOLDS = [0.55, 0.60, 0.65]
ORIGIN_WINDOW = {"name": "first10", "max_rank": 10}

MODELS = {
    "qwen2.5_7b_origin": "qwen2.5:7b-instruct",
    "gemma3_12b_origin": "gemma3:12b",
    "llama3.1_8b_origin": "llama3.1:8b",
}

ANONYMIZE_PATTERNS = [
    r"\b(?:AAPL|MSFT|NVDA|TSLA|AMZN|META|GOOGL|AMD|MSTR|COIN|HOOD|PLTR|SPY|QQQ|BTC|ETH|SOL)\b",
    r"\b(?:Apple|Microsoft|Nvidia|Tesla|Amazon|Facebook|Meta|Google|Alphabet|MicroStrategy|Coinbase|Robinhood|Palantir|Bitcoin|Ethereum|Solana)\b",
]


def log(message: str) -> None:
    print(message, flush=True)


def anonymize_text(text: str, limit: int = 500) -> str:
    text = re.sub(r"https?://\S+", " <URL> ", text or "")
    text = re.sub(r"@[A-Za-z0-9_]+", "<USER>", text)
    text = re.sub(r"\$[A-Za-z][A-Za-z0-9._-]*", "<ASSET>", text)
    for pattern in ANONYMIZE_PATTERNS:
        text = re.sub(pattern, "<ASSET>", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def rounded(value, digits=3):
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None


def item_payload(row: dict) -> dict:
    return {
        "text": anonymize_text(row.get("origin_text", "")),
        "followers_log1p": rounded(row["origin_logfoll"]),
        "verified": int(row["origin_verified"] > 0),
        "origin_rank_log": rounded(row["log_origin_rank"]),
        "elapsed_hours": rounded(row["elapsed_hours"]),
        "prior_frame_count": int(row["prior_frame_count"]),
        "stance": rounded(row["origin_stance"]),
        "global_novelty": rounded(row["novelty_global"]),
        "event_novelty": rounded(row["novelty_event"]),
        "historical_origin_count_log1p": rounded(row["hist_log_origin_count"]),
        "historical_mean_adoption_log1p": rounded(row["hist_mean_log_adopt"]),
        "historical_success_rate": rounded(row["hist_success_rate"]),
    }


def item_key(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def cache_path(model_slug: str) -> pathlib.Path:
    return CACHE_DIR / f"{model_slug}.jsonl"


def load_cache(model_slug: str) -> dict[str, dict[str, float]]:
    cache = {}
    path = cache_path(model_slug)
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                adoption = float(record["adoption_score"])
                reach = float(record["reach_score"])
                if np.isfinite(adoption) and np.isfinite(reach):
                    cache[str(record["key"])] = {
                        "adoption_score": adoption,
                        "reach_score": reach,
                    }
            except Exception:
                continue
    return cache


def append_cache(model_slug: str, records: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with cache_path(model_slug).open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prompt_for_batch(items: list[tuple[str, dict]]) -> str:
    payload = [
        {"id": index, **item}
        for index, (_key, item) in enumerate(items)
    ]
    return "\n".join([
        "You are scoring newly originated financial narrative frames for a research agent.",
        "At scoring time, exactly one KOL has expressed each frame. No follower confirmation is visible.",
        "Score every item independently; items are unrelated and their order carries no information.",
        "adoption_score: 0-100 likelihood and extent that later distinct KOLs repeat the same narrative.",
        "reach_score: 0-100 expected follower-weighted reach of those later KOLs.",
        "Use only the supplied origin text and origin-time context. Do not assume a symbol, date, or future market outcome.",
        "Return only JSON: {\"scores\":[{\"id\":0,\"adoption_score\":0,\"reach_score\":0},...]}",
        "Include every id exactly once. Scores must be finite numbers from 0 to 100.",
        "ITEMS:",
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    ])


def call_ollama(model: str, prompt: str) -> str:
    request_body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "keep_alive": "15m",
        "options": {
            "temperature": 0,
            "num_predict": 2400,
            "num_ctx": 8192,
        },
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                obj = json.loads(response.read().decode("utf-8"))
                return str(obj.get("response", ""))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt == 2:
                log(f"  call failed after retries: {error}")
                return ""
            time.sleep(3 + 2 * attempt)
    return ""


def parse_scores(response: str, n_items: int) -> dict[int, dict[str, float]]:
    try:
        obj = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            return {}
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    values = obj.get("scores", []) if isinstance(obj, dict) else []
    parsed = {}
    for value in values if isinstance(values, list) else []:
        try:
            if isinstance(value, list) and len(value) >= 3:
                index = int(value[0])
                adoption = float(value[1])
                reach = float(value[2])
            else:
                index = int(value["id"])
                adoption = float(value["adoption_score"])
                reach = float(value["reach_score"])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if 0 <= index < n_items and np.isfinite(adoption) and np.isfinite(reach):
            parsed[index] = {
                "adoption_score": float(np.clip(adoption, 0.0, 100.0)),
                "reach_score": float(np.clip(reach, 0.0, 100.0)),
            }
    return parsed


def score_missing_batch(model: str, items: list[tuple[str, dict]]) -> dict[int, dict[str, float]]:
    pending = list(enumerate(items))
    output = {}
    for attempt in range(3):
        if not pending:
            break
        sub_items = [item for _, item in pending]
        response = call_ollama(model, prompt_for_batch(sub_items))
        parsed = parse_scores(response, len(sub_items))
        next_pending = []
        for local_index, (original_index, _item) in enumerate(pending):
            if local_index in parsed:
                output[original_index] = parsed[local_index]
            else:
                next_pending.append((original_index, _item))
        pending = next_pending
        if pending and attempt < 2:
            log(f"    retrying {len(pending)} omitted/malformed items")
    return output


def run_model(
    model_slug: str,
    model: str,
    unique_items: dict[str, dict],
) -> dict[str, dict[str, float]]:
    cache = load_cache(model_slug)
    missing = [(key, unique_items[key]) for key in sorted(unique_items) if key not in cache]
    log(f"[{model_slug}] cached={len(cache)} missing={len(missing)} batches={(len(missing) + BATCH_SIZE - 1) // BATCH_SIZE}")
    failures = 0
    started = time.time()
    for start in range(0, len(missing), BATCH_SIZE):
        batch = missing[start:start + BATCH_SIZE]
        parsed = score_missing_batch(model, batch)
        records = []
        for index, (key, _payload) in enumerate(batch):
            if index not in parsed:
                failures += 1
                continue
            scores = parsed[index]
            cache[key] = scores
            records.append({"key": key, "model": model, **scores})
        append_cache(model_slug, records)
        batch_number = start // BATCH_SIZE + 1
        total_batches = (len(missing) + BATCH_SIZE - 1) // BATCH_SIZE
        if batch_number % 10 == 0 or batch_number == total_batches:
            elapsed = time.time() - started
            rate = (start + len(batch)) / max(elapsed, 1e-9)
            eta = (len(missing) - start - len(batch)) / max(rate, 1e-9)
            log(
                f"  {model_slug} {batch_number}/{total_batches} "
                f"items={min(start + len(batch), len(missing))}/{len(missing)} "
                f"failures={failures} eta={eta / 60:.1f}m"
            )
    return cache


def attach_keys(panels: dict[str, list[dict]]) -> dict[str, dict]:
    unique_items = {}
    for rows in panels.values():
        for row in rows:
            payload = item_payload(row)
            key = item_key(payload)
            row["llm_item_key"] = key
            if row["split"] == "val":
                unique_items[key] = payload
    return unique_items


def llm_score_vector(rows: list[dict], cache: dict[str, dict], target: str) -> np.ndarray:
    field = "adoption_score" if target == "log_future_adopt" else "reach_score"
    values = np.full(len(rows), np.nan, dtype=float)
    for index, row in enumerate(rows):
        if row["split"] == "val" and row["llm_item_key"] in cache:
            values[index] = cache[row["llm_item_key"]][field]
    return values


def evaluate_target(
    rows: list[dict],
    target: str,
    model_caches: dict[str, dict],
) -> dict:
    scores = p7.train_scores(rows, target)
    for model_slug, cache in model_caches.items():
        scores[model_slug] = llm_score_vector(rows, cache, target)
    event_metrics = {
        method: p7.event_rows(rows, method_scores, target)
        for method, method_scores in scores.items()
    }
    means = {}
    for method, events in event_metrics.items():
        means[method] = {
            "n_events": len(events),
            "n_symbols": len(set(event["sym"] for event in events)),
            "pooled": {metric: p7.pooled_mean(events, metric) for metric in p7.METRICS},
            "symbol_balanced": {
                metric: p7.symbal_mean(events, metric) for metric in p7.METRICS
            },
        }
        if method in p7.FEATURE_SETS:
            means[method]["global_top10"] = p7.evaluate_global(
                rows, scores[method], target,
            )

    comparisons = {}
    for model_slug in MODELS:
        pairs = p7.aligned_pairs(event_metrics["ol_origin"], event_metrics[model_slug])
        comparisons[f"ol_origin_vs_{model_slug}"] = {
            metric: {
                "pooled_bootstrap": p7.bootstrap_pooled(pairs, metric),
                "symbol_balanced_bootstrap": p7.bootstrap_symbal(pairs, metric),
            }
            for metric in p7.METRICS
        }
    return {"means": means, "comparisons": comparisons}


def aggregate_main_region(result: dict) -> dict:
    methods = [*p7.FEATURE_SETS, *MODELS]
    output = {}
    for method in methods:
        output[method] = {}
        for target in p7.TARGETS:
            values = collections.defaultdict(list)
            for threshold in MAIN_THRESHOLDS:
                setting = f"thr{threshold:.2f}_first10"
                method_result = result["by_setting"][setting]["targets"][target]["means"].get(method)
                if not method_result:
                    continue
                for metric, value in method_result["symbol_balanced"].items():
                    values[metric].append(value)
            output[method][target] = {
                metric: float(np.mean(metric_values))
                for metric, metric_values in values.items()
            }
    return output


def main() -> None:
    started = time.time()
    log("[1/6] Loading source data")
    rows_by = {}
    embeddings_by = {}
    all_rows = []
    for index, symbol in enumerate(p5.SYMS, 1):
        rows, embeddings = p5.load_symbol(symbol)
        rows_by[symbol] = rows
        embeddings_by[symbol] = embeddings
        all_rows.extend(rows)
        log(f"  {index:02d}/{len(p5.SYMS)} {symbol:<5} rows={len(rows):>6}")
    metadata = p5.compute_metadata(all_rows)
    ol = p5.compute_oltrait(all_rows, metadata)

    log("[2/6] Building frozen first10 origin-time panels")
    panels = {}
    event_counts = {}
    for threshold in MAIN_THRESHOLDS:
        history = p7.compute_origin_history(rows_by, embeddings_by, threshold)
        setting = f"thr{threshold:.2f}_first10"
        rows, events = p7.build_origin_panel(
            rows_by, embeddings_by, metadata, ol, history,
            threshold, ORIGIN_WINDOW,
        )
        panels[setting] = rows
        event_counts[setting] = len(events)

    unique_items = attach_keys(panels)
    log(f"[3/6] Unique validation origin candidates={len(unique_items)}")
    model_caches = {}
    for model_slug, model in MODELS.items():
        model_caches[model_slug] = run_model(model_slug, model, unique_items)

    log("[4/6] Evaluating frozen OL and LLM baselines")
    result = {
        "task": "pre_popularity_origin_alert_llm_baselines",
        "models": MODELS,
        "thresholds": MAIN_THRESHOLDS,
        "origin_window": ORIGIN_WINDOW,
        "batch_size": BATCH_SIZE,
        "llm_input": "anonymized origin tweet + Strong Non-OL origin-time context; no OL or diffusion evidence",
        "targets": p7.TARGETS,
        "metrics": p7.METRICS,
        "frozen_feature_sets": p7.FEATURE_SETS,
        "n_unique_llm_items": len(unique_items),
        "by_setting": {},
    }
    for setting, rows in panels.items():
        output = {
            "n_rows": len(rows),
            "n_train_rows": sum(row["split"] == "train" for row in rows),
            "n_val_rows": sum(row["split"] == "val" for row in rows),
            "n_events": event_counts[setting],
            "targets": {},
        }
        for target in p7.TARGETS:
            output["targets"][target] = evaluate_target(
                rows, target, model_caches,
            )
        result["by_setting"][setting] = output

    log("[5/6] Aggregating main region")
    result["main_region_mean"] = aggregate_main_region(result)
    result["cache_entries"] = {
        model_slug: len(cache) for model_slug, cache in model_caches.items()
    }
    result["elapsed_sec"] = time.time() - started
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"[6/6] wrote {OUT}")
    log(f"elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
