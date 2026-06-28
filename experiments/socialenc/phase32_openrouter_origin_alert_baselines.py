"""Phase32: OpenRouter commercial-model baselines for Experiment 3.

This pipeline evaluates commercial API LLMs on the same pre-popularity
origin-alert task as Phase18:

  - each item is one newly originated financial narrative frame;
  - the model sees anonymized origin text plus origin-time non-OL context;
  - the model outputs adoption_score and reach_score;
  - no OLtrait, KOL identity, symbol, date, follower confirmation, or future
    text is exposed.

The script is model-id driven. Swapping commercial baselines should only require
changing --models.
"""
from __future__ import annotations

import argparse
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
import phase18_origin_alert_llm_baselines as p18


OUT = pathlib.Path(__file__).with_name("phase32_openrouter_origin_alert_baselines_result.json")
CACHE_DIR = pathlib.Path(__file__).with_name("phase32_openrouter_cache")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_FILE = pathlib.Path.home() / ".config/alphagap/openrouter_api_key"
PROGRESS = pathlib.Path(__file__).with_name("phase32_openrouter_origin_alert_baselines_progress.json")

DEFAULT_MODELS = [
    "openai/gpt-4.1-mini",
    "anthropic/claude-sonnet-4.5",
    "google/gemini-2.5-flash",
    "deepseek/deepseek-v3.2",
]
DEFAULT_THRESHOLDS = [0.55]
ORIGIN_WINDOW = {"name": "first10", "max_rank": 10}


def log(message: str) -> None:
    print(message, flush=True)


def write_progress(path: pathlib.Path, state: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def model_slug(model_id: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "__", model_id.strip())
    return text.strip("_").lower()


def get_api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    if KEY_FILE.exists():
        return KEY_FILE.read_text(encoding="utf-8").strip()
    raise RuntimeError("OPENROUTER_API_KEY is not set and key file is missing")


def cache_path(slug: str) -> pathlib.Path:
    return CACHE_DIR / f"{slug}.jsonl"


def load_cache(slug: str) -> dict[str, dict]:
    cache = {}
    path = cache_path(slug)
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                adoption = float(record["adoption_score"])
                reach = float(record["reach_score"])
                if np.isfinite(adoption) and np.isfinite(reach):
                    cache[str(record["key"])] = record
            except Exception:
                continue
    return cache


def append_cache(slug: str, records: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with cache_path(slug).open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prompt_for_batch(items: list[tuple[str, dict]]) -> str:
    payload = [{"id": index, **item} for index, (_key, item) in enumerate(items)]
    return "\n".join([
        "You are scoring newly originated financial narrative frames for a research agent.",
        "At scoring time, exactly one KOL has expressed each frame. No follower confirmation is visible.",
        "Score every item independently; items are unrelated and their order carries no information.",
        "adoption_score: 0-100 likelihood and extent that later distinct KOLs repeat the same narrative.",
        "reach_score: 0-100 expected follower-weighted reach of those later KOLs.",
        "Use only the supplied anonymized origin text and origin-time context.",
        "Do not infer a hidden symbol, date, future market outcome, KOL identity, or popularity.",
        "Return only valid JSON with this exact schema:",
        "{\"scores\":[{\"id\":0,\"adoption_score\":0,\"reach_score\":0}]}",
        "Include every id exactly once. Scores must be finite numbers from 0 to 100.",
        "ITEMS:",
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    ])


def request_openrouter(
    api_key: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    use_response_format: bool = True,
    reasoning_effort: str | None = None,
    reasoning_exclude: bool = False,
) -> tuple[str, dict]:
    body = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict JSON scoring function. Output JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if use_response_format:
        body["response_format"] = {"type": "json_object"}
    if reasoning_effort or reasoning_exclude:
        body["reasoning"] = {}
        if reasoning_effort:
            body["reasoning"]["effort"] = reasoning_effort
        if reasoning_exclude:
            body["reasoning"]["exclude"] = True
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/alphagap/socialenc",
            "X-Title": "AlphaGap Origin Alert Baseline",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        obj = json.loads(response.read().decode("utf-8"))
    choices = obj.get("choices") or []
    content = ""
    if choices:
        message = choices[0].get("message") or {}
        content = str(message.get("content") or "")
    return content, obj


def call_openrouter(
    api_key: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    reasoning_effort: str | None,
    reasoning_exclude: bool,
) -> tuple[str, dict, float, str]:
    last_error = ""
    for attempt in range(3):
        started = time.perf_counter()
        try:
            content, obj = request_openrouter(
                api_key,
                model_id,
                prompt,
                max_tokens=max_tokens,
                use_response_format=True,
                reasoning_effort=reasoning_effort,
                reasoning_exclude=reasoning_exclude,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            return content, obj, latency_ms, ""
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {error.code}: {body[:500]}"
            # Some providers reject response_format. Fall back once per attempt.
            if error.code in {400, 422}:
                try:
                    content, obj = request_openrouter(
                        api_key,
                        model_id,
                        prompt,
                        max_tokens=max_tokens,
                        use_response_format=False,
                        reasoning_effort=reasoning_effort,
                        reasoning_exclude=reasoning_exclude,
                    )
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    return content, obj, latency_ms, ""
                except Exception as fallback_error:
                    last_error = f"{last_error}; fallback={type(fallback_error).__name__}: {str(fallback_error)[:300]}"
        except Exception as error:
            last_error = f"{type(error).__name__}: {str(error)[:500]}"
        if attempt < 2:
            time.sleep(2 + attempt * 3)
    return "", {}, float("nan"), last_error


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


def attach_keys(panels: dict[str, list[dict]]) -> dict[str, dict]:
    unique_items = {}
    for rows in panels.values():
        for row in rows:
            payload = p18.item_payload(row)
            key = p18.item_key(payload)
            row["openrouter_item_key"] = key
            if row["split"] == "val":
                unique_items[key] = payload
    return unique_items


def run_model(
    api_key: str,
    model_id: str,
    unique_items: dict[str, dict],
    batch_size: int,
    limit_items: int | None,
    max_tokens_base: int,
    max_tokens_per_item: int,
    reasoning_effort: str | None,
    reasoning_exclude: bool,
    progress_state: dict | None = None,
    progress_path: pathlib.Path | None = None,
) -> tuple[dict[str, dict], dict]:
    slug = model_slug(model_id)
    cache = load_cache(slug)
    missing = [(key, unique_items[key]) for key in sorted(unique_items) if key not in cache]
    if limit_items is not None:
        missing = missing[:limit_items]
    total_batches = (len(missing) + batch_size - 1) // batch_size
    log(f"[{slug}] model={model_id} cached={len(cache)} missing_run={len(missing)} batches={total_batches}")

    stats = {
        "model": model_id,
        "slug": slug,
        "cached_before": len(cache),
        "missing_run": len(missing),
        "batch_size": batch_size,
        "calls": 0,
        "success_items": 0,
        "failed_items": 0,
        "parse_fail_batches": 0,
        "api_fail_batches": 0,
        "prompt_tokens": [],
        "completion_tokens": [],
        "total_tokens": [],
        "latency_ms": [],
        "errors": collections.Counter(),
    }
    started = time.time()
    if progress_state is not None:
        progress_state["current_model"] = slug
        progress_state["models"].setdefault(slug, {})
        progress_state["models"][slug].update({
            "model": model_id,
            "cached_before": len(cache),
            "planned_items": len(missing),
            "planned_batches": total_batches,
            "attempted_items": 0,
            "success_items": 0,
            "failed_items": 0,
            "completed_batches": 0,
            "status": "running",
        })
        if progress_path is not None:
            write_progress(progress_path, progress_state)
    api_key_hash = hashlib.sha1(api_key.encode("utf-8")).hexdigest()[:8]
    for start in range(0, len(missing), batch_size):
        batch = missing[start:start + batch_size]
        prompt = prompt_for_batch(batch)
        max_tokens = max(256, max_tokens_per_item * len(batch) + max_tokens_base)
        content, obj, latency_ms, error = call_openrouter(
            api_key,
            model_id,
            prompt,
            max_tokens,
            reasoning_effort,
            reasoning_exclude,
        )
        stats["calls"] += 1
        if np.isfinite(latency_ms):
            stats["latency_ms"].append(float(latency_ms))
        usage = obj.get("usage") if isinstance(obj, dict) else {}
        if isinstance(usage, dict):
            for source, dest in [
                ("prompt_tokens", "prompt_tokens"),
                ("completion_tokens", "completion_tokens"),
                ("total_tokens", "total_tokens"),
            ]:
                value = usage.get(source)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    stats[dest].append(float(value))
        if error:
            stats["api_fail_batches"] += 1
            stats["errors"][error[:160]] += 1
            parsed = {}
        else:
            parsed = parse_scores(content, len(batch))
            if len(parsed) < len(batch):
                stats["parse_fail_batches"] += 1
        records = []
        for local_index, (key, _payload) in enumerate(batch):
            if local_index not in parsed:
                stats["failed_items"] += 1
                continue
            scores = parsed[local_index]
            record = {
                "key": key,
                "model": model_id,
                "model_slug": slug,
                "api_key_hash": api_key_hash,
                "adoption_score": scores["adoption_score"],
                "reach_score": scores["reach_score"],
                "latency_ms_batch": latency_ms,
                "batch_size": len(batch),
                "prompt_tokens_batch": usage.get("prompt_tokens") if isinstance(usage, dict) else None,
                "completion_tokens_batch": usage.get("completion_tokens") if isinstance(usage, dict) else None,
                "total_tokens_batch": usage.get("total_tokens") if isinstance(usage, dict) else None,
            }
            cache[key] = record
            records.append(record)
            stats["success_items"] += 1
        append_cache(slug, records)
        batch_number = start // batch_size + 1
        if batch_number % 10 == 0 or batch_number == total_batches:
            elapsed = time.time() - started
            rate = (start + len(batch)) / max(elapsed, 1e-9)
            eta = (len(missing) - start - len(batch)) / max(rate, 1e-9)
            if progress_state is not None:
                attempted = min(start + len(batch), len(missing))
                model_state = progress_state["models"][slug]
                model_state.update({
                    "attempted_items": attempted,
                    "success_items": stats["success_items"],
                    "failed_items": stats["failed_items"],
                    "completed_batches": batch_number,
                    "elapsed_sec": elapsed,
                    "eta_sec": eta,
                    "progress": attempted / max(1, len(missing)),
                })
                done_all = 0
                planned_all = 0
                for ms in progress_state["models"].values():
                    done_all += int(ms.get("attempted_items", 0))
                    planned_all += int(ms.get("planned_items", 0))
                # Include not-yet-started planned models.
                planned_all = max(planned_all, int(progress_state.get("total_planned_items", 0)))
                total_elapsed = time.time() - float(progress_state["started_at"])
                overall_rate = done_all / max(total_elapsed, 1e-9)
                overall_eta = (planned_all - done_all) / max(overall_rate, 1e-9)
                progress_state.update({
                    "attempted_items": done_all,
                    "total_elapsed_sec": total_elapsed,
                    "overall_progress": done_all / max(1, planned_all),
                    "overall_eta_sec": overall_eta,
                    "updated_at": time.time(),
                })
                if progress_path is not None:
                    write_progress(progress_path, progress_state)
            log(
                f"  {slug} {batch_number}/{total_batches} "
                f"items={min(start + len(batch), len(missing))}/{len(missing)} "
                f"success={stats['success_items']} failed={stats['failed_items']} "
                f"eta={eta / 60:.1f}m"
            )
    stats["cached_after"] = len(cache)
    if progress_state is not None:
        progress_state["models"][slug].update({
            "attempted_items": len(missing),
            "success_items": stats["success_items"],
            "failed_items": stats["failed_items"],
            "completed_batches": total_batches,
            "status": "done",
            "progress": 1.0,
            "eta_sec": 0.0,
        })
        if progress_path is not None:
            write_progress(progress_path, progress_state)
    stats["errors"] = dict(stats["errors"])
    return cache, stats


def score_vector(rows: list[dict], cache: dict[str, dict], target: str) -> np.ndarray:
    field = "adoption_score" if target == "log_future_adopt" else "reach_score"
    values = np.full(len(rows), np.nan, dtype=float)
    for index, row in enumerate(rows):
        if row["split"] == "val" and row["openrouter_item_key"] in cache:
            values[index] = cache[row["openrouter_item_key"]][field]
    return values


def evaluate_target(rows: list[dict], target: str, model_caches: dict[str, dict], model_slugs: list[str]) -> dict:
    scores = p7.train_scores(rows, target)
    for slug, cache in model_caches.items():
        scores[slug] = score_vector(rows, cache, target)
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
            "symbol_balanced": {metric: p7.symbal_mean(events, metric) for metric in p7.METRICS},
        }
    comparisons = {}
    for slug in model_slugs:
        if "ol_origin" not in event_metrics or slug not in event_metrics:
            continue
        pairs = p7.aligned_pairs(event_metrics["ol_origin"], event_metrics[slug])
        comparisons[f"ol_origin_vs_{slug}"] = {
            metric: {
                "pooled_bootstrap": p7.bootstrap_pooled(pairs, metric),
                "symbol_balanced_bootstrap": p7.bootstrap_symbal(pairs, metric),
            }
            for metric in p7.METRICS
        }
    return {"means": means, "comparisons": comparisons}


def summarize(values: list[float]) -> dict:
    vals = [float(v) for v in values if np.isfinite(v)]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    arr = np.asarray(vals, dtype=float)
    return {
        "n": len(vals),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def aggregate_cost(stats_by_slug: dict[str, dict]) -> dict:
    output = {}
    for slug, stats in stats_by_slug.items():
        calls = max(1, int(stats.get("calls", 0)))
        batch_size = max(1, int(stats.get("batch_size", 1)))
        output[slug] = {
            "model": stats.get("model"),
            "calls": stats.get("calls", 0),
            "success_items": stats.get("success_items", 0),
            "failed_items": stats.get("failed_items", 0),
            "parse_fail_batches": stats.get("parse_fail_batches", 0),
            "api_fail_batches": stats.get("api_fail_batches", 0),
            "latency_ms_per_batch": summarize(stats.get("latency_ms", [])),
            "latency_ms_per_query_mean": (
                summarize(stats.get("latency_ms", [])).get("mean") / batch_size
                if summarize(stats.get("latency_ms", [])).get("mean") is not None
                else None
            ),
            "prompt_tokens_per_batch": summarize(stats.get("prompt_tokens", [])),
            "prompt_tokens_per_query_mean": (
                summarize(stats.get("prompt_tokens", [])).get("mean") / batch_size
                if summarize(stats.get("prompt_tokens", [])).get("mean") is not None
                else None
            ),
            "completion_tokens_per_batch": summarize(stats.get("completion_tokens", [])),
            "total_tokens_per_batch": summarize(stats.get("total_tokens", [])),
            "errors": stats.get("errors", {}),
        }
    return output


def parse_models(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated OpenRouter model ids")
    parser.add_argument("--thresholds", default=",".join(str(x) for x in DEFAULT_THRESHOLDS))
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--limit-items", type=int, default=None, help="Optional per-model smoke-test item limit")
    parser.add_argument("--max-tokens-base", type=int, default=80)
    parser.add_argument("--max-tokens-per-item", type=int, default=80)
    parser.add_argument("--reasoning-effort", default=None, help="Optional OpenRouter reasoning.effort value, e.g. none/minimal/low")
    parser.add_argument("--reasoning-exclude", action="store_true", help="Ask OpenRouter to exclude reasoning content from the returned message")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--progress", default=str(PROGRESS))
    args = parser.parse_args()

    api_key = get_api_key()
    models = parse_models(args.models)
    model_slugs = [model_slug(model) for model in models]
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    out_path = pathlib.Path(args.out)

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

    log("[2/6] Building frozen origin-time panels")
    panels = {}
    event_counts = {}
    for threshold in thresholds:
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
    planned_by_slug = {}
    for model_id, slug in zip(models, model_slugs):
        cache = load_cache(slug)
        missing_count = sum(1 for key in unique_items if key not in cache)
        if args.limit_items is not None:
            missing_count = min(missing_count, args.limit_items)
        planned_by_slug[slug] = missing_count
    progress_path = pathlib.Path(args.progress)
    progress_state = {
        "task": "phase32_openrouter_origin_alert_baselines",
        "models": {
            slug: {
                "model": model_id,
                "planned_items": planned_by_slug[slug],
                "attempted_items": 0,
                "success_items": 0,
                "failed_items": 0,
                "status": "pending",
            }
            for model_id, slug in zip(models, model_slugs)
        },
        "current_model": None,
        "total_planned_items": int(sum(planned_by_slug.values())),
        "attempted_items": 0,
        "overall_progress": 0.0,
        "overall_eta_sec": None,
        "started_at": time.time(),
        "updated_at": time.time(),
    }
    write_progress(progress_path, progress_state)
    model_caches = {}
    stats_by_slug = {}
    for model_id, slug in zip(models, model_slugs):
        cache, stats = run_model(
            api_key,
            model_id,
            unique_items,
            args.batch_size,
            args.limit_items,
            args.max_tokens_base,
            args.max_tokens_per_item,
            args.reasoning_effort,
            args.reasoning_exclude,
            progress_state,
            progress_path,
        )
        model_caches[slug] = cache
        stats_by_slug[slug] = stats

    log("[4/6] Evaluating OpenRouter baselines")
    result = {
        "task": "pre_popularity_origin_alert_openrouter_baselines",
        "models": {slug: model for slug, model in zip(model_slugs, models)},
        "thresholds": thresholds,
        "origin_window": ORIGIN_WINDOW,
        "batch_size": args.batch_size,
        "max_tokens_base": args.max_tokens_base,
        "max_tokens_per_item": args.max_tokens_per_item,
        "reasoning_effort": args.reasoning_effort,
        "reasoning_exclude": args.reasoning_exclude,
        "limit_items": args.limit_items,
        "llm_input": "anonymized origin tweet + Strong Non-OL origin-time context; no OL or diffusion evidence",
        "targets": p7.TARGETS,
        "metrics": p7.METRICS,
        "frozen_feature_sets": p7.FEATURE_SETS,
        "n_unique_openrouter_items": len(unique_items),
        "run_stats": stats_by_slug,
        "cost_summary": aggregate_cost(stats_by_slug),
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
            output["targets"][target] = evaluate_target(rows, target, model_caches, model_slugs)
        result["by_setting"][setting] = output

    log("[5/6] Writing result")
    result["cache_entries"] = {slug: len(cache) for slug, cache in model_caches.items()}
    result["elapsed_sec"] = time.time() - started
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"[6/6] wrote {out_path}")
    log(f"elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
