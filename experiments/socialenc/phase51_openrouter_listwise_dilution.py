"""Phase51: listwise OpenRouter GPT dilution and OL routing diagnostic.

This experiment fixes the downstream LLM and varies how many same-day candidate
narratives it sees. It uses listwise selection rather than independent scalar
scores, because the paper claim is about agent attention under noisy candidate
sets.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import pathlib
import random
import re
import time
import urllib.error
import urllib.request

import numpy as np

import phase5_sentiment_reconstruction as p5
import phase7_origin_alert as p7
import phase18_origin_alert_llm_baselines as p18
import phase21_streaming_agent_encoder_baselines as p21
import phase28_origin_alert_encoder_baselines as p28


OUT = pathlib.Path(__file__).with_name("phase51_openrouter_listwise_dilution_result.json")
TABLE_OUT = pathlib.Path(__file__).with_name("phase51_openrouter_listwise_dilution_table.md")
CACHE_DIR = pathlib.Path(__file__).with_name("phase51_openrouter_listwise_cache")
KEY_FILE = pathlib.Path.home() / ".config/alphagap/openrouter_api_key"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-chat-latest"
SLUG = "openrouter__openai__gpt-chat-latest__listwise_dilution_v1"
THRESHOLD = 0.55
ORIGIN_WINDOW = {"name": "first10", "max_rank": 10}
TARGET = "log_future_reach"
RAW_TARGET = "future_reach"
K_VALUES = [10, 20, 30]
SHORTLISTS = [10, 20]
SELECT_R = 3
RNG_SEED = 20260627


def log(message: str) -> None:
    print(message, flush=True)


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


def cache_path() -> pathlib.Path:
    return CACHE_DIR / f"{SLUG}.jsonl"


def load_cache() -> dict[str, dict]:
    cache = {}
    path = cache_path()
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                if record.get("cache_key"):
                    cache[str(record["cache_key"])] = record
            except Exception:
                continue
    return cache


def append_cache(records: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with cache_path().open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def request_openrouter(api_key: str, prompt: str, max_tokens: int) -> tuple[str, dict]:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a strict JSON financial research routing function. Output JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/alphagap/socialenc",
            "X-Title": "AlphaGap Listwise Dilution Diagnostic",
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


def call_openrouter(api_key: str, prompt: str, max_tokens: int) -> tuple[str, dict, float, str]:
    last_error = ""
    for attempt in range(3):
        started = time.perf_counter()
        try:
            content, obj = request_openrouter(api_key, prompt, max_tokens)
            return content, obj, (time.perf_counter() - started) * 1000.0, ""
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {error.code}: {body[:500]}"
        except Exception as error:
            last_error = f"{type(error).__name__}: {str(error)[:500]}"
        if attempt < 2:
            time.sleep(2 + 3 * attempt)
    return "", {}, float("nan"), last_error


def parse_selection(response: str, candidate_ids: set[int], r: int) -> tuple[list[dict], str]:
    try:
        obj = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            return [], "json_decode"
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            recovered = []
            used = set()
            for m in re.finditer(r'"id"\s*:\s*(\d+).*?"attention"\s*:\s*([0-9.]+)', response, flags=re.DOTALL):
                cid = int(m.group(1))
                if cid in candidate_ids and cid not in used:
                    recovered.append({"id": cid, "attention": float(m.group(2)), "reason": ""})
                    used.add(cid)
                if len(recovered) >= r:
                    break
            if len(recovered) >= r:
                total = sum(max(0.0, item["attention"]) for item in recovered)
                for item in recovered:
                    item["attention_norm"] = max(0.0, item["attention"]) / max(total, 1e-12)
                return recovered, "json_recovered"
            return [], "json_decode"
    selected = obj.get("selected") or obj.get("selections") or []
    if not isinstance(selected, list):
        return [], "missing_selected"
    out = []
    used = set()
    for item in selected:
        try:
            cid = int(item.get("id"))
        except Exception:
            continue
        if cid in candidate_ids and cid not in used:
            attention = item.get("attention", item.get("attention_points", 0))
            try:
                attention = float(attention)
            except Exception:
                attention = 0.0
            reason = str(item.get("reason", ""))[:300]
            out.append({"id": cid, "attention": attention, "reason": reason})
            used.add(cid)
        if len(out) >= r:
            break
    if len(out) < r:
        return out, "too_few_selected"
    total = sum(max(0.0, item["attention"]) for item in out)
    if total > 0:
        for item in out:
            item["attention_norm"] = max(0.0, item["attention"]) / total
    else:
        for item in out:
            item["attention_norm"] = 1.0 / len(out)
    return out, ""


def attention_entropy(selection: list[dict]) -> float:
    probs = np.asarray([max(0.0, item.get("attention_norm", 0.0)) for item in selection], dtype=float)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return float("nan")
    return float(-(probs * np.log(probs)).sum() / max(math.log(len(selection)), 1e-12))


def build_panel() -> list[dict]:
    rows_by = {}
    emb_by = {}
    all_rows = []
    log("[1/5] Loading source data")
    for index, sym in enumerate(p5.SYMS, 1):
        rows, emb = p5.load_symbol(sym)
        rows_by[sym] = rows
        emb_by[sym] = emb
        all_rows.extend(rows)
        log(f"  {index:02d}/{len(p5.SYMS)} {sym:<5} rows={len(rows):>6}")
    meta = p5.compute_metadata(all_rows)
    ol = p5.compute_oltrait(all_rows, meta)
    log("[2/5] Building 0.55/first10 origin panel")
    hist = p7.compute_origin_history(rows_by, emb_by, THRESHOLD)
    rows, events = p7.build_origin_panel(rows_by, emb_by, meta, ol, hist, THRESHOLD, ORIGIN_WINDOW)
    log(f"  panel rows={len(rows)} events={len(events)}")
    return rows


def row_uid(row: dict) -> str:
    payload = p18.item_payload(row)
    return p18.item_key(payload)


def item_payload_for_prompt(row: dict, cid: int) -> dict:
    payload = p18.item_payload(row)
    return {
        "id": cid,
        "text": payload["text"],
        "followers_log1p": payload["followers_log1p"],
        "verified": payload["verified"],
        "origin_rank_log": payload["origin_rank_log"],
        "elapsed_hours": payload["elapsed_hours"],
        "prior_frame_count": payload["prior_frame_count"],
        "stance": payload["stance"],
        "global_novelty": payload["global_novelty"],
        "event_novelty": payload["event_novelty"],
        "historical_origin_count_log1p": payload["historical_origin_count_log1p"],
        "historical_mean_adoption_log1p": payload["historical_mean_adoption_log1p"],
        "historical_success_rate": payload["historical_success_rate"],
    }


def build_prompt(rows: list[dict], candidate_indices: list[int], r: int) -> str:
    items = [item_payload_for_prompt(rows[index], cid) for cid, index in enumerate(candidate_indices)]
    return "\n".join([
        "You are routing newly originated financial social-media narratives for a research agent.",
        "All candidates are available at the same decision time. No future follower confirmation is visible.",
        f"Select exactly {r} narratives that are most worth sending to expensive downstream research.",
        "Allocate exactly 100 total attention points across the selected narratives.",
        "Prefer candidates likely to be repeated by later distinct KOLs and to achieve high follower-weighted reach.",
        "Use only the anonymized text and point-in-time context supplied below.",
        "Do not infer hidden symbols, dates, KOL identities, or future market outcomes.",
        "Return only valid JSON with this exact schema:",
        "{\"selected\":[{\"id\":0,\"attention\":34,\"reason\":\"short reason\"}]}",
        "Include exactly the selected ids; ids must come from the candidate list.",
        "CANDIDATES:",
        json.dumps(items, ensure_ascii=False, separators=(",", ":")),
    ])


def cache_key(day: int, policy: str, candidate_indices: list[int], r: int) -> str:
    payload = {
        "version": "phase51_openrouter_v1",
        "model": MODEL,
        "day": str(day),
        "policy": policy,
        "r": int(r),
        "uids": [row_uid_for_cache[index] for index in candidate_indices],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


row_uid_for_cache: dict[int, str] = {}


def select_top(indices: list[int], scores: dict[int, float], n: int) -> list[int]:
    valid = [index for index in indices if np.isfinite(scores.get(index, np.nan))]
    valid.sort(key=lambda index: (-scores[index], row_uid_for_cache[index]))
    return valid[:n]


def build_tasks(rows: list[dict], limit_days: int | None, include_controls: bool, include_qwen: bool) -> tuple[list[dict], dict[str, np.ndarray]]:
    scores_arr = p7.train_scores(rows, TARGET)
    scores = {
        "ol": {i: float(v) for i, v in enumerate(scores_arr["ol_origin"]) if np.isfinite(v)},
        "follower": {i: float(v) for i, v in enumerate(scores_arr["followers"]) if np.isfinite(v)},
        "no_ol": {i: float(v) for i, v in enumerate(scores_arr["no_ol_strong"]) if np.isfinite(v)},
    }
    if include_qwen:
        log("  loading qwen3_embedding_4b_st origin-text readout")
        model_selection = {}
        emb_cache = p21.load_embedding_cache("qwen3_embedding_4b_st")
        matrix = p28.origin_text_matrix(rows, emb_cache)
        qwen_scores = p28.fit_matrix_score(rows, matrix, TARGET, "qwen3_embedding_4b_st_origin_text", model_selection)
        scores["qwen3_4b"] = {i: float(v) for i, v in enumerate(qwen_scores) if np.isfinite(v)}
    by_day = collections.defaultdict(list)
    for index, row in enumerate(rows):
        if row["split"] == "val" and np.isfinite(float(row[TARGET])):
            by_day[str(row["day"])].append(index)
    eligible_days = [
        day for day, indices in by_day.items()
        if len(indices) >= max(K_VALUES)
    ]
    eligible_days.sort(key=lambda day: (-len(by_day[day]), day))
    if limit_days is not None:
        eligible_days = eligible_days[:limit_days]
    rng = random.Random(RNG_SEED)
    tasks = []
    for day in eligible_days:
        indices = sorted(by_day[day], key=lambda index: (rows[index]["sym"], rows[index]["event_id"], rows[index]["frame_j"], row_uid_for_cache[index]))
        rng.shuffle(indices)
        full_pool = indices[:max(K_VALUES)]
        for k in K_VALUES:
            candidates = full_pool[:k]
            tasks.append({
                "day": day,
                "policy": f"full_k{k}",
                "policy_family": "full_llm",
                "k": k,
                "shortlist": None,
                "candidate_indices": candidates,
            })
        for b in SHORTLISTS:
            random_candidates = full_pool[:]
            rng.shuffle(random_candidates)
            tasks.append({
                "day": day,
                "policy": f"random_b{b}",
                "policy_family": "random_to_llm",
                "k": max(K_VALUES),
                "shortlist": b,
                "candidate_indices": random_candidates[:b],
            })
            for name in ["ol"] + (["follower", "no_ol"] if include_controls else []) + (["qwen3_4b"] if include_qwen else []):
                tasks.append({
                    "day": day,
                    "policy": f"{name}_b{b}",
                    "policy_family": f"{name}_to_llm",
                    "k": max(K_VALUES),
                    "shortlist": b,
                    "candidate_indices": select_top(full_pool, scores[name], b),
                })
    return tasks, scores_arr


def evaluate_task(rows: list[dict], task: dict, selected: list[dict]) -> dict:
    candidates = task["candidate_indices"]
    selected_indices = [candidates[item["id"]] for item in selected if 0 <= item["id"] < len(candidates)]
    r = len(selected_indices)
    y = {index: float(rows[index][TARGET]) for index in candidates}
    raw = {index: float(rows[index][RAW_TARGET]) for index in candidates}
    selected_log = float(sum(y[index] for index in selected_indices))
    selected_raw = float(sum(raw[index] for index in selected_indices))
    oracle = sorted(candidates, key=lambda index: (-y[index], row_uid_for_cache[index]))[:r]
    oracle_raw = sorted(candidates, key=lambda index: (-raw[index], row_uid_for_cache[index]))[:r]
    full_pool = task.get("full_pool_indices") or candidates
    full_oracle = sorted(full_pool, key=lambda index: (-float(rows[index][TARGET]), row_uid_for_cache[index]))[:r]
    return {
        "selected_log_reach": selected_log,
        "selected_raw_reach": selected_raw,
        "capture_within_shown": float(selected_log / max(sum(y[index] for index in oracle), 1e-12)),
        "raw_capture_within_shown": float(selected_raw / max(sum(raw[index] for index in oracle_raw), 1e-12)),
        "selected_vs_full_pool_oracle": float(selected_log / max(sum(float(rows[index][TARGET]) for index in full_oracle), 1e-12)),
        "attention_entropy": attention_entropy(selected),
        "selected_ids": [int(item["id"]) for item in selected],
        "selected_uids": [row_uid_for_cache[index] for index in selected_indices],
    }


def summarize_metrics(records: list[dict]) -> dict:
    out = {}
    for key in ["capture_within_shown", "selected_vs_full_pool_oracle", "attention_entropy", "latency_ms", "prompt_tokens", "completion_tokens", "total_tokens"]:
        vals = [float(record[key]) for record in records if record.get(key) is not None and np.isfinite(float(record[key]))]
        if vals:
            arr = np.asarray(vals, dtype=float)
            out[key] = {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "p10": float(np.percentile(arr, 10)),
                "p90": float(np.percentile(arr, 90)),
            }
    out["n"] = len(records)
    out["parse_failures"] = int(sum(1 for record in records if record.get("parse_error") and record.get("parse_error") != "json_recovered"))
    out["json_recovered"] = int(sum(1 for record in records if record.get("parse_error") == "json_recovered"))
    return out


def markdown_table(summary: dict) -> str:
    lines = [
        "| Policy | N | Capture shown | Capture Kmax oracle | Attention entropy | Latency sec | Prompt tok | Parse fail |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, stats in sorted(summary["by_policy"].items()):
        def mean(name):
            return stats.get(name, {}).get("mean")
        lines.append(
            "| {policy} | {n} | {cap} | {fixed} | {ent} | {lat} | {ptok} | {fail} |".format(
                policy=policy,
                n=stats["n"],
                cap="" if mean("capture_within_shown") is None else f"{mean('capture_within_shown'):.3f}",
                fixed="" if mean("selected_vs_full_pool_oracle") is None else f"{mean('selected_vs_full_pool_oracle'):.3f}",
                ent="" if mean("attention_entropy") is None else f"{mean('attention_entropy'):.3f}",
                lat="" if mean("latency_ms") is None else f"{mean('latency_ms')/1000:.2f}",
                ptok="" if mean("prompt_tokens") is None else f"{mean('prompt_tokens'):.0f}",
                fail=stats["parse_failures"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    global MODEL, SLUG, CACHE_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--table-out", default=str(TABLE_OUT))
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--slug", default=None)
    parser.add_argument("--limit-days", type=int, default=None)
    parser.add_argument("--limit-calls", type=int, default=None)
    parser.add_argument("--max-tokens-base", type=int, default=512)
    parser.add_argument("--max-tokens-per-selection", type=int, default=160)
    parser.add_argument("--rerun-parse-failures", action="store_true")
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--include-qwen", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    MODEL = args.model
    SLUG = args.slug or f"openrouter__{model_slug(MODEL)}__listwise_dilution_v1"
    CACHE_DIR = pathlib.Path(args.cache_dir)

    started = time.time()
    rows = build_panel()
    global row_uid_for_cache
    row_uid_for_cache = {index: row_uid(row) for index, row in enumerate(rows)}
    tasks, _scores = build_tasks(rows, args.limit_days, args.include_controls, args.include_qwen)
    full_pool_by_day = {}
    for task in tasks:
        if task["policy"] == f"full_k{max(K_VALUES)}":
            full_pool_by_day[task["day"]] = task["candidate_indices"]
    for task in tasks:
        task["full_pool_indices"] = full_pool_by_day[task["day"]]

    log(f"[3/5] tasks={len(tasks)} eligible_days={len(set(t['day'] for t in tasks))} include_controls={args.include_controls} include_qwen={args.include_qwen}")
    if args.dry_run:
        by_policy = collections.Counter(task["policy"] for task in tasks)
        print(json.dumps({"tasks": len(tasks), "by_policy": dict(by_policy)}, indent=2), flush=True)
        return

    api_key = get_api_key()
    cache = load_cache()
    log(f"[4/5] cache={len(cache)}")
    records = []
    new_records = []
    calls = 0
    for i, task in enumerate(tasks, 1):
        key = cache_key(task["day"], task["policy"], task["candidate_indices"], SELECT_R)
        cached_parse_error = bool(key in cache and cache[key].get("parse_error") and cache[key].get("parse_error") != "json_recovered")
        if key in cache and not (args.rerun_parse_failures and cached_parse_error):
            record = cache[key]
            if len(record.get("selected") or []) < SELECT_R and record.get("response"):
                selected, parse_error = parse_selection(record["response"], set(range(len(task["candidate_indices"]))), SELECT_R)
                if len(selected) >= SELECT_R:
                    record = {**record, "selected": selected, "parse_error": parse_error}
        else:
            if args.limit_calls is not None and calls >= args.limit_calls:
                log(f"  reached limit_calls={args.limit_calls}")
                break
            prompt = build_prompt(rows, task["candidate_indices"], SELECT_R)
            max_tokens = args.max_tokens_base + args.max_tokens_per_selection * SELECT_R
            content, obj, latency_ms, error = call_openrouter(api_key, prompt, max_tokens)
            calls += 1
            usage = obj.get("usage") if isinstance(obj, dict) else {}
            selected, parse_error = parse_selection(content, set(range(len(task["candidate_indices"]))), SELECT_R)
            record = {
                "cache_key": key,
                "model": MODEL,
                "policy": task["policy"],
                "policy_family": task["policy_family"],
                "day": task["day"],
                "k": task["k"],
                "shortlist": task["shortlist"],
                "r": SELECT_R,
                "candidate_uids": [row_uid_for_cache[index] for index in task["candidate_indices"]],
                "response": content,
                "selected": selected,
                "parse_error": parse_error or error,
                "latency_ms": latency_ms,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
            append_cache([record])
            cache[key] = record
            new_records.append(record)
            if calls % 10 == 0:
                log(f"  calls={calls} tasks={i}/{len(tasks)}")
        selected = record.get("selected") or []
        metrics = evaluate_task(rows, task, selected) if len(selected) >= SELECT_R else {}
        records.append({**task, **record, **metrics})

    log("[5/5] Summarizing")
    by_policy_records = collections.defaultdict(list)
    for record in records:
        by_policy_records[record["policy"]].append(record)
    by_policy = {policy: summarize_metrics(vals) for policy, vals in by_policy_records.items()}
    result = {
        "task": "openrouter_listwise_dilution",
        "model": MODEL,
        "threshold": THRESHOLD,
        "origin_window": ORIGIN_WINDOW,
        "target": TARGET,
        "k_values": K_VALUES,
        "shortlists": SHORTLISTS,
        "select_r": SELECT_R,
        "n_tasks_planned": len(tasks),
        "n_records_evaluated": len(records),
        "n_new_api_calls": calls,
        "include_controls": args.include_controls,
        "include_qwen": args.include_qwen,
        "limit_days": args.limit_days,
        "limit_calls": args.limit_calls,
        "by_policy": by_policy,
        "records": records,
        "elapsed_sec": time.time() - started,
        "note": "Same-day cross-asset validation batches. K=30 is used because only 1 validation day has >=50 candidates.",
    }
    out_path = pathlib.Path(args.out)
    table_path = pathlib.Path(args.table_out)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    table_path.write_text(markdown_table(result), encoding="utf-8")
    log(f"wrote {out_path}")
    log(f"wrote {table_path}")
    log(f"elapsed={result['elapsed_sec']:.1f}s new_calls={calls}")


if __name__ == "__main__":
    main()
