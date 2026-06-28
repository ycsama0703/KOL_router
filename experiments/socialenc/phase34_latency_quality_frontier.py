"""Latency-quality frontier for pre-popularity origin routing.

This small experiment consolidates existing quality and cost measurements under
the main setting:
  - semantic threshold = 0.55
  - origin window = first10
  - target = future follower-weighted Reach

It does not retrain models or call any LLM/API. It creates:
  1) a JSON result file;
  2) a CSV table for plotting;
  3) a Markdown table for paper notes.
"""
from __future__ import annotations

import csv
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent

PHASE7 = ROOT / "phase7_origin_alert_result.json"
PHASE18 = ROOT / "phase18_origin_alert_llm_baselines_result.json"
PHASE28 = ROOT / "phase28_origin_alert_encoder_baselines_result.json"
PHASE29 = ROOT / "phase29_origin_alert_text_surface_diagnostic_result.json"
PHASE31 = ROOT / "phase31_origin_alert_cost_benchmark_result.json"
PHASE32 = ROOT / "phase32_openrouter_origin_alert_baselines_result.json"

OUT_JSON = ROOT / "phase34_latency_quality_frontier_result.json"
OUT_CSV = ROOT / "phase34_latency_quality_frontier_table.csv"
OUT_MD = ROOT / "phase34_latency_quality_frontier_table.md"
OUT_PNG = ROOT / "phase34_latency_quality_frontier_ndcg.png"

SETTING = "thr0.55_first10"
TARGET = "log_future_reach"
METRICS = ["ndcg3", "hit1", "mass3", "js"]


def load(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def nested(d, path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def quality(source, key):
    return nested(source, ["by_setting", SETTING, "targets", TARGET, "means", key])


def cost(costs, key):
    m = costs["methods"][key]
    latency = m.get("batch_ms_per_query")
    inp = nested(m, ["input_tokens", "mean"])
    input_kind = "tokens"
    if inp is None:
        inp = nested(m, ["input_words", "mean"])
        input_kind = "words"
    if inp is None:
        inp = 0.0
        input_kind = "none"
    return float(latency), float(inp), input_kind


def add_row(rows, family, method, source_key, source, cost_key, costs):
    q = quality(source, source_key)
    if not q:
        raise KeyError(f"missing quality for {source_key}")
    sb = q["symbol_balanced"]
    latency, input_len, input_kind = cost(costs, cost_key)
    rows.append({
        "family": family,
        "method": method,
        "source_key": source_key,
        "events": int(q["n_events"]),
        "symbols": int(q["n_symbols"]),
        "ndcg3": float(sb["ndcg3"]),
        "hit1": float(sb["hit1"]),
        "mass3": float(sb["mass3"]),
        "js": float(sb["js"]),
        "input_len": input_len,
        "input_kind": input_kind,
        "latency_ms_per_query": latency,
        "latency_source": "batch_amortized_local",
    })


def add_api_row(rows, family, method, source_key, api_source):
    q = quality(api_source, source_key)
    c = api_source["cost_summary"][source_key]
    sb = q["symbol_balanced"]
    rows.append({
        "family": family,
        "method": method,
        "source_key": source_key,
        "events": int(q["n_events"]),
        "symbols": int(q["n_symbols"]),
        "ndcg3": float(sb["ndcg3"]),
        "hit1": float(sb["hit1"]),
        "mass3": float(sb["mass3"]),
        "js": float(sb["js"]),
        "input_len": float(c["prompt_tokens_per_query_mean"]),
        "input_kind": "prompt_tokens",
        "latency_ms_per_query": float(c["latency_ms_per_query_mean"]),
        "latency_source": "batch10_api_wall_clock",
        "success_items": int(c["success_items"]),
        "failed_items": int(c["failed_items"]),
        "parse_fail_batches": int(c["parse_fail_batches"]),
    })


def mark_frontier(rows, quality_metric, higher_is_better=True):
    for r in rows:
        r[f"pareto_{quality_metric}"] = True
    for r in rows:
        for s in rows:
            if r is s:
                continue
            better_latency = s["latency_ms_per_query"] <= r["latency_ms_per_query"]
            if higher_is_better:
                better_quality = s[quality_metric] >= r[quality_metric]
                strict = (
                    s["latency_ms_per_query"] < r["latency_ms_per_query"]
                    or s[quality_metric] > r[quality_metric]
                )
            else:
                better_quality = s[quality_metric] <= r[quality_metric]
                strict = (
                    s["latency_ms_per_query"] < r["latency_ms_per_query"]
                    or s[quality_metric] < r[quality_metric]
                )
            if better_latency and better_quality and strict:
                r[f"pareto_{quality_metric}"] = False
                break


def fmt(x, nd=3):
    if x is None or not math.isfinite(float(x)):
        return ""
    return f"{float(x):.{nd}f}"


def fmt_latency_ms(x):
    x = float(x)
    if x < 0.01:
        return f"{x:.4f}"
    return f"{x:.3f}"


def write_plot(rows):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skip plot: matplotlib unavailable ({exc})")
        return None

    colors = {
        "Scale": "#6b7280",
        "Context": "#9ca3af",
        "Origin Role": "#dc2626",
        "Surface Text": "#2563eb",
        "Text Encoder": "#7c3aed",
        "Local LLM": "#f97316",
        "Commercial API": "#059669",
    }
    markers = {
        "Scale": "o",
        "Context": "o",
        "Origin Role": "*",
        "Surface Text": "s",
        "Text Encoder": "^",
        "Local LLM": "D",
        "Commercial API": "P",
    }
    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=180)
    for r in rows:
        ax.scatter(
            r["latency_ms_per_query"],
            r["ndcg3"],
            s=130 if r["method"] == "OL-Origin" else 58,
            color=colors.get(r["family"], "#111827"),
            marker=markers.get(r["family"], "o"),
            edgecolor="#111827" if r["method"] == "OL-Origin" else "white",
            linewidth=1.0,
            alpha=0.95,
            zorder=4 if r["method"] == "OL-Origin" else 3,
        )
        if r["method"] in {
            "OL-Origin", "Follower", "Text surface", "E5-origin text",
            "BGE-origin text", "Llama3.1-8B", "Claude Sonnet 4.5",
            "Gemini 2.5 Flash",
        }:
            ax.annotate(
                r["method"],
                (r["latency_ms_per_query"], r["ndcg3"]),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8,
            )
    ax.set_xscale("log")
    ax.set_xlabel("Latency per query (ms, log scale)")
    ax.set_ylabel("Reach NDCG@3")
    ax.set_title("Latency-quality frontier for pre-popularity origin routing")
    ax.grid(True, which="both", linestyle="--", linewidth=0.45, alpha=0.35)
    handles = []
    labels = []
    for fam in colors:
        handles.append(plt.Line2D([0], [0], marker=markers[fam], color="w", markerfacecolor=colors[fam], markersize=8))
        labels.append(fam)
    ax.legend(handles, labels, loc="lower left", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    plt.close(fig)
    return str(OUT_PNG)


def main():
    p7 = load(PHASE7)
    p18 = load(PHASE18)
    p28 = load(PHASE28)
    p29 = load(PHASE29)
    p31 = load(PHASE31)
    p32 = load(PHASE32)

    rows = []

    for family, method, key in [
        ("Scale", "Follower", "followers"),
        ("Scale", "Visibility", "visibility"),
        ("Context", "Rank/Time", "rank_time"),
        ("Context", "Sentiment", "sentiment"),
        ("Context", "Novelty", "novelty"),
        ("Context", "History", "history"),
        ("Context", "No-OL Strong", "no_ol_strong"),
        ("Origin Role", "OL Only", "ol_only"),
        ("Origin Role", "OL-Origin", "ol_origin"),
    ]:
        add_row(rows, family, method, key, p7, key, p31)

    for family, method, key in [
        ("Surface Text", "Symbol one-hot", "symbol_onehot"),
        ("Surface Text", "Text surface", "text_surface"),
        ("Surface Text", "Symbol + surface", "symbol_plus_surface"),
    ]:
        add_row(rows, family, method, key, p29, key, p31)

    for family, method, key in [
        ("Text Encoder", "BERT-origin text", "bert_base_origin_text"),
        ("Text Encoder", "FinBERT-origin text", "finbert_encoder_origin_text"),
        ("Text Encoder", "E5-origin text", "e5_base_origin_text"),
        ("Text Encoder", "BGE-origin text", "bge_base_origin_text"),
    ]:
        add_row(rows, family, method, key, p28, key, p31)

    for family, method, key in [
        ("Local LLM", "Llama3.1-8B", "llama3.1_8b_origin"),
        ("Local LLM", "Qwen2.5-7B", "qwen2.5_7b_origin"),
    ]:
        add_row(rows, family, method, key, p18, key, p31)

    for family, method, key in [
        ("Commercial API", "GPT-4.1-mini", "openai__gpt-4.1-mini"),
        ("Commercial API", "Claude Sonnet 4.5", "anthropic__claude-sonnet-4.5"),
        ("Commercial API", "Gemini 2.5 Flash", "google__gemini-2.5-flash"),
        ("Commercial API", "DeepSeek v3.2", "deepseek__deepseek-v3.2"),
    ]:
        add_api_row(rows, family, method, key, p32)

    ol = next(r for r in rows if r["method"] == "OL-Origin")
    for r in rows:
        r["latency_multiple_vs_ol"] = r["latency_ms_per_query"] / ol["latency_ms_per_query"]
        r["ndcg_delta_vs_ol"] = r["ndcg3"] - ol["ndcg3"]
        r["hit_delta_vs_ol"] = r["hit1"] - ol["hit1"]
        r["js_delta_vs_ol"] = r["js"] - ol["js"]
        r["qps"] = 1000.0 / r["latency_ms_per_query"] if r["latency_ms_per_query"] > 0 else None
        r["seconds_per_1k_queries"] = r["latency_ms_per_query"]
        r["ndcg_per_ms"] = r["ndcg3"] / r["latency_ms_per_query"]

    mark_frontier(rows, "ndcg3", higher_is_better=True)
    mark_frontier(rows, "hit1", higher_is_better=True)
    mark_frontier(rows, "js", higher_is_better=False)

    rows_sorted = sorted(rows, key=lambda r: (
        ["Scale", "Context", "Origin Role", "Surface Text", "Text Encoder", "Local LLM", "Commercial API"].index(r["family"]),
        r["method"],
    ))

    result = {
        "task": "latency_quality_frontier",
        "setting": SETTING,
        "target": TARGET,
        "latency_note": "Local methods use batch-amortized scoring latency from Phase31; commercial API methods use batch-size-10 per-query wall-clock latency from Phase32.",
        "excluded": {
            "gemma3_12b_origin": "Excluded because the full run was parse/cache unstable and only partial coverage was available."
        },
        "ol_origin_latency_ms_per_query": ol["latency_ms_per_query"],
        "rows": rows_sorted,
        "pareto_ndcg_methods": [r["method"] for r in rows_sorted if r["pareto_ndcg3"]],
        "pareto_hit_methods": [r["method"] for r in rows_sorted if r["pareto_hit1"]],
        "pareto_js_methods": [r["method"] for r in rows_sorted if r["pareto_js"]],
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    fieldnames = [
        "family", "method", "events", "symbols", "ndcg3", "hit1", "mass3", "js",
        "input_len", "input_kind", "latency_ms_per_query", "latency_multiple_vs_ol",
        "qps", "seconds_per_1k_queries", "ndcg_delta_vs_ol", "hit_delta_vs_ol",
        "js_delta_vs_ol", "pareto_ndcg3", "pareto_hit1", "pareto_js",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_sorted:
            w.writerow({k: r.get(k) for k in fieldnames})

    md_lines = [
        "| Family | Method | NDCG@3 | Hit@1 | JS ↓ | Latency ms/q | x OL latency | Input Len | Pareto NDCG |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows_sorted:
        method = r["method"]
        if method == "OL-Origin":
            method = f"**{method}**"
        md_lines.append(
            f"| {r['family']} | {method} | {fmt(r['ndcg3'])} | {fmt(r['hit1'])} | "
            f"{fmt(r['js'])} | {fmt_latency_ms(r['latency_ms_per_query'])} | "
            f"{fmt(r['latency_multiple_vs_ol'], 1)} | {fmt(r['input_len'], 1)} | "
            f"{'yes' if r['pareto_ndcg3'] else 'no'} |"
        )
    OUT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    plot_path = write_plot(rows_sorted)

    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")
    if plot_path:
        print(f"wrote {plot_path}")
    print("Pareto NDCG:", ", ".join(result["pareto_ndcg_methods"]))
    print("Pareto Hit:", ", ".join(result["pareto_hit_methods"]))
    print("Pareto JS:", ", ".join(result["pareto_js_methods"]))


if __name__ == "__main__":
    main()
