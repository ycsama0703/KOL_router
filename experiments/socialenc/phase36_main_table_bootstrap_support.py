"""Main-table bootstrap support for OL-Origin.

This script builds a paper-facing statistical support table for the main
method-level experiment. It is not an ablation study.

Most comparisons are already stored in previous result JSON files. The only
missing main-table family is the surface-text diagnostic, whose original result
stored point estimates but not paired bootstrap comparisons. For that family,
this script reconstructs the main 0.55/first10 panel and computes OL-Origin vs
Text surface / Symbol+surface paired bootstrap.
"""
from __future__ import annotations

import json
import math
import pathlib
import time

import numpy as np

import phase5_sentiment_reconstruction as p5
import phase7_origin_alert as p7
import phase28_origin_alert_encoder_baselines as p28
import phase29_origin_alert_text_surface_diagnostic as p29

ROOT = pathlib.Path(__file__).resolve().parent
OUT_JSON = ROOT / "phase36_main_table_bootstrap_support_result.json"
OUT_MD = ROOT / "phase36_main_table_bootstrap_support_table.md"

SETTING = "thr0.55_first10"
TARGET = "log_future_reach"
THR = 0.55
ORIGIN_WINDOW = {"name": "first10", "max_rank": 10}
Z90 = 1.6448536269514722


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def add_pvals(block):
    out = dict(block)
    obs = float(out["observed"])
    lo = float(out["ci05"])
    hi = float(out["ci95"])
    se = (hi - lo) / (2.0 * Z90)
    z = obs / se if se > 0 else math.inf
    out["se_approx"] = se
    out["z_approx"] = z
    out["p_one_sided_positive_approx"] = 1.0 - norm_cdf(z)
    out["p_two_sided_approx"] = min(1.0, 2.0 * min(norm_cdf(z), 1.0 - norm_cdf(z)))
    return out


def extract_comparison(source, comp_key):
    comp = source["by_setting"][SETTING]["targets"][TARGET]["comparisons"][comp_key]
    return {
        metric: add_pvals(comp[metric]["symbol_balanced_bootstrap"])
        for metric in ["ndcg3", "hit1", "mass3", "js"]
    }


def surface_comparisons():
    rows_by = {}
    emb_by = {}
    all_rows = []
    for sym in p5.SYMS:
        rows, emb = p5.load_symbol(sym)
        rows_by[sym] = rows
        emb_by[sym] = emb
        all_rows.extend(rows)
    meta = p5.compute_metadata(all_rows)
    ol = p5.compute_oltrait(all_rows, meta)
    hist = p7.compute_origin_history(rows_by, emb_by, THR)
    rows, _events = p7.build_origin_panel(rows_by, emb_by, meta, ol, hist, THR, ORIGIN_WINDOW)
    matrices = p29.build_matrices(rows)
    scores = p7.train_scores(rows, TARGET)
    for method, X in matrices.items():
        sc = p28.fit_matrix_score(rows, X, TARGET, method, {})
        if sc is not None:
            scores[method] = sc
    evs = {name: p7.event_rows(rows, score, TARGET) for name, score in scores.items()}
    out = {}
    for base in ["text_surface", "symbol_plus_surface", "symbol_onehot"]:
        pairs = p7.aligned_pairs(evs["ol_origin"], evs[base])
        out[f"ol_origin_vs_{base}"] = {
            metric: add_pvals(p7.bootstrap_symbal(pairs, metric))
            for metric in ["ndcg3", "hit1", "mass3", "js"]
        }
    return out


def metric_cell(m):
    return (
        f"{m['observed']:+.3f} "
        f"[{m['ci05']:+.3f}, {m['ci95']:+.3f}], "
        f"p={m['p_one_sided_positive_approx']:.3f}"
    )


def main():
    t0 = time.time()
    p7res = load(ROOT / "phase7_origin_alert_result.json")
    p18 = load(ROOT / "phase18_origin_alert_llm_baselines_result.json")
    p28res = load(ROOT / "phase28_origin_alert_encoder_baselines_result.json")
    p32 = load(ROOT / "phase32_openrouter_origin_alert_baselines_result.json")

    comps = {}
    comps["No-OL Strong"] = extract_comparison(p7res, "ol_origin_vs_no_ol_strong")
    comps["Follower"] = extract_comparison(p7res, "ol_origin_vs_followers")
    comps["BERT-origin text"] = extract_comparison(p28res, "ol_origin_vs_bert_base_origin_text")
    comps["FinBERT-origin text"] = extract_comparison(p28res, "ol_origin_vs_finbert_encoder_origin_text")
    comps["E5-origin text"] = extract_comparison(p28res, "ol_origin_vs_e5_base_origin_text")
    comps["BGE-origin text"] = extract_comparison(p28res, "ol_origin_vs_bge_base_origin_text")
    comps["Llama3.1-8B"] = extract_comparison(p18, "ol_origin_vs_llama3.1_8b_origin")
    comps["Qwen2.5-7B"] = extract_comparison(p18, "ol_origin_vs_qwen2.5_7b_origin")
    comps["GPT-4.1-mini"] = extract_comparison(p32, "ol_origin_vs_openai__gpt-4.1-mini")
    comps["Claude Sonnet 4.5"] = extract_comparison(p32, "ol_origin_vs_anthropic__claude-sonnet-4.5")
    comps["Gemini 2.5 Flash"] = extract_comparison(p32, "ol_origin_vs_google__gemini-2.5-flash")
    comps["DeepSeek v3.2"] = extract_comparison(p32, "ol_origin_vs_deepseek__deepseek-v3.2")
    surface = surface_comparisons()
    comps["Symbol one-hot"] = surface["ol_origin_vs_symbol_onehot"]
    comps["Text surface"] = surface["ol_origin_vs_text_surface"]
    comps["Symbol + surface"] = surface["ol_origin_vs_symbol_plus_surface"]

    families = {
        "No-OL Strong": "Context",
        "Follower": "Scale",
        "Symbol one-hot": "Surface Text",
        "Text surface": "Surface Text",
        "Symbol + surface": "Surface Text",
        "BERT-origin text": "Text Encoder",
        "FinBERT-origin text": "Text Encoder",
        "E5-origin text": "Text Encoder",
        "BGE-origin text": "Text Encoder",
        "Llama3.1-8B": "Local LLM",
        "Qwen2.5-7B": "Local LLM",
        "GPT-4.1-mini": "Commercial API",
        "Claude Sonnet 4.5": "Commercial API",
        "Gemini 2.5 Flash": "Commercial API",
        "DeepSeek v3.2": "Commercial API",
    }
    order = [
        "No-OL Strong", "Follower", "Symbol one-hot", "Text surface", "Symbol + surface",
        "BERT-origin text", "FinBERT-origin text", "E5-origin text", "BGE-origin text",
        "Llama3.1-8B", "Qwen2.5-7B",
        "GPT-4.1-mini", "Claude Sonnet 4.5", "Gemini 2.5 Flash", "DeepSeek v3.2",
    ]
    rows = []
    for name in order:
        c = comps[name]
        rows.append({
            "family": families[name],
            "baseline": name,
            "ndcg3": c["ndcg3"],
            "hit1": c["hit1"],
            "mass3": c["mass3"],
            "js": c["js"],
        })

    result = {
        "task": "main_table_bootstrap_support",
        "setting": SETTING,
        "target": TARGET,
        "note": "All deltas are OL-Origin minus baseline; JS is baseline JS minus OL-Origin JS, so positive is better. p-values are approximate one-sided diagnostics inferred from stored bootstrap CIs except surface rows, where bootstrap was computed here.",
        "rows": rows,
        "elapsed_sec": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    md = [
        "| Family | Baseline | ΔNDCG@3 [90% CI], p | ΔHit@1 [90% CI], p | ΔMass@3 [90% CI], p | JS improvement [90% CI], p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['family']} | {r['baseline']} | "
            f"{metric_cell(r['ndcg3'])} | {metric_cell(r['hit1'])} | "
            f"{metric_cell(r['mass3'])} | {metric_cell(r['js'])} |"
        )
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(f"elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
