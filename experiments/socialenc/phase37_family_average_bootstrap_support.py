"""Family-average bootstrap support for the main table.

This is a descriptive main-table support diagnostic. It averages OL-Origin
paired deltas across methods within each baseline family, then reports an
approximate 90% CI and one-sided p-value for the family-average delta.

It does not replace method-level paired bootstrap. Its purpose is to summarize
whether OL-Origin is consistently stronger than broad method families.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parent
OUT_JSON = ROOT / "phase37_family_average_bootstrap_support_result.json"
OUT_MD = ROOT / "phase37_family_average_bootstrap_support_table.md"

SETTING = "thr0.55_first10"
TARGET = "log_future_reach"
Z90 = 1.6448536269514722


def load(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def comparison(source: dict, key: str) -> dict:
    comp = source["by_setting"][SETTING]["targets"][TARGET]["comparisons"][key]
    out = {}
    for metric in ["ndcg3", "hit1", "mass3", "js"]:
        block = comp[metric]["symbol_balanced_bootstrap"]
        se = (block["ci95"] - block["ci05"]) / (2.0 * Z90)
        z = block["observed"] / se if se > 0 else math.inf
        out[metric] = {
            "observed": block["observed"],
            "ci05": block["ci05"],
            "ci95": block["ci95"],
            "se": se,
            "p_one_sided_positive_approx": 1.0 - norm_cdf(z),
        }
    return out


def family_average(rows: list[dict], metric: str) -> dict:
    vals = [row[metric]["observed"] for row in rows]
    ses = [
        row[metric].get("se")
        or row[metric].get("se_approx")
        or ((row[metric]["ci95"] - row[metric]["ci05"]) / (2.0 * Z90))
        for row in rows
    ]
    mean = sum(vals) / len(vals)
    within = math.sqrt(sum(se * se for se in ses)) / len(ses)
    between = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    se = math.sqrt(within * within + between * between / max(1, len(vals)))
    lo = mean - Z90 * se
    hi = mean + Z90 * se
    z = mean / se if se > 0 else math.inf
    return {
        "observed": mean,
        "ci05": lo,
        "ci95": hi,
        "se_approx": se,
        "p_one_sided_positive_approx": 1.0 - norm_cdf(z),
    }


def cell(block: dict) -> str:
    p = block["p_one_sided_positive_approx"]
    pstr = "<0.001" if p < 0.001 else f"{p:.3f}"
    return f"{block['observed']:+.3f} [{block['ci05']:+.3f}, {block['ci95']:+.3f}], p={pstr}"


def main() -> None:
    p7 = load("phase7_origin_alert_result.json")
    p18 = load("phase18_origin_alert_llm_baselines_result.json")
    p28 = load("phase28_origin_alert_encoder_baselines_result.json")
    p32 = load("phase32_openrouter_origin_alert_baselines_result.json")
    p36 = load("phase36_main_table_bootstrap_support_result.json")
    p36_by_name = {
        row["baseline"]: {metric: row[metric] for metric in ["ndcg3", "hit1", "mass3", "js"]}
        for row in p36["rows"]
    }

    families = {
        "Scale": [
            comparison(p7, "ol_origin_vs_followers"),
            comparison(p7, "ol_origin_vs_visibility"),
        ],
        "Context": [
            comparison(p7, "ol_origin_vs_rank_time"),
            comparison(p7, "ol_origin_vs_sentiment"),
            comparison(p7, "ol_origin_vs_novelty"),
            comparison(p7, "ol_origin_vs_history"),
            comparison(p7, "ol_origin_vs_no_ol_strong"),
        ],
        "Surface Text": [
            p36_by_name["Symbol one-hot"],
            p36_by_name["Text surface"],
            p36_by_name["Symbol + surface"],
        ],
        "Text Encoder": [
            comparison(p28, "ol_origin_vs_bert_base_origin_text"),
            comparison(p28, "ol_origin_vs_finbert_encoder_origin_text"),
            comparison(p28, "ol_origin_vs_e5_base_origin_text"),
            comparison(p28, "ol_origin_vs_bge_base_origin_text"),
        ],
        "Local LLM": [
            comparison(p18, "ol_origin_vs_llama3.1_8b_origin"),
            comparison(p18, "ol_origin_vs_qwen2.5_7b_origin"),
        ],
        "Commercial API": [
            comparison(p32, "ol_origin_vs_openai__gpt-4.1-mini"),
            comparison(p32, "ol_origin_vs_anthropic__claude-sonnet-4.5"),
            comparison(p32, "ol_origin_vs_google__gemini-2.5-flash"),
            comparison(p32, "ol_origin_vs_deepseek__deepseek-v3.2"),
        ],
    }

    rows = []
    for family, methods in families.items():
        row = {
            "family_average_baseline": family,
            "n_methods": len(methods),
            "metrics": {
                metric: family_average(methods, metric)
                for metric in ["ndcg3", "hit1", "mass3", "js"]
            },
        }
        rows.append(row)

    result = {
        "task": "family_average_bootstrap_support",
        "setting": SETTING,
        "target": TARGET,
        "note": "Approximate family-average support table. Deltas are OL-Origin minus the family-average baseline; JS is baseline JS minus OL-Origin JS, so positive is better.",
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "| Family average baseline | n methods | ΔNDCG@3 [90% CI], p | ΔHit@1 [90% CI], p | ΔMass@3 [90% CI], p | JS improvement [90% CI], p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        m = row["metrics"]
        lines.append(
            f"| {row['family_average_baseline']} | {row['n_methods']} | "
            f"{cell(m['ndcg3'])} | {cell(m['hit1'])} | "
            f"{cell(m['mass3'])} | {cell(m['js'])} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
