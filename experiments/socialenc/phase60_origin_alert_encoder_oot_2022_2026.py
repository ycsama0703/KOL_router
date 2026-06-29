"""Phase60: 2022-2026 out-of-time encoder baselines for origin alert.

This is the OOT counterpart to phase28/phase39/phase41. It freezes:
  - historical source score: before 2020-01-01
  - router/readout training: 2020-01-01 to 2021-06-01

and evaluates only on 2022-06-01 onward.

Default encoder set is intentionally narrow: BGE plus Qwen3-4B, because those
are the strongest/relevant challengers in the current paper narrative. Use
`--slugs all` for BERT/FinBERT/E5-base/BGE/Qwen3-4B, and add
`--include-e5-mistral` only if the large model is worth the runtime.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import time

import numpy as np

import phase5_sentiment_reconstruction as p5
import phase7_origin_alert as p7
import phase21_streaming_agent_encoder_baselines as p21
import phase28_origin_alert_encoder_baselines as p28
import phase39_qwen3_origin_alert_encoder_probe as p39
import phase41_e5_mistral_origin_alert_encoder_probe as p41

OUT_JSON = pathlib.Path(__file__).with_name("phase60_origin_alert_encoder_oot_2022_2026_result.json")
OUT_MD = pathlib.Path(__file__).with_name("phase60_origin_alert_encoder_oot_2022_2026_table.md")

MODEL_SPLIT = "2021-06-01"
OOT_START = "2022-06-01"
OOT_END = "2026-06-23"
THR = 0.55
ORIGIN_WINDOW = {"name": "first10", "max_rank": 10}
TARGET = "log_future_reach"

BASE_SLUGS = ["bert_base", "finbert_encoder", "e5_base", "bge_base"]
DEFAULT_SLUGS = ["bge_base", p39.QWEN3_4B_SLUG]


def log(msg: str) -> None:
    print(msg, flush=True)


def configure_time_bounds() -> None:
    p5.TRAIN_END = "2020-01-01"
    p5.MODEL_SPLIT = MODEL_SPLIT
    p5.VAL_END = OOT_END
    p7.B = 600


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slugs",
        default=",".join(DEFAULT_SLUGS),
        help="Comma-separated encoder slugs, or 'all'.",
    )
    parser.add_argument(
        "--include-e5-mistral",
        action="store_true",
        help="Include intfloat/e5-mistral-7b-instruct. This can be slow.",
    )
    return parser.parse_args()


def requested_slugs(args) -> list[str]:
    if args.slugs.strip() == "all":
        slugs = list(BASE_SLUGS) + [p39.QWEN3_4B_SLUG]
    else:
        slugs = [s.strip() for s in args.slugs.split(",") if s.strip()]
    if args.include_e5_mistral and p41.E5_MISTRAL_SLUG not in slugs:
        slugs.append(p41.E5_MISTRAL_SLUG)
    return slugs


def encoder_configs(slugs):
    configs = dict(p21.MODEL_CONFIGS)
    configs[p39.QWEN3_4B_SLUG] = p39.QWEN3_4B_CONFIG
    configs[p41.E5_MISTRAL_SLUG] = p41.E5_MISTRAL_CONFIG
    missing = [slug for slug in slugs if slug not in configs]
    if missing:
        raise ValueError(f"Unknown encoder slug(s): {missing}")
    return {slug: configs[slug] for slug in slugs}


def encode_for_slug(slug: str, config: dict, texts: list[str]):
    if slug == p39.QWEN3_4B_SLUG:
        p21.BATCH_SIZE = 4
        return p39.encode_qwen3_st_texts(slug, config, texts)
    if slug == p41.E5_MISTRAL_SLUG:
        p21.BATCH_SIZE = 1
        return p41.encode_e5_mistral_texts(slug, config, texts)
    p21.BATCH_SIZE = 64
    return p21.encode_texts(slug, config, texts)


def event_rows_oot(rows, scores, target):
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r["split"] == "val" and OOT_START <= r["day"] < OOT_END and np.isfinite(scores[i]):
            groups[r["event_id"]].append((r["sym"], float(r[target]), float(scores[i])))
    out = []
    for event_id, vals in groups.items():
        if len(vals) < 2:
            continue
        y = np.array([v[1] for v in vals], dtype=float)
        s = np.array([v[2] for v in vals], dtype=float)
        if not np.isfinite(y).all() or not np.isfinite(s).all() or np.nanmax(y) <= 0:
            continue
        p = y / max(float(y.sum()), 1e-12)
        q = p5.softmax(s)
        order = np.argsort(-s)
        ideal = np.argsort(-y)
        k = min(3, len(y))
        best = set(np.where(y == np.nanmax(y))[0].tolist())
        out.append({
            "event_id": event_id,
            "sym": vals[0][0],
            "ndcg3": float(p5.dcg(y[order[:k]]) / max(p5.dcg(y[ideal[:k]]), 1e-12)),
            "hit1": float(1.0 if order[0] in best else 0.0),
            "mass3": float(p[order[:k]].sum()),
            "js": float(p5.js_divergence(p, q)),
        })
    return out


def summarize_scores(rows, scores_by_method):
    evs = {name: event_rows_oot(rows, scores, TARGET) for name, scores in scores_by_method.items()}
    means = {}
    for name, ee in evs.items():
        means[name] = {
            "n_events": len(ee),
            "n_symbols": len(set(e["sym"] for e in ee)),
            "symbol_balanced": {m: p7.symbal_mean(ee, m) for m in p7.METRICS},
            "pooled": {m: p7.pooled_mean(ee, m) for m in p7.METRICS},
        }
    return evs, means


def bootstrap_vs_ol(evs, methods):
    comps = {}
    if "ol_origin" not in evs:
        return comps
    for base in methods:
        if base not in evs or base == "ol_origin":
            continue
        pairs = p7.aligned_pairs(evs["ol_origin"], evs[base])
        comps[f"ol_origin_vs_{base}"] = {
            metric: {
                "symbol_balanced_bootstrap": p7.bootstrap_symbal(pairs, metric),
                "pooled_bootstrap": p7.bootstrap_pooled(pairs, metric),
            }
            for metric in p7.METRICS
        }
    return comps


def fmt(x, digits=3):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def write_md(result):
    no_ol = result["means"]["no_ol_strong"]["symbol_balanced"]
    rows = []
    labels = {
        "no_ol_strong": "No-OL strong",
        "ol_origin": "OL-Origin",
        "bert_base_origin_text": "BERT-origin text",
        "finbert_encoder_origin_text": "FinBERT-origin text",
        "e5_base_origin_text": "E5-base-origin text",
        "bge_base_origin_text": "BGE-origin text",
        f"{p39.QWEN3_4B_SLUG}_origin_text": "Qwen3-4B-origin text",
        f"{p41.E5_MISTRAL_SLUG}_origin_text": "E5-Mistral-7B-origin text",
    }
    for key in ["no_ol_strong", "ol_origin", *result["encoder_methods"]]:
        if key not in result["means"]:
            continue
        m = result["means"][key]
        sb = m["symbol_balanced"]
        label = labels.get(key, key)
        if key == "ol_origin":
            label = f"**{label}**"
        rows.append(
            f"| {label} | {m['n_events']} | {m['n_symbols']} | "
            f"{fmt(sb['ndcg3'])} | {fmt(sb['hit1'])} | {fmt(sb['mass3'])} | {fmt(sb['js'])} | "
            f"{sb['ndcg3'] - no_ol['ndcg3']:+.3f} | {sb['hit1'] - no_ol['hit1']:+.3f} |"
        )
    lines = [
        "# Phase60 2022-2026 OOT Encoder Baselines",
        "",
        "Frozen protocol: source-score history before 2020-01-01; router train 2020-01-01 to 2021-06-01; OOT test 2022-06-01 to 2026-06-22.",
        "",
        "| Method | Events | Symbols | NDCG@3 | Hit@1 | Mass@3 | JS ↓ | ΔNDCG vs No-OL | ΔHit vs No-OL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "Bootstrap support, OL-Origin vs selected encoder rows:",
        "",
        "| Comparison | ΔNDCG@3 90% CI | ΔHit@1 90% CI | JS improvement 90% CI |",
        "|---|---:|---:|---:|",
    ]
    for name, comp in result["comparisons"].items():
        nd = comp["ndcg3"]["symbol_balanced_bootstrap"]
        hit = comp["hit1"]["symbol_balanced_bootstrap"]
        js = comp["js"]["symbol_balanced_bootstrap"]
        lines.append(
            f"| {name} | {nd['observed']:+.3f} [{nd['ci05']:+.3f}, {nd['ci95']:+.3f}] | "
            f"{hit['observed']:+.3f} [{hit['ci05']:+.3f}, {hit['ci95']:+.3f}] | "
            f"{js['observed']:+.3f} [{js['ci05']:+.3f}, {js['ci95']:+.3f}] |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    slugs = requested_slugs(args)
    configs = encoder_configs(slugs)
    started = time.time()
    configure_time_bounds()

    log("[1/6] Loading tweets and embeddings")
    rows_by = {}
    emb_by = {}
    all_rows = []
    latest = {}
    for i, sym in enumerate(p5.SYMS, 1):
        rows, emb = p5.load_symbol(sym)
        rows_by[sym] = rows
        emb_by[sym] = emb
        all_rows.extend(rows)
        latest[sym] = max(r["day"] for r in rows) if rows else None
        log(f"  {i:02d}/{len(p5.SYMS)} {sym:<5} rows={len(rows):>6}")

    log("[2/6] Building first10 thr=0.55 origin panel")
    metadata = p5.compute_metadata(all_rows)
    ol = p5.compute_oltrait(all_rows, metadata)
    hist = p7.compute_origin_history(rows_by, emb_by, THR)
    panel, events = p7.build_origin_panel(rows_by, emb_by, metadata, ol, hist, THR, ORIGIN_WINDOW)

    unique_texts = sorted({
        p21.clean_text(row.get("origin_text", ""))
        for row in panel
        if p21.clean_text(row.get("origin_text", ""))
    })
    log(f"[3/6] Encoding origin texts n={len(unique_texts)} slugs={','.join(slugs)}")
    caches = {slug: encode_for_slug(slug, configs[slug], unique_texts) for slug in slugs}

    log("[4/6] Fitting structural and encoder readouts")
    scores = p7.train_scores(panel, TARGET)
    model_selection = {}
    encoder_methods = []
    for slug in slugs:
        method = f"{slug}_origin_text"
        X = p28.origin_text_matrix(panel, caches[slug])
        score = p28.fit_matrix_score(panel, X, TARGET, method, model_selection)
        if score is not None:
            scores[method] = score
            encoder_methods.append(method)

    evs, means = summarize_scores(panel, scores)
    result = {
        "task": "origin_alert_encoder_oot_2022_2026",
        "protocol": {
            "source_score_history_end": p5.TRAIN_END,
            "router_train_period": [p5.TRAIN_END, MODEL_SPLIT],
            "development_validation_period": [MODEL_SPLIT, OOT_START],
            "oot_test_period": [OOT_START, OOT_END],
            "semantic_threshold": THR,
            "origin_window": ORIGIN_WINDOW,
            "target": TARGET,
        },
        "encoder_slugs": slugs,
        "encoder_models": configs,
        "encoder_methods": encoder_methods,
        "latest_by_symbol": latest,
        "panel_counts": {
            "n_rows": len(panel),
            "n_train_rows": sum(r["split"] == "train" for r in panel),
            "n_post_train_rows": sum(r["split"] == "val" for r in panel),
            "n_oot_rows": sum(r["split"] == "val" and OOT_START <= r["day"] < OOT_END for r in panel),
            "n_events_raw": len(events),
            "n_unique_origin_texts": len(unique_texts),
        },
        "model_selection": model_selection,
        "means": means,
        "comparisons": bootstrap_vs_ol(evs, ["no_ol_strong", *encoder_methods]),
        "elapsed_sec": time.time() - started,
    }
    log("[5/6] Writing outputs")
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_md(result)
    log(f"[6/6] wrote {OUT_JSON.name} and {OUT_MD.name}")
    log(f"elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
