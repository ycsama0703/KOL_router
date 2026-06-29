"""Phase67: PIT Qwen3-4B encoder benchmark for the 2025-2026 final test.

This uses the same point-in-time protocol as phase61:
  - train:      2022-06-01 to 2024-06-01
  - validation: 2024-06-01 to 2025-06-01
  - test:       2025-06-01 to 2026-06-01

Only the benchmark row is added here:
  - Qwen3-Embedding-4B over origin tweet text

The linear readout is selected by validation NDCG@3 and refit on
train+validation before evaluating the final test.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np

import phase5_sentiment_reconstruction as p5
import phase7_origin_alert as p7
import phase21_streaming_agent_encoder_baselines as p21
import phase28_origin_alert_encoder_baselines as p28
import phase39_qwen3_origin_alert_encoder_probe as p39
import phase65_pit_lightweight_2025_2026 as p61

OUT_JSON = pathlib.Path(__file__).with_name("phase67_pit_qwen3_4b_2025_2026_result.json")
OUT_MD = pathlib.Path(__file__).with_name("phase67_pit_qwen3_4b_2025_2026_table.md")

METHOD = f"{p39.QWEN3_4B_SLUG}_origin_text"
LABEL = "Qwen3-4B-origin text"


def log(msg: str) -> None:
    print(msg, flush=True)


def origin_text_matrix(rows: list[dict], cache: dict[str, np.ndarray]) -> np.ndarray:
    dim = len(next(iter(cache.values())))
    vectors = []
    for row in rows:
        text = p21.clean_text(row.get("origin_text", ""))
        vectors.append(cache.get(text, np.zeros(dim, dtype=np.float32)))
    return np.vstack(vectors).astype(np.float64)


def fit_select_matrix_score(rows: list[dict], X: np.ndarray):
    target = p61.TARGET
    y = np.array([r[target] for r in rows], dtype=float)
    splits = np.array([r["split"] for r in rows])
    train = splits == "train"
    val = splits == "val"
    train_val = train | val

    alpha_scores = {}
    for alpha in p61.RIDGE_ALPHAS:
        pred = p28.ridge_predict(X, y, train & np.isfinite(y), train & np.isfinite(y), alpha)
        evs = p61.event_rows_for_split(rows, pred, target, "val")
        alpha_scores[str(alpha)] = p7.symbal_mean(evs, "ndcg3")
    best_alpha = max(alpha_scores, key=lambda k: (alpha_scores[k], float(k)))
    best_alpha_f = float(best_alpha)
    final = p28.ridge_predict(X, y, train_val & np.isfinite(y), train_val & np.isfinite(y), best_alpha_f)
    return final, {"selected_alpha": best_alpha_f, "val_ndcg3_by_alpha": alpha_scores}


def summarize_method(rows, score):
    evs = p61.event_rows_for_split(rows, score, p61.TARGET, "test")
    return evs, {
        "n_events": len(evs),
        "n_symbols": len(set(e["sym"] for e in evs)),
        "symbol_balanced": {m: p7.symbal_mean(evs, m) for m in p7.METRICS},
        "pooled": {m: p7.pooled_mean(evs, m) for m in p7.METRICS},
    }


def fmt(x, digits=3):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def write_md(result):
    q = result["qwen3_4b"]["symbol_balanced"]
    lines = [
        "# Phase67 PIT Qwen3-4B Benchmark 2025-2026",
        "",
        "Protocol matches phase65: train 2022-06-01 to 2024-06-01; validation 2024-06-01 to 2025-06-01; final test 2025-06-01 to 2026-06-01.",
        "",
        "| Method | Events | Symbols | NDCG@3 | Hit@1 | Mass@3 | JS ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| {LABEL} | {result['qwen3_4b']['n_events']} | {result['qwen3_4b']['n_symbols']} | {fmt(q['ndcg3'])} | {fmt(q['hit1'])} | {fmt(q['mass3'])} | {fmt(q['js'])} |",
        "",
    ]
    if result.get("comparisons"):
        lines += [
            "Bootstrap comparisons:",
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
    t0 = time.time()
    log("[1/7] Loading tweets and embeddings")
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

    log("[2/7] Computing PIT block states")
    states = p61.compute_block_state(rows_by, emb_by, all_rows)

    log("[3/7] Building continuous PIT origin panel")
    panel, events = p61.build_pit_origin_panel(rows_by, emb_by, states)
    unique_texts = sorted({
        p21.clean_text(row.get("origin_text", ""))
        for row in panel
        if p21.clean_text(row.get("origin_text", ""))
    })
    log(
        f"  rows={len(panel)} train={sum(r['split']=='train' for r in panel)} "
        f"val={sum(r['split']=='val' for r in panel)} test={sum(r['split']=='test' for r in panel)} "
        f"events={len(events)} unique_origin_texts={len(unique_texts)}"
    )

    log("[4/7] Encoding Qwen3-4B origin texts")
    p21.BATCH_SIZE = 4
    cache = p39.encode_qwen3_st_texts(p39.QWEN3_4B_SLUG, p39.QWEN3_4B_CONFIG, unique_texts)

    log("[5/7] Fitting Qwen3-4B ridge readout")
    X = origin_text_matrix(panel, cache)
    score, selection = fit_select_matrix_score(panel, X)
    evs, means = summarize_method(panel, score)
    log(
        f"  {LABEL}: alpha={selection['selected_alpha']:g} "
        f"test_NDCG={means['symbol_balanced']['ndcg3']:.3f} "
        f"Hit={means['symbol_balanced']['hit1']:.3f} "
        f"JS={means['symbol_balanced']['js']:.3f}"
    )

    log("[6/7] Optional comparison against phase61 rows if available")
    comparisons = {}
    phase61_path = pathlib.Path(__file__).with_name("phase65_pit_lightweight_2025_2026_result.json")
    if phase61_path.exists():
        p61res = json.loads(phase61_path.read_text(encoding="utf-8"))
        # Refit the two structural rows on the same in-memory panel to avoid
        # depending on event-level data not stored in phase61 JSON.
        _scores, structural_evs, _means, _sel = p61.evaluate_methods(
            panel,
            {
                "no_ol_strong": p7.FEATURE_SETS["no_ol_strong"],
                "ol_origin": p7.FEATURE_SETS["ol_origin"],
            },
        )
        ev_map = {METHOD: evs, **structural_evs}
        comparisons = p61.comparisons(
            ev_map,
            [
                (METHOD, "no_ol_strong"),
                (METHOD, "ol_origin"),
                ("ol_origin", METHOD),
            ],
        )
    else:
        log("  phase61 result not found; skipping structural comparisons")

    result = {
        "task": "pit_qwen3_4b_origin_text_2025_2026",
        "protocol": {
            "train": [p61.TRAIN_START, p61.TRAIN_END],
            "validation": [p61.TRAIN_END, p61.VAL_END],
            "test": [p61.VAL_END, p61.TEST_END],
            "train_blocks": {
                "train_2022_2023": [p61.TRAIN_START, p61.TRAIN_MID],
                "train_2023_2024": [p61.TRAIN_MID, p61.TRAIN_END],
            },
            "test_blocks": {
                "test_2025_2026": [p61.VAL_END, p61.TEST_END],
            },
            "semantic_threshold": p61.THR,
            "origin_window": p61.ORIGIN_WINDOW,
            "target": p61.TARGET,
        },
        "latest_by_symbol": latest,
        "panel_counts": {
            "n_rows": len(panel),
            "n_train_rows": sum(r["split"] == "train" for r in panel),
            "n_val_rows": sum(r["split"] == "val" for r in panel),
            "n_test_rows": sum(r["split"] == "test" for r in panel),
            "n_events": len(events),
            "n_unique_origin_texts": len(unique_texts),
        },
        "encoder_model": p39.QWEN3_4B_CONFIG,
        "method": METHOD,
        "model_selection": selection,
        "qwen3_4b": means,
        "comparisons": comparisons,
        "elapsed_sec": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_md(result)
    log(f"[7/7] wrote {OUT_JSON.name} and {OUT_MD.name}")
    log(f"elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
