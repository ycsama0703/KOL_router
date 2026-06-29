"""Phase63: PIT surface-text diagnostics for the 2024-2026 final test.

Same protocol as phase61:
  - train:      2022-06-01 to 2023-06-01
  - validation: 2023-06-01 to 2024-06-01
  - test:       2024-06-01 to 2026-06-01

Rows:
  - symbol_onehot
  - text_surface
  - symbol_plus_surface
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np

import phase5_sentiment_reconstruction as p5
import phase7_origin_alert as p7
import phase28_origin_alert_encoder_baselines as p28
import phase29_origin_alert_text_surface_diagnostic as p29
import phase61_pit_lightweight_2024_2026 as p61

OUT_JSON = pathlib.Path(__file__).with_name("phase63_pit_surface_2024_2026_result.json")
OUT_MD = pathlib.Path(__file__).with_name("phase63_pit_surface_2024_2026_table.md")

METHOD_LABELS = {
    "symbol_onehot": "Symbol one-hot",
    "text_surface": "Text surface",
    "symbol_plus_surface": "Symbol + surface",
}


def log(msg: str) -> None:
    print(msg, flush=True)


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
    lines = [
        "# Phase63 PIT Surface Diagnostics 2024-2026",
        "",
        "Protocol matches phase61: train 2022-06-01 to 2023-06-01; validation 2023-06-01 to 2024-06-01; final test 2024-06-01 to 2026-06-01.",
        "",
        "| Method | Events | Symbols | NDCG@3 | Hit@1 | Mass@3 | JS ↓ | ΔNDCG vs No-OL | ΔHit vs No-OL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    no_ol = result["structural_means"]["no_ol_strong"]["symbol_balanced"]
    for method in ["symbol_onehot", "text_surface", "symbol_plus_surface"]:
        m = result["surface_means"][method]
        sb = m["symbol_balanced"]
        lines.append(
            f"| {METHOD_LABELS[method]} | {m['n_events']} | {m['n_symbols']} | "
            f"{fmt(sb['ndcg3'])} | {fmt(sb['hit1'])} | {fmt(sb['mass3'])} | {fmt(sb['js'])} | "
            f"{sb['ndcg3'] - no_ol['ndcg3']:+.3f} | {sb['hit1'] - no_ol['hit1']:+.3f} |"
        )
    lines += [
        "",
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

    log("[2/6] Computing PIT block states")
    states = p61.compute_block_state(rows_by, emb_by, all_rows)

    log("[3/6] Building continuous PIT origin panel")
    panel, events = p61.build_pit_origin_panel(rows_by, emb_by, states)
    log(
        f"  rows={len(panel)} train={sum(r['split']=='train' for r in panel)} "
        f"val={sum(r['split']=='val' for r in panel)} test={sum(r['split']=='test' for r in panel)} "
        f"events={len(events)}"
    )

    log("[4/6] Fitting surface diagnostics")
    matrices = p29.build_matrices(panel)
    surface_scores = {}
    surface_means = {}
    surface_evs = {}
    model_selection = {}
    for method in ["symbol_onehot", "text_surface", "symbol_plus_surface"]:
        score, sel = fit_select_matrix_score(panel, matrices[method])
        evs, means = summarize_method(panel, score)
        surface_scores[method] = score
        surface_evs[method] = evs
        surface_means[method] = means
        model_selection[method] = sel
        log(
            f"  {method:<20} alpha={sel['selected_alpha']:g} "
            f"NDCG={means['symbol_balanced']['ndcg3']:.3f} "
            f"Hit={means['symbol_balanced']['hit1']:.3f} "
            f"JS={means['symbol_balanced']['js']:.3f}"
        )

    log("[5/6] Fitting structural references for comparisons")
    _scores, structural_evs, structural_means, structural_selection = p61.evaluate_methods(
        panel,
        {
            "no_ol_strong": p7.FEATURE_SETS["no_ol_strong"],
            "ol_origin": p7.FEATURE_SETS["ol_origin"],
        },
    )
    ev_map = {**surface_evs, **structural_evs}
    comps = p61.comparisons(
        ev_map,
        [
            ("symbol_onehot", "no_ol_strong"),
            ("text_surface", "no_ol_strong"),
            ("symbol_plus_surface", "no_ol_strong"),
            ("ol_origin", "symbol_onehot"),
            ("ol_origin", "text_surface"),
            ("ol_origin", "symbol_plus_surface"),
        ],
    )

    result = {
        "task": "pit_surface_diagnostics_2024_2026",
        "protocol": {
            "train": [p61.TRAIN_START, p61.TRAIN_END],
            "validation": [p61.TRAIN_END, p61.VAL_END],
            "test": [p61.VAL_END, p61.TEST_END],
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
        },
        "surface_model_selection": model_selection,
        "structural_model_selection": structural_selection,
        "surface_means": surface_means,
        "structural_means": structural_means,
        "comparisons": comps,
        "elapsed_sec": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_md(result)
    log(f"[6/6] wrote {OUT_JSON.name} and {OUT_MD.name}")
    log(f"elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
