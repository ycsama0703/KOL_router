"""Phase29: diagnose why origin-text encoders are strong in Experiment 3.

This diagnostic tests whether simple surface cues in the origin tweet explain a
large part of the BERT-family origin-text baseline strength.

Rows:
  - symbol one-hot
  - text surface features
  - symbol + text surface

No OLtrait, KOL metadata, future text, or learned text encoder embeddings are
used in these diagnostic rows.
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import re
import time

import numpy as np

import phase5_sentiment_reconstruction as p5
import phase7_origin_alert as p7
import phase28_origin_alert_encoder_baselines as p28


OUT = pathlib.Path(__file__).with_name("phase29_origin_alert_text_surface_diagnostic_result.json")

THRESHOLDS = [0.55, 0.60, 0.65]
ORIGIN_WINDOW = {"name": "first10", "max_rank": 10}
SYMS = list(p5.SYMS)


def log(message: str) -> None:
    print(message, flush=True)


def text_surface_features(text: str, sym: str) -> list[float]:
    raw = text or ""
    lower = raw.lower()
    tokens = re.findall(r"[A-Za-z0-9_$#%]+", raw)
    chars = len(raw)
    letters = [c for c in raw if c.isalpha()]
    upper_ratio = sum(1 for c in letters if c.isupper()) / max(1, len(letters))
    cashtags = re.findall(r"\$[A-Za-z][A-Za-z0-9_]*", raw)
    hashtags = re.findall(r"#[A-Za-z][A-Za-z0-9_]*", raw)
    mentions = re.findall(r"@[A-Za-z][A-Za-z0-9_]*", raw)
    urls = re.findall(r"https?://\\S+|www\\.\\S+", raw)
    nums = re.findall(r"(?<![A-Za-z])[-+]?\\d+(?:\\.\\d+)?%?", raw)
    sym_lower = sym.lower()
    contains_sym = 1.0 if re.search(rf"(?<![A-Za-z])(?:\\$)?{re.escape(sym_lower)}(?![A-Za-z])", lower) else 0.0
    keyword_groups = {
        "earnings": ["earnings", "eps", "revenue", "guidance", "quarter"],
        "options": ["option", "calls", "puts", "flow", "strike"],
        "analyst": ["upgrade", "downgrade", "price target", "pt", "rating"],
        "macro": ["fed", "cpi", "inflation", "rates", "jobs"],
        "crypto": ["etf", "halving", "staking", "wallet", "chain", "defi"],
        "ai": [" ai ", "gpu", "chips", "datacenter", "model"],
        "legal": ["sec", "lawsuit", "court", "probe", "investigation"],
        "trade": ["breakout", "support", "resistance", "long", "short"],
    }
    features = [
        math.log1p(chars),
        math.log1p(len(tokens)),
        upper_ratio,
        float(len(cashtags)),
        float(len(hashtags)),
        float(len(mentions)),
        float(len(urls)),
        float(len(nums)),
        1.0 if "?" in raw else 0.0,
        1.0 if "!" in raw else 0.0,
        1.0 if "%" in raw else 0.0,
        contains_sym,
    ]
    padded = f" {lower} "
    for words in keyword_groups.values():
        features.append(1.0 if any(word in padded for word in words) else 0.0)
    return features


def build_matrices(rows: list[dict]) -> dict[str, np.ndarray]:
    sym_index = {sym: i for i, sym in enumerate(SYMS)}
    symbol = np.zeros((len(rows), len(SYMS)), dtype=float)
    surface = []
    for i, row in enumerate(rows):
        symbol[i, sym_index[row["sym"]]] = 1.0
        surface.append(text_surface_features(row.get("origin_text", ""), row["sym"]))
    surface = np.asarray(surface, dtype=float)
    return {
        "symbol_onehot": symbol,
        "text_surface": surface,
        "symbol_plus_surface": np.column_stack([symbol, surface]),
    }


def evaluate_scores(rows: list[dict], scores_by_method: dict[str, np.ndarray], target: str) -> dict:
    evs = {name: p7.event_rows(rows, scores, target) for name, scores in scores_by_method.items()}
    means = {}
    for name, ee in evs.items():
        means[name] = {
            "n_events": len(ee),
            "n_symbols": len(set(e["sym"] for e in ee)),
            "symbol_balanced": {metric: p7.symbal_mean(ee, metric) for metric in p7.METRICS},
        }
    return {"means": means}


def aggregate_main_region(by_setting: dict) -> dict:
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for setting in by_setting.values():
        for target, out in setting.get("targets", {}).items():
            for method, metrics in out.get("means", {}).items():
                for metric in ["ndcg3", "hit1", "mass3", "js"]:
                    value = metrics.get("symbol_balanced", {}).get(metric)
                    if value is not None and np.isfinite(value):
                        acc[method][target][metric].append(float(value))
    return {
        method: {
            target: {
                metric: float(np.mean(values)) if values else None
                for metric, values in metric_map.items()
            }
            for target, metric_map in target_map.items()
        }
        for method, target_map in acc.items()
    }


def main() -> None:
    started = time.time()
    log("[1/4] Loading tweets and embeddings")
    rows_by = {}
    emb_by = {}
    all_rows = []
    for i, sym in enumerate(p5.SYMS, 1):
        rows, emb = p5.load_symbol(sym)
        rows_by[sym] = rows
        emb_by[sym] = emb
        all_rows.extend(rows)
        log(f"  {i:02d}/{len(p5.SYMS)} {sym:<5} rows={len(rows):>6}")
    metadata = p5.compute_metadata(all_rows)
    ol = p5.compute_oltrait(all_rows, metadata)

    result = {
        "task": "origin_alert_text_surface_diagnostic",
        "origin_window": ORIGIN_WINDOW,
        "thresholds": THRESHOLDS,
        "targets": p7.TARGETS,
        "metrics": p7.METRICS,
        "methods": ["symbol_onehot", "text_surface", "symbol_plus_surface"],
        "by_setting": {},
    }

    log("[2/4] Building panels and fitting diagnostics")
    for threshold in THRESHOLDS:
        key = f"thr{threshold:.2f}_{ORIGIN_WINDOW['name']}"
        hist = p7.compute_origin_history(rows_by, emb_by, threshold)
        rows, events = p7.build_origin_panel(rows_by, emb_by, metadata, ol, hist, threshold, ORIGIN_WINDOW)
        matrices = build_matrices(rows)
        out = {
            "n_rows": len(rows),
            "n_train_rows": sum(row["split"] == "train" for row in rows),
            "n_val_rows": sum(row["split"] == "val" for row in rows),
            "n_events": len(events),
            "targets": {},
        }
        for target in p7.TARGETS:
            scores = {}
            for method, X in matrices.items():
                score = p28.fit_matrix_score(rows, X, target, method, result.setdefault("model_selection", {}))
                if score is not None:
                    scores[method] = score
            out["targets"][target] = evaluate_scores(rows, scores, target)
        result["by_setting"][key] = out

    log("[3/4] Aggregating")
    result["main_region_mean"] = aggregate_main_region(result["by_setting"])
    result["elapsed_sec"] = time.time() - started
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"[4/4] wrote {OUT}")
    log(f"elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
