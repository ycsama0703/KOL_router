"""Phase52: top-q% BIG-EVENT prediction (early breakout detection), zero-LLM-in-loop.

Same panel / routers as phase51 (main window 25.6-26.6, thr=0.50, first10, reach),
but reframed as a POINTWISE binary task: at origination time (popularity unobservable),
will this freshly-originated frame's future follower-weighted reach land in the GLOBAL
top-q% of all test-period frames? Pure classification -> no LLM needed in the pipeline.

Competitors (each = a per-frame reach scorer; ranked then scored against the binary label):
  ol         our zero-text graph router {context+O_k+g_net} (LambdaMART)
  bge_base   strongest text encoder in the main table (ridge readout)
  casms      strongest prior-art (CasMS-style text+node2vec -> LambdaMART)
  qwen3_4b   text encoder (requested)
  deepseek   FULL LLM pointwise reach prediction (reads text)  -- reused from phase85 cache, free
  follower / random   trivial floors
  (+ appendix: other encoders, romero/yamada/zhou prior-art signals)

Metrics: PR-AUC (avg precision, primary for rare positive), ROC-AUC, R-precision (P@P),
precision@50, lift@P. q=10% global.
"""
from __future__ import annotations

import json
import pathlib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

import phase7_origin_alert as p7
import phase18_origin_alert_llm_baselines as p18
import phase65_pit_lightweight_2025_2026 as p65
import phase51_graph_listwise_dilution as p51

Q = 0.10
RNG_SEED = 20260630
SOC = pathlib.Path.home() / "workspace/projects/alphagap/experiments/socialenc"
PHASE85_CACHE = SOC / "phase85_deepseek_thr050_cache.jsonl"
LLM_CACHES = {  # competitor name -> cached pointwise reach_score jsonl (keyed by p18 item_key)
    "deepseek": SOC / "phase85_deepseek_thr050_cache.jsonl",            # commercial (main table)
    "gemma3_12b": SOC / "phase86_gemma3_12b_cache.jsonl",              # stronger local (NEW, added to main table too)
    "qwen2_5_7b": SOC / "phase86_qwen2_5_7b-instruct_cache.jsonl",     # local (main table)
    "llama3_1_8b": SOC / "phase86_llama3_1_8b_cache.jsonl",            # local (main table)
}
OUT = pathlib.Path.home() / "workspace/projects/alphagap/experiments/socialenc/phase52_event_prediction_result.json"
TABLE_OUT = pathlib.Path.home() / "workspace/projects/alphagap/experiments/socialenc/phase52_event_prediction_table.md"
HEADLINE = ["ol", "no_ol", "bge_base", "qwen3_4b", "casms",
            "deepseek", "gemma3_12b", "qwen2_5_7b", "llama3_1_8b", "follower", "random"]


def linear_ol_score(panel):
    # tier-1 hero: linear ridge over zero-text structure (OL-Origin feature set), train -> all
    feats = p7.FEATURE_SETS["ol_origin"]
    X = p65.matrix(panel, feats)
    y = np.array([float(r.get("log_future_reach", np.nan)) for r in panel], dtype=float)
    tr = np.array([r["split"] == "train" for r in panel], dtype=bool)
    trn = tr & np.isfinite(y)
    X = np.where(np.isfinite(X), X, 0.0)
    scaler = StandardScaler().fit(X[trn])
    Xs = scaler.transform(X)
    m = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0]).fit(Xs[trn], y[trn])
    pred = m.predict(Xs)
    return {i: float(pred[i]) for i in range(len(panel))}


def log(m):
    print(m, flush=True)


def load_llm_by_key(path):
    by_key = {}
    if not path.exists():
        return by_key
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                o = json.loads(line)
            except Exception:
                continue
            v = o.get("reach_score")
            if v is not None and str(v) != "None" and o.get("key"):
                try:
                    by_key[str(o["key"])] = float(v)
                except Exception:
                    pass
    return by_key


def metrics_for(score, y, test_idx, rng):
    # score: dict {panel_index: value} (may be missing rows); impute missing with median over test
    vals = np.array([score.get(i, np.nan) if score is not None else np.nan for i in test_idx], dtype=float)
    if not np.isfinite(vals).any():
        vals = rng.random(len(test_idx))
    med = np.nanmedian(vals) if np.isfinite(vals).any() else 0.0
    vals = np.where(np.isfinite(vals), vals, med)
    P = int(y.sum())
    order = np.argsort(-vals, kind="stable")
    topP = order[:P]
    top50 = order[:50]
    base_rate = y.mean()
    return {
        "pr_auc": float(average_precision_score(y, vals)),
        "roc_auc": float(roc_auc_score(y, vals)) if 0 < y.sum() < len(y) else float("nan"),
        "r_precision": float(y[topP].mean()),          # precision@P == recall@P
        "precision_at_50": float(y[top50].mean()),
        "lift_at_P": float(y[topP].mean() / base_rate) if base_rate > 0 else float("nan"),
        "n_pos": P,
        "base_rate": float(base_rate),
    }


def metrics_per_symbol(score, panel, test_idx, reach, rng, min_frames=30):
    # per-symbol top-q% label, rank WITHIN symbol, macro-average across symbols
    by_sym = {}
    for i in test_idx:
        by_sym.setdefault(panel[i]["sym"], []).append(i)
    pr, roc, rprec, lift, used = [], [], [], [], []
    for sym, idx in by_sym.items():
        if len(idx) < min_frames:
            continue
        r = reach[idx]
        thr = float(np.quantile(r, 1.0 - Q))
        y = (r >= thr).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        vals = np.array([score.get(i, np.nan) if score is not None else np.nan for i in idx], dtype=float)
        if not np.isfinite(vals).any():
            vals = rng.random(len(idx))
        med = np.nanmedian(vals) if np.isfinite(vals).any() else 0.0
        vals = np.where(np.isfinite(vals), vals, med)
        P = int(y.sum())
        order = np.argsort(-vals, kind="stable")
        pr.append(float(average_precision_score(y, vals)))
        roc.append(float(roc_auc_score(y, vals)))
        rprec.append(float(y[order[:P]].mean()))
        lift.append(float(y[order[:P]].mean() / y.mean()))
        used.append(sym)
    return {"pr_auc": float(np.mean(pr)), "roc_auc": float(np.mean(roc)),
            "r_precision": float(np.mean(rprec)), "lift_at_P": float(np.mean(lift)),
            "n_symbols": len(used)}


def bootstrap_ci(scores, panel, test_idx, reach, pairs, methods, B=2000, min_frames=30, seed=777):
    by_sym = {}
    for i in test_idx:
        by_sym.setdefault(panel[i]["sym"], []).append(i)
    sym_data = {}
    rng0 = np.random.default_rng(seed)
    for s, idx in by_sym.items():
        if len(idx) < min_frames:
            continue
        r = reach[idx]
        thr = float(np.quantile(r, 1.0 - Q))
        y = (r >= thr).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        vals = {}
        for name in methods:
            sc = scores.get(name)
            v = np.array([sc.get(i, np.nan) if sc is not None else np.nan for i in idx], dtype=float)
            if not np.isfinite(v).any():
                v = rng0.random(len(idx))
            med = np.nanmedian(v) if np.isfinite(v).any() else 0.0
            vals[name] = np.where(np.isfinite(v), v, med)
        sym_data[s] = (y, vals)
    symlist = list(sym_data)
    rng = np.random.default_rng(seed + 1)
    pr_draws = {m: [] for m in methods}
    roc_draws = {m: [] for m in methods}
    for _ in range(B):
        chosen = rng.choice(len(symlist), size=len(symlist), replace=True)
        accpr = {m: [] for m in methods}
        accroc = {m: [] for m in methods}
        for ci_ in chosen:
            y, vals = sym_data[symlist[ci_]]
            ii = rng.integers(0, len(y), len(y))
            yy = y[ii]
            if yy.sum() == 0 or yy.sum() == len(yy):
                continue
            for m in methods:
                vv = vals[m][ii]
                accpr[m].append(average_precision_score(yy, vv))
                accroc[m].append(roc_auc_score(yy, vv))
        for m in methods:
            pr_draws[m].append(np.mean(accpr[m]))
            roc_draws[m].append(np.mean(accroc[m]))
    out = {}
    for a, b in pairs:
        dpr = np.array(pr_draws[a]) - np.array(pr_draws[b])
        drc = np.array(roc_draws[a]) - np.array(roc_draws[b])
        def summ(d):
            return {"observed": float(d.mean()), "ci90": [float(np.percentile(d, 5)), float(np.percentile(d, 95))],
                    "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_gt0": float((d > 0).mean())}
        out[f"{a}_vs_{b}"] = {"pr_auc": summ(dpr), "roc_auc": summ(drc)}
    return out


def main():
    log("[1/4] Building GRAPH panel (main window 25.6-26.6, thr=0.50)")
    panel, inner, all_rows, tr_s, te_s = p51.build_graph_panel()

    log("[2/4] Fitting routers / encoders / prior-art / casms (no LLM calls)")
    scores = p51.routing_scores(panel, inner, include_qwen=True, include_encoders=True,
                                include_priorart=True, include_casms=True,
                                all_rows=all_rows, tr_s=tr_s, te_s=te_s)

    # LLM pointwise competitors (reuse cached reach_score, keyed by p18 item_key)
    key_of = [p18.item_key(p18.item_payload(r)) for r in panel]
    for name, path in LLM_CACHES.items():
        by_key = load_llm_by_key(path)
        sc = {i: by_key[k] for i, k in enumerate(key_of) if k in by_key}
        scores[name] = sc
        log(f"  {name} pointwise: {len(by_key)} cached keys, mapped {len(sc)}/{len(panel)} panel rows")

    # linear OL (tier-1 zero-text hero): ridge over structure
    scores["linear_ol"] = linear_ol_score(panel)

    # follower / random
    scores["follower"] = {i: float(r.get("origin_logfoll", np.nan)) for i, r in enumerate(panel)}
    rng = np.random.default_rng(RNG_SEED)
    scores["random"] = {i: float(rng.random()) for i in range(len(panel))}

    # test set + global top-q% label
    reach = np.array([float(r.get("future_reach", np.nan)) for r in panel], dtype=float)
    test_idx = [i for i, r in enumerate(panel) if r["split"] == "test" and np.isfinite(reach[i])]
    rt = reach[test_idx]
    thr = float(np.quantile(rt, 1.0 - Q))
    y = (rt >= thr).astype(int)
    log(f"[3/4] test frames={len(test_idx)}  top-{int(Q*100)}% threshold(reach)={thr:.1f}  positives={int(y.sum())} ({y.mean()*100:.1f}%)")

    mrng = np.random.default_rng(RNG_SEED + 1)
    res = {}
    res_sym = {}
    for name, sc in scores.items():
        res[name] = metrics_for(sc, y, test_idx, mrng)
        res_sym[name] = metrics_per_symbol(sc, panel, test_idx, reach, mrng)

    ordered = [n for n in HEADLINE if n in res] + sorted([n for n in res if n not in HEADLINE], key=lambda n: -res[n]["pr_auc"])

    def gtable():
        lines = ["### Global top-10%", "| Method | PR-AUC | ROC-AUC | R-precision (P@P) | P@50 | Lift@P |", "|---|---:|---:|---:|---:|---:|"]
        for n in ordered:
            m = res[n]
            lines.append(f"| {n} | {m['pr_auc']:.3f} | {m['roc_auc']:.3f} | {m['r_precision']:.3f} | {m['precision_at_50']:.3f} | {m['lift_at_P']:.2f} |")
        return "\n".join(lines)

    def stable():
        lines = [f"### Per-symbol top-10% (symbol-balanced macro, >=30 test frames; symbols used={res_sym[ordered[0]]['n_symbols']})",
                 "| Method | PR-AUC | ROC-AUC | R-precision (P@P) | Lift@P |", "|---|---:|---:|---:|---:|"]
        for n in ordered:
            m = res_sym[n]
            lines.append(f"| {n} | {m['pr_auc']:.3f} | {m['roc_auc']:.3f} | {m['r_precision']:.3f} | {m['lift_at_P']:.2f} |")
        return "\n".join(lines)

    log("computing symbol-balanced bootstrap CIs (B=2000) ...")
    pairs = [  # key external comparisons (family reps + LLMs)
        ("ol", "no_ol"), ("ol", "bge_base"), ("ol", "qwen3_4b"), ("ol", "casms"),
        ("ol", "deepseek"), ("ol", "gemma3_12b"),
        # appendix: internal structure increments (graph / O_k)
        ("ol", "lmart_ctxOk"), ("lmart_ctxOk", "no_ol")]
    methods = sorted({m for p in pairs for m in p})
    boot = bootstrap_ci(scores, panel, test_idx, reach, pairs, methods, B=2000)

    def btable():
        lines = ["### Symbol-balanced bootstrap CI (B=2000), per-symbol top-10%",
                 "| Comparison | metric | obs Δ | 90% CI | 95% CI | P(Δ>0) |", "|---|---|---:|---|---|---:|"]
        for k, v in boot.items():
            for metric in ["pr_auc", "roc_auc"]:
                s = v[metric]
                sig = "**SIG**" if (s["ci95"][0] > 0 or s["ci95"][1] < 0) else ("90%" if (s["ci90"][0] > 0 or s["ci90"][1] < 0) else "ns")
                lines.append(f"| {k} | {metric} | {s['observed']:+.4f} | [{s['ci90'][0]:+.4f},{s['ci90'][1]:+.4f}] | [{s['ci95'][0]:+.4f},{s['ci95'][1]:+.4f}] | {s['p_gt0']:.3f} {sig} |")
        return "\n".join(lines)

    table = gtable() + "\n\n" + stable() + "\n\n" + btable() + "\n"
    TABLE_OUT.write_text(table, encoding="utf-8")
    OUT.write_text(json.dumps({"q": Q, "window": "25.6-26.6", "split": "test", "n_test": len(test_idx),
                               "n_pos": int(y.sum()), "reach_threshold": thr,
                               "results_global": res, "results_per_symbol": res_sym, "bootstrap": boot}, indent=2), encoding="utf-8")
    log("[4/4] done")
    log(table)


if __name__ == "__main__":
    main()
