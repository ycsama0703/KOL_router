# socialenc — script manifest (graph paper vs legacy)

The scripts are kept **flat** in one directory on purpose: every `phaseN` script does
`import phaseM_...` of the others, and the **graph small-experiments reuse legacy-era machinery**
(e.g. `phase51_graph_listwise_dilution` imports `phase50` for the DeepSeek call/cache code,
`phase18`/`phase28` for prompts/encoders). Moving files into subdirectories would break the flat
import graph, so the graph-vs-legacy split is documented **here** rather than by physical folders.

Current paper = the **Lead-Lag Router (LLR)** graph version. Docs: `../../docs/` (legacy writeups in
`../../docs/legacy/`).

## A. Graph-version paper (current) — reproduction set

**Main table / structure (graph):**
- `phase98_graph_struct.py`, `phase99_graph_single_2566.py` — graph structure pooled / single window
- `phase100_graph_consolidate.py` — 95% CI + g_net feature attribution
- `phase103_graph_shuffle_control.py` — real vs shuffled graph (structure-reality main evidence)
- `phase104_priorart_baselines.py` — Romero / Yamada / Zhou prior-art (+ Romero IP), controlled
- `phase105_casms_baseline.py` — best-effort CasMS (text + node2vec) baseline
- `phase93_ablation_redesign_core.py`, `phase97_pooled_ablation_relational.py` — ablations
- `phase84_*` (main table build), `phase85_deepseek_thr050.py` / `phase86_localllm_thr050.py` — full-LLM rows
  (phase86 = Gemma3-12B / Qwen2.5-7B / Llama3.1-8B local pointwise)

**Application layer (§4 of GRAPH_ROUTER_RESULTS_AND_METHOD.md):**
- `phase51_graph_listwise_dilution.py` — LLM-triage routing (LLR shortlister → DeepSeek), capture/dilution + router latency
- `phase52_event_prediction.py` — top-10% big-event prediction (symbol-balanced + bootstrap CI)

**Shared base modules (legacy-era files, but in the graph closure — DO NOT treat as dead):**
- `phase5_sentiment_reconstruction.py` (load/meta/OLtrait), `phase7_origin_alert.py` (panel/FEATURE_SETS/metrics),
  `phase65_pit_lightweight_2025_2026.py` (PIT panel build), `phase33_origin_alert_ablation.py` (raw OL),
  `phase18_origin_alert_llm_baselines.py` (LLM prompt/call/parse, item_key), `phase21`/`phase28`
  (encoder caches + ridge readout), `phase39_qwen3_…` (qwen slug), `phase31`/`phase42` (cost benchmarks),
  `phase50_deepseek_listwise_dilution.py` (reused by phase51_graph for DeepSeek call/cache machinery)

## B. Legacy (pre-graph; superseded / dead — kept for archive only)

Not in the graph reproduction set; mostly old window/threshold sweeps and the original scalar-O_k line:
- `phase9/15/19/22/25/29/32/34/36/37_*` — early routing / encoder / surface / bootstrap-support probes
- `phase51_openrouter_listwise_dilution.py`, `phase53/54/55_*` — original (non-graph) LLM small-experiments
- `phase56/57_*` — econometric appendix / ridge params (old)
- `phase58/59/60_*_oot_2022_2026.py` — old out-of-time protocol
- `phase61–74_*` — old PIT/surface/qwen sweeps over 2024-2026 windows and thr 0.55/0.60

> Note: legacy leaf scripts would not import-resolve if relocated (they depend on the shared base
> modules in §A). They are archival; the live paper is §A.
