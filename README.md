# KOL Origin-Aware Narrative Router

This repository contains the supplementary materials for the KOL origin-aware
narrative routing paper topic. It includes the current paper working document,
the scripts used by the retained experiments, the derived social-media data
panel used by those scripts, and the result artifacts referenced in the paper
document.

## Repository Layout

```text
docs/
  KOL_NARRATIVE_AGENT_RESEARCH_LOG.md   main paper working document
  KOL_ROUTER_THEORETICAL_FRAMING.md      theoretical framing and equations
  KOL_ORIGINATION_STRUCTURE.md          supporting mechanism notes
  FINDATA_CATALOG.md                    data source catalog
  FINDATA_VERIFICATION.md               data verification notes

data/socialenc/
  {SYMBOL}.jsonl                        tweet and KOL metadata
  {SYMBOL}.npz                          MiniLM embeddings aligned by row index
  SHA256SUMS.txt                        data checksums
  file_sizes.csv                        data file sizes

experiments/socialenc/
  phase*.py                             retained experiment scripts and helpers
  phase*_result.json                    retained result artifacts
  phase*_table.{md,csv}                 retained tables
  phase*_cache.jsonl                    retained listwise LLM response caches
```

## Included Experiments

The repository follows the current paper document. Deprecated intermediate
experiments are intentionally excluded.

Included experiment groups:

- Main pre-popularity origin-alert experiment.
- Main-table latency analysis.
- Main-table bootstrap/statistical support.
- OL-Origin ablation study.
- Text-surface and text-encoder diagnostics used in the current table.
- Current agent-facing listwise routing small experiment:
  - DeepSeek V4 Flash
  - Claude Sonnet 4.6
  - GPT-5.4
  - Gemini 2.5 Flash Lite

Excluded examples:

- The earlier GPT Chat Latest listwise small-experiment result.
- The earlier Gemini 3.1 Flash Lite listwise small-experiment result.
- Other abandoned streaming/memory-write diagnostics not used by the current
  paper narrative.

## Data

The included data are the derived 17-symbol social-media panel used by the
retained experiments:

```text
AAPL MSFT NVDA TSLA AMZN META GOOGL AMD MSTR COIN HOOD PLTR SPY QQQ BTC ETH SOL
```

Each `{SYMBOL}.jsonl` row contains tweet/KOL metadata and text. Each
`{SYMBOL}.npz` contains 384-dimensional MiniLM embeddings aligned to the JSONL
row order. The unused `windows.npz` artifact from the broader internal project
is not included because none of the retained scripts reads it.

## Reproducing Existing Tables

Most retained result files are already committed under `experiments/socialenc/`.
Scripts are provided for auditability and reruns. Run commands from the
repository root, for example:

```bash
python experiments/socialenc/phase7_origin_alert.py
python experiments/socialenc/phase33_origin_alert_ablation.py
python experiments/socialenc/phase34_latency_quality_frontier.py
python experiments/socialenc/phase36_main_table_bootstrap_support.py
```

The listwise LLM scripts require API keys for reruns:

```text
~/.config/alphagap/deepseek_api_key
~/.config/alphagap/openrouter_api_key
```

The committed JSON/JSONL/Markdown result artifacts are sufficient to inspect the
reported listwise LLM results without rerunning paid API calls.

## Notes

- Scripts use point-in-time splits documented in
  `docs/KOL_NARRATIVE_AGENT_RESEARCH_LOG.md`.
- Local output paths were adapted for this standalone repository.
- Some encoder reruns require large Hugging Face models and GPU memory.
