# KOL Router 2022-2026 Out-of-Time Retest Plan

Status: paused / superseded for now. The first frozen-train attempt used
`2020-01-01` to `2021-06-01` for training and `2022-06-01` to `2026-06-22`
for testing. This gap is too long for the intended evaluation question, because
it mixes model robustness with multi-year distribution drift. The resulting
`phase58` and `phase59` files should be treated as diagnostics only, not as the
paper-facing 2022-2026 retest.

Motivation: the current main table evaluates on 2021-06-01 to 2022-06-01. Since
the local archive extends to 2026-06-22, the paper needs a newer out-of-time
test. The intended protocol is to freeze the historical source-score
construction and router training, then evaluate on 2022-06-01 onward.

## Time Protocol

| Role | Period | Use |
|---|---|---|
| Source-score history | before 2020-01-01 | estimate KOL historical source score |
| Router train | 2020-01-01 to 2021-06-01 | fit scalar/ridge readouts |
| Development validation | 2021-06-01 to 2022-06-01 | existing main-table validation |
| Out-of-time test | 2022-06-01 to latest available data, currently 2026-06-22 | new paper-facing robustness test |

The retest should not retrain on 2021-2022 if we want a clean out-of-time claim.
Rows after 2021-06-01 can be scored, but only rows from 2022-06-01 onward enter
the OOT metrics.

## Retest Coverage Matrix

| Batch | Family | Methods | Existing source | 2022-2026 reproducibility | Action |
|---|---|---|---|---|---|
| 1 | Scale | Follower, Visibility | `phase7_origin_alert.py` | direct from scalar features | run in `phase58` |
| 1 | Context | Rank/Time, Sentiment, Novelty, History, No-OL Strong | `phase7_origin_alert.py` | direct from scalar features | run in `phase58` |
| 1 | Origin Role | OL Only, OL-Origin | `phase7_origin_alert.py` | direct from scalar features | run in `phase58` |
| 1 | Surface Text | Symbol one-hot, Text surface, Symbol + surface | `phase29_origin_alert_text_surface_diagnostic.py` | direct from origin text surface features | run in `phase58` |
| 1 | Ablation | No-OL, OL only, shuffled OL, follower replacement, raw OL, full OL | `phase33_origin_alert_ablation.py` | direct from scalar features | run in `phase59` or extend after phase58 |
| 2 | Text Encoder | BERT, FinBERT, BGE | `phase28_origin_alert_encoder_baselines.py` | reusable embedding cache; may need encode new 2022-2026 texts | run after phase58 |
| 2 | Text Encoder | Qwen3-4B | `phase39_qwen3_origin_alert_encoder_probe.py` | GPU needed, likely luyao4; may need encode new texts | run after phase58 |
| 2 | Text Encoder | E5-Mistral-7B | `phase41_e5_mistral_origin_alert_encoder_probe.py` | GPU needed, large model; may need encode new texts | optional if cost/time acceptable |
| 3 | Local LLM | Llama3.1-8B, Qwen2.5-7B | `phase18_origin_alert_llm_baselines.py` | Ollama needed on luyao4; new cache keys for 2022-2026 | run only if local models available |
| 3 | Commercial API | GPT Chat Latest, Claude Sonnet, Gemini, DeepSeek, Qwen | `phase32/phase43` | API cost; new prompts for 2022-2026 | run selectively after structural results |
| 4 | Listwise small experiment | Full LLM, Follower, No-OL, OL, Qwen selector | `phase50/53/54/55` | API cost; candidate pools shift after 2022 | defer until main OOT table stabilizes |

## Reproducibility Checklist

All rows below use the same frozen temporal protocol:

- Source-score history: before `2020-01-01`.
- Router/readout training: `2020-01-01` to `2021-06-01`.
- Development validation: `2021-06-01` to `2022-06-01`.
- OOT test: `2022-06-01` to `2026-06-22`.

| Object | Paper role | Reproduce with | Inputs required | Output target | Current status |
|---|---|---|---|---|---|
| Follower | scale baseline | `phase58_origin_alert_oot_2022_2026.py` | scalar panel only | `phase58_*` | completed on luyao4 |
| Visibility | scale baseline | `phase58_origin_alert_oot_2022_2026.py` | scalar panel only | `phase58_*` | completed on luyao4 |
| Rank/Time | timing/context baseline | `phase58_origin_alert_oot_2022_2026.py` | scalar panel only | `phase58_*` | completed on luyao4 |
| Sentiment | context baseline | `phase58_origin_alert_oot_2022_2026.py` | scalar panel only | `phase58_*` | completed on luyao4 |
| Novelty | context baseline | `phase58_origin_alert_oot_2022_2026.py` | scalar panel only | `phase58_*` | completed on luyao4 |
| History | context baseline | `phase58_origin_alert_oot_2022_2026.py` | scalar panel only | `phase58_*` | completed on luyao4 |
| No-OL Strong | strongest non-origin structural baseline | `phase58_origin_alert_oot_2022_2026.py` | scalar panel only | `phase58_*` | completed on luyao4 |
| OL Only | origin-role-only diagnostic | `phase58_origin_alert_oot_2022_2026.py` | pre-2020 OL score | `phase58_*` | completed on luyao4 |
| OL-Origin | proposed router row | `phase58_origin_alert_oot_2022_2026.py` | pre-2020 OL score + scalar context | `phase58_*` | completed on luyao4 |
| Symbol one-hot | surface diagnostic | `phase58_origin_alert_oot_2022_2026.py` | symbol id | `phase58_*` | completed on luyao4 |
| Text surface | shallow text baseline | `phase58_origin_alert_oot_2022_2026.py` | origin text surface features | `phase58_*` | completed on luyao4 |
| Symbol + surface | shallow text + asset baseline | `phase58_origin_alert_oot_2022_2026.py` | symbol id + origin text surface | `phase58_*` | completed on luyao4 |
| Ablation rows | mechanism check | `phase59_origin_alert_ablation_oot_2022_2026.py` | scalar panel only | `phase59_*` | completed on luyao4 |
| BERT/FinBERT/BGE | encoder challengers | `phase60_origin_alert_encoder_oot_2022_2026.py --slugs all` | embedding caches; encode missing 2022-2026 origin texts | `phase60_*` | script ready on luyao4 |
| Qwen3-4B | strongest encoder challenger | `phase60_origin_alert_encoder_oot_2022_2026.py --slugs bge_base,qwen3_embedding_4b_st` | GPU + Qwen embedding cache; encode missing 2022-2026 origin texts | `phase60_*` | running on luyao4 with BGE |
| E5-Mistral-7B | large encoder challenger | `phase60_origin_alert_encoder_oot_2022_2026.py --include-e5-mistral` | GPU + large model cache; likely slow | `phase60_*` | optional |
| Local LLM rows | LLM benchmark family | OOT wrapper around `phase18_origin_alert_llm_baselines.py` | local model server/cache | TBD | not started |
| Commercial API rows | API benchmark family | OOT wrapper around `phase32/phase43` | paid API budget/cache | TBD | defer unless needed |
| Listwise small experiment | applied routing scenario | OOT wrapper around `phase50/53/54/55` | API budget/cache | TBD | defer until table stabilizes |

## Luyao4 Execution Log

Current remote run:

```bash
cd /home/ycliu0703/workspace/projects/alphagap
nohup python3 experiments/socialenc/phase58_origin_alert_oot_2022_2026.py \
  > experiments/socialenc/phase58_origin_alert_oot_2022_2026_luyao4.log 2>&1 &
```

Monitor:

```bash
tail -f /home/ycliu0703/workspace/projects/alphagap/experiments/socialenc/phase58_origin_alert_oot_2022_2026_luyao4.log
```

## Immediate Execution Order

1. Run `phase58_origin_alert_oot_2022_2026.py`.
   - Produces the frozen-train 2022-2026 main structural/surface table.
   - Confirms event counts and whether OL-Origin remains competitive.
2. Add OOT bootstrap support for OL-Origin vs No-OL Strong and top baselines.
3. Run encoder OOT only for the strongest/most relevant challengers:
   - BGE
   - Qwen3-4B
   - optionally BERT/FinBERT/E5-Mistral for completeness.
4. Decide whether API/LLM OOT is worth cost after seeing batch 1 and 2.

## Success Criteria

Minimum acceptable paper update:

- OOT table for 2022-2026 including Scale, Context, Origin, Surface, and at
  least one strong encoder challenger.
- Bootstrap support for OL-Origin vs No-OL Strong on OOT.
- Clear statement that existing 2021-2022 validation was development
  validation, while 2022-2026 is the out-of-time test.

Full version:

- Complete OOT main table matching the current main table families.
- OOT ablation.
- OOT latency table remains unchanged for local scalar/encoder rows where
  scoring mechanics are the same; API latency should be remeasured if API
  baselines are rerun.
