# Experiments

This directory contains the scripts and result artifacts referenced by the
current paper document.

Core scripts:

```text
phase5_sentiment_reconstruction.py         shared data loading and utilities
phase7_origin_alert.py                     main origin-alert experiment
phase28_origin_alert_encoder_baselines.py  text-encoder baselines
phase29_origin_alert_text_surface_diagnostic.py
phase31_origin_alert_cost_benchmark.py
phase33_origin_alert_ablation.py
phase34_latency_quality_frontier.py
phase36_main_table_bootstrap_support.py
phase37_family_average_bootstrap_support.py
phase39_qwen3_origin_alert_encoder_probe.py
phase41_e5_mistral_origin_alert_encoder_probe.py
phase42_new_encoder_cost_benchmark.py
phase50_deepseek_listwise_dilution.py
phase51_openrouter_listwise_dilution.py    shared OpenRouter listwise runner
```

Helper scripts are included when imported by the retained scripts. Their old
result artifacts are not included unless they are referenced by the paper
document.

Listwise small-experiment result files are included only for the retained
backends: DeepSeek V4 Flash, Claude Sonnet 4.6, GPT-5.4, and Gemini 2.5 Flash
Lite.
