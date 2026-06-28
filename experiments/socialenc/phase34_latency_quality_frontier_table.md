| Family | Method | NDCG@3 | Hit@1 | JS ↓ | Latency ms/q | x OL latency | Input Len | Pareto NDCG |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Scale | Follower | 0.724 | 0.477 | 0.292 | 0.0004 | 0.2 | 0.0 | yes |
| Scale | Visibility | 0.723 | 0.477 | 0.292 | 0.0005 | 0.3 | 0.0 | no |
| Context | History | 0.632 | 0.339 | 0.323 | 0.0006 | 0.3 | 0.0 | no |
| Context | No-OL Strong | 0.712 | 0.421 | 0.301 | 0.0013 | 0.8 | 0.0 | no |
| Context | Novelty | 0.673 | 0.362 | 0.328 | 0.0005 | 0.3 | 0.0 | no |
| Context | Rank/Time | 0.678 | 0.372 | 0.320 | 0.0006 | 0.3 | 0.0 | no |
| Context | Sentiment | 0.691 | 0.427 | 0.314 | 0.0005 | 0.3 | 0.0 | no |
| Origin Role | OL Only | 0.650 | 0.292 | 0.314 | 0.0004 | 0.2 | 0.0 | no |
| Origin Role | **OL-Origin** | 0.755 | 0.493 | 0.263 | 0.0016 | 1.0 | 0.0 | yes |
| Surface Text | Symbol + surface | 0.735 | 0.462 | 0.288 | 0.050 | 31.4 | 28.8 | no |
| Surface Text | Symbol one-hot | 0.678 | 0.372 | 0.319 | 0.0002 | 0.1 | 0.0 | yes |
| Surface Text | Text surface | 0.741 | 0.463 | 0.293 | 0.048 | 30.4 | 28.8 | no |
| Text Encoder | BERT-origin text | 0.745 | 0.440 | 0.298 | 0.645 | 406.4 | 51.6 | no |
| Text Encoder | BGE-origin text | 0.742 | 0.476 | 0.287 | 0.452 | 285.0 | 51.6 | no |
| Text Encoder | E5-origin text | 0.749 | 0.481 | 0.285 | 0.532 | 335.6 | 53.6 | no |
| Text Encoder | FinBERT-origin text | 0.742 | 0.447 | 0.295 | 0.534 | 336.7 | 51.6 | no |
| Local LLM | Llama3.1-8B | 0.619 | 0.291 | 0.474 | 172.487 | 108734.5 | 309.7 | no |
| Local LLM | Qwen2.5-7B | 0.612 | 0.273 | 0.493 | 190.410 | 120033.0 | 329.0 | no |
| Commercial API | Claude Sonnet 4.5 | 0.733 | 0.442 | 0.343 | 496.733 | 313136.6 | 197.3 | no |
| Commercial API | DeepSeek v3.2 | 0.699 | 0.433 | 0.378 | 794.315 | 500730.1 | 183.0 | no |
| Commercial API | GPT-4.1-mini | 0.644 | 0.322 | 0.449 | 332.244 | 209444.3 | 178.0 | no |
| Commercial API | Gemini 2.5 Flash | 0.711 | 0.394 | 0.396 | 195.458 | 123215.3 | 198.5 | no |
