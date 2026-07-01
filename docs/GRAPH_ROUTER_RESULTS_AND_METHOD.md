# KOL Origin-Aware Graph Router — Results and Method (thr=0.50, reach)

最终模型: **Lead-Lag Router (LLR)** = listwise GBDT on {context + O_k + lead-lag 净领先度数 g_net}, 用 **listwise 交叉熵目标**(LightGBM `rank_xendcg`)训练.
（口径修正: objective 是 `rank_xendcg`(listwise 交叉熵)[Bruch'21]，**不是 LambdaMART**；它享 NDCG 一致性保证，见 THEORY.md §4 R3。）
任务: pre-popularity origin-alert 排序. event=symbol-day, 候选=first10 新发起语义 frame, 目标=未来 follower-weighted reach.
主窗: train 2024-06..2025-06 / test 2025-06..2026-06, 17 标的. 消融/显著性用 2021-2026 五年 pooled.

---

## 0. 核心发现 (Story)

KOL 语料里存在一个稳定、point-in-time 可测的 **originator lead-lag 结构**(谁先发起叙事、谁跟随)。围绕它本文给出两层结果,结构是两层的主角:

### 第一层 — 结构强到: 线性 + 零文本 > 文本 SOTA
简单线性 ridge over 结构化特征,**零文本**(0 token, 纯 CPU),主窗 NDCG@3=**0.745**,打败全部 BERT 家族编码器(≤0.732)与全部 full LLM(DeepSeek 0.623 / **最强本地 Gemma3-12B 0.621** / Qwen2.5 0.555 / Llama 0.534)。

### 第二层 — 排序算法 + 图结构 = SOTA 之上的显著增量
换 listwise GBDT 接同一结构底座 -> 0.811;再加 lead-lag 图净度数 g_net -> 0.812(即 **Lead-Lag Router**)。结构对 context 的增量 pooled 显著(+0.0072 NDCG / +0.0135 Hit, 90%)。真图 vs 打乱图 +0.011(95% 显著),打乱图甚至低于不用图 -> 增量来自真实拓扑。

### 一句话
KOL originator 结构信号强到 **线性+零文本即超 SOTA**;它是**真实的网络拓扑信号**(shuffle 控制证实),且**优于所有先验账号信号**(Romero/Yamada/Zhou, 见第3节);成本比文本编码器低 250-6000x、比 full LLM 低约 6 个数量级。

---

## 1. 主表 (主窗 25.6-26.6, NDCG@3 降序; LLM 行全覆盖)

| Family | Method | Events | NDCG@3 | Hit@1 | Mass@3 | JS(低好) | Latency ms/q | Tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Graph Router (本文) | **Lead-Lag Router (LLR)** {context+O_k+g_net} | 2176 | 0.812 | 0.557 | 0.878 | 0.225 | ~0.002 | 0 |
| Ranking Algo | listwise GBDT {context+O_k} | 2176 | 0.808-0.811 | 0.541 | 0.882 | 0.228 | ~0.002 | 0 |
| Ranking Algo | listwise GBDT {context} | 2176 | 0.805 | 0.529 | 0.878 | 0.215 | ~0.002 | 0 |
| Ranking Algo | XGBoost {context+O_k} | 2176 | 0.807 | 0.537 | 0.876 | 0.227 | ~0.002 | 0 |
| Origin Role (线性) | OL-Origin (ridge, 无文本) | 2176 | 0.745 | 0.436 | 0.836 | 0.274 | 0.0016 | 0 |
| Text Encoder | BGE-base | 2176 | 0.732 | 0.449 | 0.819 | 0.274 | 0.45 | 51.6 |
| Text Encoder | BERT-base | 2176 | 0.731 | 0.446 | 0.828 | 0.280 | 0.64 | 51.6 |
| Text Encoder | Qwen3-Embedding-4B | 2176 | 0.729 | 0.444 | 0.819 | 0.285 | 11.86 | 59.3 |
| Text Encoder | FinBERT | 2176 | 0.722 | 0.448 | 0.807 | 0.280 | 0.53 | 51.6 |
| Text Encoder | E5-Mistral-7B | 2176 | 0.696 | 0.424 | 0.788 | 0.313 | 25.99 | 65.2 |
| Prior-art method | CasMS-style (text+node2vec, gen-stage) | 2176 | 0.695 | 0.429 | 0.794 | 0.271 | ~12 | 59 |
| Prior-art method | node2vec graph-position only | 2176 | 0.494 | 0.208 | 0.629 | 0.324 | ~0.5 | 0 |
| Commercial LLM | DeepSeek V4 Flash (full, pointwise) | 2176 | 0.623 | 0.366 | 0.727 | 0.417 | 2688 | 206 |
| Local LLM | Gemma3-12B (full, pointwise) | 2176 | 0.621 | 0.322 | 0.737 | 0.434 | ~170 | ~330 |
| Local LLM | Qwen2.5-7B (full, pointwise) | 2174 | 0.555 | 0.257 | 0.685 | 0.483 | ~190 | ~330 |
| Local LLM | Llama3.1-8B (full, pointwise) | 2176 | 0.534 | 0.238 | 0.665 | 0.490 | ~172 | ~310 |

### 1b. Prior-art 账号信号基线 (standalone 单信号排序, 主窗 25.6-26.6)
先验方法的账号信号单独当排序分(类比 OL-Only 的 standalone):
| Family | Method (signal) | NDCG@3 | Hit@1 |
|---|---|---:|---:|
| Prior-art baseline | Zhou track-record (= hist_success_rate) | 0.682 | 0.393 |
| Prior-art baseline | Yamada source-spreader (= hist_mean_log_adopt) | 0.680 | 0.387 |
| Prior-art baseline | g_net (ours, 对照) | 0.604 | 0.287 |
| Prior-art baseline | Romero IP-influence | 0.557 | 0.239 |
| Prior-art baseline | O_k (ours, 对照) | 0.554 | 0.251 |

注: standalone 裸排, Yamada/Zhou (历史类) 比 O_k/g_net 强 — 因为历史采纳率天然和 reach 相关; **但它们正是 context 里已有的特征**. 真正的比较是控制对照(第3节): 在已含 Yamada/Zhou 的强 context 之上, 只有我们的结构还能显著加分.

LLM 行已全覆盖. 结构/GBDT 行 ~0.002 ms/q, 0 token, 纯 CPU.

---

## 2. 方法

### 2.1 数据与事件
findata KOL 推特档案, 17 标的. event=(symbol, UTC-day), >=8 KOL 参与; 每 KOL 取该事件首条推文按时间排序.
### 2.2 语义 frame 与候选
MiniLM(384维)在线贪心聚类, cos>=THR(0.50)并入否则新开 frame. 候选=first10. 目标=该 frame 24h 内后续 KOL 的 follower-weighted reach(log1p).
### 2.3 originator 标量 O_k
每事件按 ts 排序, 位置 i 的 net-lead=k+1-2*(i+1)(发帖时序, 非被转发计数); 跨事件平均(min n>=4)对 median UTC hour 二次回归取残差=O_k.
### 2.4 lead-lag 图特征
PIT 历史建有向图(事件内 早->晚 加边); 取 **g_net=净领先度数(out-in)**. (PageRank/HITS 实测无用, 已弃.)
### 2.5 排序器
**listwise GBDT = LightGBM objective `rank_xendcg`(listwise 交叉熵目标 [Bruch'21]), 非 LambdaMART/lambdarank**. event 作 query group, 标签=reach 全局 32 分位整数等级, inner-CV 选超参. listwise 是主引擎(+0.094 vs ridge); GBDT 约等于 XGBoost 约等于神经(~0.81 平台). 该交叉熵目标直接享 NDCG 一致性保证(Ravikumar'11/Bruch'19), LambdaMART 仅经验局部最优.
### 2.6 PIT 与评估
所有特征按各 block cutoff 之前历史估, point-in-time. 指标 NDCG@3/Hit@1/Mass@3/JS, symbol-balanced; bootstrap B=4000.

---

## 3. 消融与显著性 (pooled 5 窗 2021-2026, 9509 事件, reach)

| 对比 | ΔNDCG@3 | 显著 | 结论 |
|---|---|---|---|
| 树 listwise vs pointwise ridge | +0.094 | 显著 | 排序目标是主引擎 |
| **{context+O_k+g_net} vs {context}** | **+0.0072** | **90% 显著 [+0.0021,+0.0124]; Hit +0.0135 显著** | 结构对 context 的净增量 |
| {context+O_k}(标量) vs {context} | +0.0021 | 跨0 | 标量不够 |
| 关系特征 vs {context} | -0.0019 | 跨0 | 关系编码失败 |
| g_net vs PageRank/HITS (over ctx+O_k) | g_net +0.005; PR -0.003; HITS -0.005 | - | 起作用的是净度数, 非网络层级 |
| **真图 vs 打乱图 (结构真实性主证据)** | **+0.0112** | **90% 且 95% 显著** | 增量来自真实拓扑, 非模型容量 |
| 真 O_k vs 打乱 O_k | +0.0058 | 90% 显著 | 标量身份也是真信号 |
| {full} vs {O_k only} | +0.24 | 显著 | context 必需 |

### 3b. vs 先验账号信号 (撞车防御; 控制对照, pooled)
| 对比 | ΔNDCG@3 | 显著 | 结论 |
|---|---|---|---|
| **ours {ctx+O_k+g_net} vs {context}** | **+0.0072** | 90% 显著 | 我们的结构在 context 之上加分 |
| {context+Romero-IP} vs {context} | -0.0021 | 跨0 | 通用图影响力(Romero)零增量, 同 PageRank/HITS |
| **ours vs {context+Romero-IP}** | **+0.0092** | 90% 显著; Hit +0.0147 | 显著优于 Romero-增强模型 |
| Yamada source / Zhou track | (= hist_mean_log_adopt / hist_success_rate) | — | 已在 context 里, 被吸收 |

结论: 先验账号信号要么已被 context 吸收(Yamada/Zhou=历史特征), 要么对 context 零增量(Romero, 同 PageRank/HITS); **只有去混淆 lead-lag 结构 O_k+g_net 提供显著增量**. 详见 ABLATION_SUMMARY.md. 附: 换 adopt 目标图不加分(reach-specific), 主线用 reach.

---

## 4. 应用层小实验 (LLR 当廉价前置层; 主窗 25.6-26.6 test)

下游证明 LLR 不只是"排个序", 而是真能省钱省 token 还更准. 两条线, 都把每个对手取主表里最强的 family 代表.

### 4.1 LLM-triage 前置路由 (capture / dilution, phase51)
把 LLR 当 shortlister 接在 DeepSeek 前面: 同一个 LLM, 先用零文本结构把候选从全 30 池筛到 top-b 再喂. 指标 = capture (选中 reach 占全池 oracle top-3 的比例). **路由组件延迟 = 实测 ms/候选 (phase31/42) × 池 30**.

| Shortlister | family | capture@b10 | capture@b20 | 路由延迟/决策 |
|---|---|---:|---:|---:|
| **Lead-Lag Router (LLR)** | 本文 | **0.636** | 0.536 | **~0.05 ms** |
| Qwen3-Embedding-4B | 文本编码器 | 0.552 | 0.551 | ~356 ms |
| BGE-base | 文本编码器(主表最强) | 0.548 | 0.510 | ~13.6 ms |
| CasMS-style | prior-art(主表最强) | 0.534 | 0.523 | ~356 ms |
| follower | trivial | 0.455 | 0.497 | ~0.01 ms |
| random | 地板 | 0.350 | 0.453 | ~0 ms |
| **full_k30 (LLM 读全池)** | LLM 稀释基线 | **0.455** | — | **~2390 ms** |

读数: LLM 直接读全 30 池 capture 最差 (0.455, 稀释); LLR 筛到 10 再喂, 同一 LLM 升到 **0.636**, 且只花全池 ~37% token. **LLR 以 ~0.05ms 完成 triage, 比让 LLM 自己路由 (~2390ms) 快 ~5×10⁴, 比最强文本编码器 BGE 快 ~270x, 还 capture 最高.**

### 4.2 top-q% 大事件预测 (早期爆发检测, phase52)
重构为 pointwise 二分类: 起源时刻预测一条刚发起叙事的未来 reach 是否进 **全标的内 top-10%** (symbol-balanced, 按标的各取 top-10% 再宏平均, 与主表同口径). **无 LLM 在管线; LLM 仅作并列评分选手** (full LLM pointwise reach 打分, 复用主表缓存, 零额外成本).

| Method | family | PR-AUC | ROC-AUC | R-prec |
|---|---|---:|---:|---:|
| **Lead-Lag Router (LLR)** | 本文 | **0.351** | **0.812** | 0.348 |
| CasMS-style | prior-art | 0.281 | 0.738 | 0.298 |
| BGE-base | 文本编码器 | 0.266 | 0.724 | 0.296 |
| Qwen3-Embedding-4B | 文本编码器 | 0.262 | 0.727 | 0.275 |
| **Gemma3-12B (full LLM)** | LLM(最强本地) | **0.154** | **0.598** | 0.164 |
| DeepSeek V4 Flash (full LLM) | LLM(商用) | 0.145 | 0.580 | 0.148 |
| Qwen2.5-7B (full LLM) | LLM | 0.112 | 0.491 | 0.111 |
| Llama3.1-8B (full LLM) | LLM | 0.108 | 0.476 | 0.104 |
| follower / random | 地板 | 0.152/0.112 | 0.60/0.49 | — |

**symbol-balanced bootstrap CI (B=2000, 按标的 top-10%)**:

| 对比 | PR-AUC Δ (95% CI) | ROC-AUC Δ (95% CI) | 判定 |
|---|---|---|---|
| LLR vs Gemma3-12B (最强本地 LLM) | +0.201 [+0.124,+0.299] | +0.213 [+0.152,+0.297] | **95% 显著** |
| LLR vs DeepSeek (商用 LLM) | +0.208 [+0.128,+0.308] | +0.231 [+0.170,+0.306] | **95% 显著** |
| LLR vs BGE / Qwen3-4B (encoder) | +0.080 / +0.086 | +0.088 / +0.085 | **95% 显著** |
| LLR vs CasMS (prior-art) | +0.068 [−0.020,+0.169] | +0.074 [+0.025,+0.125] | **ROC 95% / PR 擦边** |

读数: ① **整个 LLM 阵营 (含最强本地 Gemma3-12B) 在全局爆发检测上近随机 (ROC 0.48–0.60), LLR 全 95% 显著碾压** —— 贵 LLM 在这个任务上失效; ② LLR 显著超所有文本编码器与 prior-art (CasMS ROC 95%). (注: 图 vs 纯 context 的结构增量是消融的范畴, 见 §3 + 真图/打乱图控制; 应用层只主张"廉价结构 ≫ 文本/LLM".)

---

## 5. 诚实边界
1. 第一层 线性>文本 是主窗结论; cheap context 承担大头, O_k/g_net 是被验证为真的结构成分.
2. 结构对 context 增量 +0.0072(90% NDCG+Hit 显著); 配 capacity-controlled 的 真图vs打乱图 +0.011(95% 显著) -> 结构真实性硬.
3. 图价值来自净度数 g_net, 非 PageRank/HITS; standalone 裸排 Yamada/Zhou 更强, 但它们已在 context 里, 控制对照下只有我们的结构加分.
4. 残差化/手搓交互/花哨排序器/PageRank-HITS 均非必需, 最终模型极简.

---

## 6. 复现脚本 (experiments/socialenc/)
- phase98/99 — 图结构 pooled / 主窗主表
- phase100 — 95% CI + 图特征归因(g_net 驱动)
- phase103 — 真图 vs 打乱图(结构真实性主证据)
- phase105 — CasMS 尽力版 baseline (text+node2vec gen-stage); ours 0.813 vs CasMS 0.695 (+0.118)
- phase104 — 先验账号信号基线(Romero/Yamada/Zhou)standalone + 控制对照
- phase92/93/95/102 — 消融(排序器/特征轴, pooled/单窗)
- phase94 — 神经 LTR vs GBDT(平台)
- phase84 — 结构+surface+encoder 主表; phase85/86 — full LLM(全覆盖; phase86 含 Gemma3-12B/Qwen2.5-7B/Llama3.1-8B 本地); phase101 — adopt(附录)
- **应用层(§4)**: phase51 — LLM-triage 路由 capture/dilution(LLR shortlister + DeepSeek); phase52 — top-10% 事件预测(symbol-balanced + bootstrap, LLM 选手复用 phase85/86 缓存)
- 注: 这些应用层脚本(phase51_graph/52/86)目前在 luyao4, 本地 repo 待同步
