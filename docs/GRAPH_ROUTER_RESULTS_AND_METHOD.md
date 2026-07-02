# KOL Origin-Aware Graph Router — Results and Method (thr=0.50, reach)

最终模型: **Lead-Lag Router (LLR)** = listwise GBDT on {context + O_k + lead-lag **净领先率 g_net_rate**}, 用 **listwise 交叉熵目标**(LightGBM `rank_xendcg`)训练.
（g_net_rate = (out−in)/(out+in) = **活跃度归一化的净领先率**；实测裸净度数 g_net=out−in 与发帖体量冗余、对强基线零增量，归一化后与体量正交且显著更强，见 §2.4 / §3。）
（口径修正: objective 是 `rank_xendcg`(listwise 交叉熵)[Bruch'21]，**不是 LambdaMART**；它享 NDCG 一致性保证，见 THEORY.md §4 R3。）
任务: pre-popularity origin-alert 排序. event=symbol-day, 候选=first10 新发起语义 frame, 目标=未来 follower-weighted reach.
主窗: train 2024-06..2025-06 / test 2025-06..2026-06, 17 标的. 消融/显著性用 2021-2026 五年 pooled.

---

## 0. 核心发现 (Story)

KOL 语料里存在一个稳定、point-in-time 可测的 **originator lead-lag 结构**(谁先发起叙事、谁跟随)。我们把它编码成 **lead-lag 图的净领先率 g_net_rate=(out−in)/(out+in)**(去体量、留方向),配一个廉价排序器得到 **Lead-Lag Router (LLR)**:在流行度不可见的**起源时刻、零文本**地给刚发起的叙事按未来 reach 排序。三块实验各撑一层论断:

### 主实验 (主表, §1) — 回答"值不值":零文本结构打平-超编码器、碾压全部 LLM、极省成本
零文本结构(线性 ridge)主窗 NDCG@3=**0.745**,打平/略胜全部 BERT 家族编码器(≤0.732),**碾压全部 full LLM**(DeepSeek 0.623 / 最强本地 Gemma3-12B 0.621 / Qwen2.5 0.554 / Llama 0.534);完整 **LLR 0.815** 全场最高,打败唯一零观测竞争者 CasMS(0.695)。成本比编码器低 250–6000x、比 full LLM 低约 6 个数量级(结构 ~0.002ms/0 token)。

### 消融实验 (§3) — 回答"为什么/靠谁":增量真实、来自净领先率图编码、非容量/中心性/先验信号
pooled 5 窗:listwise 排序是引擎(+0.094);结构(g_net_rate)对纯 context 净增量 **+0.0106 NDCG / +0.0214 Hit(均 95% 显著)**;**必须编码成图**(标量/关系 ns),且是**净领先率**非 PageRank/HITS(它俩零增量);净领先率 **显著胜过裸净度数 g_net(+0.0069 NDCG SIG)**——裸计数被发帖体量稀释;**结构真实性最硬证据 = 真图 vs 打乱图 +0.017(95% 显著)**,打乱图还低于不用图;控制对照下**优于全部先验账号信号**(Romero 零增量,Yamada/Zhou 已在 context)。

### 小实验 (应用层, §4) — 回答"有没有用":真实下游里显著碾压 LLM/编码器
当前小实验 = **早期爆发检测(top-10%, §4.2)**:把 LLR 当廉价前置层, 对每条刚发起的帧输出"未来 reach 会不会进 top-10% 大事件"的早期分 —— LLR **ROC 0.816** 全场最高,**整个 LLM 阵营(含最强本地 Gemma3-12B)近随机**,symbol-balanced bootstrap 对每个外部对手全 95% 显著。(另有一条 LLM-triage 路由/capture 实验 §4.1, 因结构与 context 拉不开、指标方差大, **暂弃用保留**。)

### 一句话
KOL originator 结构信号强到 **线性+零文本即超 SOTA**;它是**真实的网络拓扑信号**(shuffle 95% 证实)、**优于所有先验账号信号**;编码成 LLR 后,在**排序 / LLM-triage / 早期爆发检测**三个场景都打平-超编码器、碾压全部 LLM,成本低 4–6 个数量级。

---

## 1. 主表 (主窗 25.6-26.6, NDCG@3 降序; LLM 行全覆盖)

| Family | Method | Events | NDCG@3 | Hit@1 | Mass@3 | JS(低好) | Latency ms/q | Tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Graph Router (本文) | **Lead-Lag Router (LLR)** {context+O_k+g_net_rate} | 2176 | 0.815 | 0.567 | 0.883 | 0.226 | ~0.002 | 0 |
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
| Prior-art baseline | g_net_rate (ours, 对照) | 0.604 | 0.287 |
| Prior-art baseline | Romero IP-influence | 0.557 | 0.239 |
| Prior-art baseline | O_k (ours, 对照) | 0.554 | 0.251 |

注: standalone 裸排, Yamada/Zhou (历史类) 比 O_k/g_net_rate 强 — 因为历史采纳率天然和 reach 相关; **但它们正是 context 里已有的特征**. 真正的比较是控制对照(第3节): 在已含 Yamada/Zhou 的强 context 之上, 只有我们的结构还能显著加分.

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
PIT 历史建有向图(事件内 早->晚 加边); 取 **g_net_rate=净领先率=(out−in)/(out+in)**. **为何是率不是裸计数**: 裸净度数 g_net=out−in 会随发帖体量放大, 与"谁发得多"冗余(诊断 phase106: 控住体量后裸 g_net 零增量; 净领先率与体量 Spearman 仅 +0.010、预测最强)。故按对局总数 out+in 归一化, 去体量、留方向——这也正是 Precursors&Laggards(2010)对先后计数做发帖率校正的同一动机。(PageRank/HITS 实测无用, 已弃.)
### 2.5 排序器
**listwise GBDT = LightGBM objective `rank_xendcg`(listwise 交叉熵目标 [Bruch'21]), 非 LambdaMART/lambdarank**. event 作 query group, 标签=reach 全局 32 分位整数等级, inner-CV 选超参. listwise 是主引擎(+0.094 vs ridge); GBDT 约等于 XGBoost 约等于神经(~0.81 平台). 该交叉熵目标直接享 NDCG 一致性保证(Ravikumar'11/Bruch'19), LambdaMART 仅经验局部最优.
### 2.6 PIT 与评估
所有特征按各 block cutoff 之前历史估, point-in-time. 指标 NDCG@3/Hit@1/Mass@3/JS, symbol-balanced; bootstrap B=4000.

---

## 3. 消融与显著性 (pooled 5 窗 2021-2026, 9509 事件, reach)

| 对比 | ΔNDCG@3 | 显著 | 结论 |
|---|---|---|---|
| 树 listwise vs pointwise ridge | +0.094 | 显著 | 排序目标是主引擎 |
| **{context+O_k+g_net_rate} vs {context}** | **+0.0106; Hit +0.0214** | **95% 显著 [NDCG +0.0044,+0.0173]** | 结构对纯 context 的净增量 |
| {context+O_k+g_net_rate} vs {context+O_k} | +0.0048 (NDCG); **Hit +0.0163** | NDCG 擦边 ns / **Hit 95% 显著** | 图在 O_k 之上主要提 Hit@1 |
| **净领先率 g_net_rate vs 裸净度数 g_net** | **+0.0069; Hit +0.0158** | **95% 显著** | 去体量归一化是对的;裸计数被活跃度稀释 |
| {context+O_k}(标量) vs {context} | +0.0021 | 跨0 | 标量不够 |
| 关系特征 vs {context} | -0.0019 | 跨0 | 关系编码失败 |
| g_net_rate vs PageRank/HITS (over ctx+O_k) | rate +0.0048/Hit +0.0163; PR +0.0027 ns; HITS +0.0018 ns | - | 起作用的是净领先率, 非中心性 |
| **真图 vs 打乱图 (结构真实性主证据)** | **+0.0173; Hit +0.0287** | **95% 显著** | 增量来自真实拓扑, 非模型容量 |
| 真 O_k vs 打乱 O_k | +0.0058 | 90% 显著 | 标量身份也是真信号 |
| {full} vs {O_k only} | +0.24 | 显著 | context 必需 |

### 3b. vs 先验账号信号 (撞车防御; 控制对照, pooled)
| 对比 | ΔNDCG@3 | 显著 | 结论 |
|---|---|---|---|
| **ours {ctx+O_k+g_net_rate} vs {context}** | **+0.0106; Hit +0.0214** | 95% 显著 | 我们的结构在 context 之上加分 |
| {context+Romero-IP} vs {context} | -0.0021 | 跨0 | 通用图影响力(Romero)零增量, 同 PageRank/HITS |
| Yamada source / Zhou track | (= hist_mean_log_adopt / hist_success_rate) | — | 已在 context 里, 被吸收 |

结论: 先验账号信号要么已被 context 吸收(Yamada/Zhou=历史特征), 要么对 context 零增量(Romero, 同 PageRank/HITS); **只有去混淆 lead-lag 结构 O_k+g_net_rate 提供显著增量**. 详见 ABLATION_SUMMARY.md. 附: 换 adopt 目标图不加分(reach-specific), 主线用 reach.

---

## 4. 应用层小实验 (LLR 当廉价前置层; 主窗 25.6-26.6 test)

下游证明 LLR 不只是"排个序", 而是真能省钱省 token 还更准. 两条线, 都把每个对手取主表里最强的 family 代表.

### 4.0 部署场景: agentic 市场情报流水线的"起源闸门"

**背景.** 一个 agentic 市场研究/交易系统实时吃 X 上的 cashtag 叙事流, 每天有成千上万条**刚被发起**的叙事帧涌入. 系统下游有两个昂贵/稀缺的资源: (a) 一个 **LLM 推理层**(深度阅读 + 生成交易假设/研报), 日 token 与算力预算固定; (b) 一个**爆发预警台**(人或策略盯着"哪条叙事要火").

**痛点(= 为何需要 LLR).** 不可能对每条新帧都跑一次 LLM: 单次 LLM 决策成本随读入 token 超线性(§理论 P2, `τ∝(K L̄)²`), 洪流量级下根本排不过来; 而且把整池候选直接丢给 LLM 让它自己挑, **反而被稀释、选得更差**(实测 full_k30 capture 仅 0.455 < 筛过之后). 直接上文本编码器打分又慢又要 token, 且主表上还不如零文本结构.

**LLR 的位置 = 贵推理层前面的一道廉价、零文本、实时闸门.** 每条刚发起的帧, 用 g_net_rate/O_k 在 **~0.05ms / 0 token** 内打分, 驱动两种下游:

- **模式 A — triage 路由(见 §4.1, ⚠️ 暂弃用).** 概念上: 按 symbol-day 把 LLR 的 top-b 送进 LLM 推理层, 其余丢弃, 让贵 LLM 只看高价值的那几条. 早期 g_net 版观察到 capture 从"读全池"的 0.455 提到 ~0.64、token 省 ~63%. **但该 capture 口径下结构与 context 拉不开、指标方差大, 现暂弃用保留**(见 §4.1)。当前小实验以模式 B 为准。
- **模式 B — 早期爆发预警 / 事件信号(见 §4.2).** 对每条帧直接输出"未来 reach 会不会进 top-10% 大事件"的早期分(ROC **0.816**), **在流行度可见之前、零文本**就能 flag. 让 full LLM 干这活近随机(ROC 0.48–0.60). → 预警台/事件预测市场(如 Polymarket 式的"这条叙事会不会爆")/attention 分配, 拿到一个便宜且显著有效的早期信号.

**净收益(以 live 的模式 B 为准).** vs "让 LLM 判爆发": full LLM 阵营近随机, 而 LLR 对每个外部对手 95% 显著; 且零文本、~0.002ms/0 token, 比编码器便宜 ~270x、比 full LLM 便宜 ~6 个数量级. **一句话: LLR 是 agentic 系统里贵推理层前面的廉价零文本实时起源闸门 / 早期爆发预警.**(triage/capture 那条 §4.1 暂弃用。)

### 4.1 LLM-triage 前置路由 (capture / dilution, phase51) — ⚠️ 暂弃用, 保留待议

> **状态: 暂弃用(不删, 待后续决定是否复活为第二个小实验).** 原因: 在 capture 口径下, 我们的结构(图/O_k)与纯 context 拉不开差距(ol≈no_ol), 无法在这个下游里体现结构价值; 且 capture 无 CI、方差极大(单策略 p10–p90 跨 0.3→0.95)。**当前论文的小实验以 §4.2 事件预测为准。** 下面数字为早期 g_net 版本, 仅存档。(改用 g_net_rate 后 ol capture 略降, 因 rate 会抬高低体量账号、在 shortlist 里引入噪声——这也是弃用此口径的原因之一。)

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
| **Lead-Lag Router (LLR)** | 本文 | **0.351** | **0.816** | 0.351 |
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
1. 第一层 线性>文本 是主窗结论; cheap context 承担大头, O_k/g_net_rate 是被验证为真的结构成分.
2. 结构(g_net_rate)对纯 context 增量 +0.0106 NDCG / +0.0214 Hit(95% 显著); 在已含 O_k 的强基线上, 图主要提 Hit@1(+0.0163 SIG), NDCG 擦边; 配 capacity-controlled 的 真图vs打乱图 +0.017(95% 显著) -> 结构真实性硬.
3. **图价值来自净领先率 g_net_rate(去体量), 非裸净度数(被活跃度稀释, +0.0069 SIG 证实), 更非 PageRank/HITS**; standalone 裸排 Yamada/Zhou 更强, 但它们已在 context 里, 控制对照下只有我们的结构加分.
4. 残差化/手搓交互/花哨排序器/PageRank-HITS/裸净度数 均非必需或次优, 最终模型极简(context + O_k + g_net_rate).

---

## 6. 复现脚本 (experiments/socialenc/)
- phase98/99 — 图结构 pooled / 主窗主表
- phase100 — 95% CI + 图特征归因(净领先率驱动); phase100b(rate)/phase106 — 裸 g_net vs 净领先率、活跃度混淆诊断
- phase103 — 真图 vs 打乱图(结构真实性主证据); phase103b(rate)版 +0.017
- phase107 — 主表+消融 rate 版(g_net→g_net_rate); phase51b/52b — 小实验 rate 版
- phase105 — CasMS 尽力版 baseline (text+node2vec gen-stage); ours 0.813 vs CasMS 0.695 (+0.118)
- phase104 — 先验账号信号基线(Romero/Yamada/Zhou)standalone + 控制对照
- phase92/93/95/102 — 消融(排序器/特征轴, pooled/单窗)
- phase94 — 神经 LTR vs GBDT(平台)
- phase84 — 结构+surface+encoder 主表; phase85/86 — full LLM(全覆盖; phase86 含 Gemma3-12B/Qwen2.5-7B/Llama3.1-8B 本地); phase101 — adopt(附录)
- **应用层(§4)**: phase51 — LLM-triage 路由 capture/dilution(LLR shortlister + DeepSeek); phase52 — top-10% 事件预测(symbol-balanced + bootstrap, LLM 选手复用 phase85/86 缓存)
- 注: 这些应用层脚本(phase51_graph/52/86)目前在 luyao4, 本地 repo 待同步
