# KOL Origin-Aware Graph Router — Results and Method (thr=0.50, reach)

最终模型: **LambdaMART (listwise GBDT) on {context + O_k + lead-lag 净领先度数 g_net}**.
任务: pre-popularity origin-alert 排序. event=symbol-day, 候选=first10 新发起语义 frame, 目标=未来 follower-weighted reach.
主窗: train 2024-06..2025-06 / test 2025-06..2026-06, 17 标的. 消融/显著性用 2021-2026 五年 pooled.

---

## 0. 核心发现 (Story)

KOL 语料里存在一个稳定、point-in-time 可测的 **originator lead-lag 结构**(谁先发起叙事、谁跟随)。围绕它,本文给出两层结果,结构是两层的主角:

### 第一层 — 结构强到: 线性 + 零文本 > 文本 SOTA
一个**简单线性模型**(ridge)over 结构化 originator 特征,**完全不读文本**(0 token, 纯 CPU),主窗 NDCG@3 = **0.745**,打败:
- **全部 BERT 家族编码器**: BGE 0.732 / BERT 0.731 / Qwen3-4B 0.729 / FinBERT 0.722 / E5-Mistral-7B 0.696;
- **全部 full LLM**: DeepSeek V4 Flash 0.623 / Qwen2.5-7B 0.555 / Llama3.1-8B 0.534.

即: 这个结构信号强到 **线性 + 零文本就超过所有文本 SOTA(含 4B/7B 编码器与商用/本地 full LLM)**。

### 第二层 — 排序算法 + 图结构 = SOTA 之上的显著增量
把线性读出换成 listwise GBDT(LambdaMART)接在**同一结构底座**上 -> **0.811**(+0.066);再把结构编码成 lead-lag 图(g_net)-> **0.812**,这个增量 pooled 90% 显著。
关键控制: 把真图中心性换成身份打乱版,会**显著掉点**(真图 vs 打乱图 +0.011, 连 95% 也显著; 打乱图甚至低于不用图)-> 增量来自**真实的 origination 网络拓扑**,不是加列蹭噪声。

### 一句话
KOL originator 结构信号强到 **线性+零文本即超 SOTA**; 它是**真实的网络拓扑信号**(shuffle 控制证实); 而成本比文本编码器低 250-6000x、比 full LLM 低约 6 个数量级。
证据: 主表见第 1 节; 消融与显著性见第 3 节 / ABLATION_SUMMARY.md。

---

## 1. 主表 (主窗 25.6-26.6, NDCG@3 降序; LLM 行全覆盖)

| Family | Method | Events | NDCG@3 | Hit@1 | Mass@3 | JS(低好) | Latency ms/q | Tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Graph Router (本文) | LambdaMART {context+O_k+graph} | 2176 | 0.812 | 0.557 | 0.878 | 0.225 | ~0.002 | 0 |
| Ranking Algo | LambdaMART {context+O_k} | 2176 | 0.808-0.811 | 0.541 | 0.882 | 0.228 | ~0.002 | 0 |
| Ranking Algo | LambdaMART {context} | 2176 | 0.805 | 0.529 | 0.878 | 0.215 | ~0.002 | 0 |
| Ranking Algo | XGBoost {context+O_k} | 2176 | 0.807 | 0.537 | 0.876 | 0.227 | ~0.002 | 0 |
| Origin Role (线性) | OL-Origin (ridge, 无文本) | 2176 | 0.745 | 0.436 | 0.836 | 0.274 | 0.0016 | 0 |
| Text Encoder | BGE-base | 2176 | 0.732 | 0.449 | 0.819 | 0.274 | 0.45 | 51.6 |
| Text Encoder | BERT-base | 2176 | 0.731 | 0.446 | 0.828 | 0.280 | 0.64 | 51.6 |
| Text Encoder | Qwen3-Embedding-4B | 2176 | 0.729 | 0.444 | 0.819 | 0.285 | 11.86 | 59.3 |
| Text Encoder | FinBERT | 2176 | 0.722 | 0.448 | 0.807 | 0.280 | 0.53 | 51.6 |
| Text Encoder | E5-Mistral-7B | 2176 | 0.696 | 0.424 | 0.788 | 0.313 | 25.99 | 65.2 |
| Commercial LLM | DeepSeek V4 Flash (full, pointwise) | 2176 | 0.623 | 0.366 | 0.727 | 0.417 | 2688 | 206 |
| Local LLM | Qwen2.5-7B (full, pointwise) | 2174 | 0.555 | 0.257 | 0.685 | 0.483 | ~190 | ~330 |
| Local LLM | Llama3.1-8B (full, pointwise) | 2176 | 0.534 | 0.238 | 0.665 | 0.490 | ~172 | ~310 |

LLM 行已全覆盖 (重查 + 残留中性填补; deepseek 约 5.3% 候选填中性, 本地约 0%). 结构/GBDT 行 ~0.002 ms/q, 0 token, 纯 CPU.

---

## 2. 方法

### 2.1 数据与事件
findata KOL 推特档案, 17 标的. event=(symbol, UTC-day), 要求 >=8 KOL 参与; 每 KOL 取该事件首条推文按时间排序.

### 2.2 语义 frame 与候选
MiniLM(384维)在线贪心聚类, cos>=THR(0.50)并入否则新开 frame. 候选=first10. 目标=该 frame 24h 内后续 KOL 的 follower-weighted reach(log1p).

### 2.3 originator 标量 O_k
每事件按 ts 排序, 位置 i 的 net-lead=k+1-2*(i+1); 跨事件平均(min n>=4)对 median UTC hour 二次回归取残差=O_k. 注: 树下残差化非必需; O_k 单独对强 context 冗余.

### 2.4 lead-lag 图特征 (本文新增)
PIT 历史建有向图: 事件内每个有序对 早->晚 加边, 权重=共现次数. 特征: g_out/g_in(加权出入度 log1p), g_net(净度数=out-in), g_pr(反向图 PageRank), g_hub(HITS hub).
关键: 起作用的是 g_net(净领先度数); g_pr/g_hub(PageRank/HITS)无用甚至拖累. 图的价值=未塌缩的净领先度数, 而非网络层级.

### 2.5 排序器
LambdaMART(LightGBM, rank_xendcg, listwise), event 作 query group, 标签=reach 全局 32 分位整数等级. alpha/n_estimators 用 train 内 inner-CV(2025-01 切)选.
listwise 目标是主引擎(比 pointwise ridge +0.094 NDCG, 显著). LambdaMART 约等于 XGBoost 约等于神经 listwise/self-attention(~0.81 平台). 换排序器架构无意义.

### 2.6 PIT 与评估
O_k/history/图特征均按各 block cutoff 之前历史估, 严格 point-in-time. 指标 NDCG@3/Hit@1/Mass@3/JS, symbol-balanced. 显著性: symbol-balanced 双重 bootstrap, B=4000.

---

## 3. 消融与显著性 (pooled 5 窗 2021-2026, 9509 事件, reach)

| 对比 | ΔNDCG@3 | 显著 | 结论 |
|---|---|---|---|
| 树 listwise vs pointwise ridge | +0.094 | 显著 | 排序目标是主引擎 |
| {context+O_k+图} vs {context} | +0.0050 | 90%是/95%否 | 结构对 context 的增量, 边界 |
| 标量 O_k vs {context} | +0.0021 | 跨0 | 标量不够 |
| 关系特征 vs {context} | -0.0019 | 跨0 | 关系编码失败 |
| g_net vs {context+O_k} | +0.0050 | 边界 | g_net 是图增量驱动 |
| PageRank / HITS vs {context+O_k} | -0.003 / -0.005 | 跨0 | 网络层级无用 |
| **真图 vs 打乱图 (主真实性证据)** | **+0.0112** | **90% 且 95% 显著** | 增量来自真实拓扑, 非模型容量 |
| 真 O_k vs 打乱 O_k | +0.0058 | 90% 显著 | 标量身份也是真信号 |
| {full} vs {O_k only} | +0.24 | 显著 | context 必需 |

详见 ABLATION_SUMMARY.md (逐步搭建故事 + 三部分明细). 附: 换 adopt 目标图不再加分(reach-specific), 主线用 reach.

---

## 4. 诚实边界
1. 第一层 线性>文本 是主窗(最新一年)结论; cheap context 承担大头, O_k/g_net 是被验证为真的结构成分.
2. 第二层 图增量小且边界(+0.005, 90%是/95%否); 但 真图 vs 打乱图(+0.011)连 95% 也显著 — 结构真实性是硬的.
3. 图价值来自简单净度数 g_net, 非 PageRank/HITS; 别吹 network-topology 层级.
4. 残差化/手搓交互/花哨排序器/PageRank-HITS 均非必需, 最终模型极简.

---

## 5. 复现脚本 (experiments/socialenc/)
- phase98/99 — 图结构 pooled / 主窗主表
- phase100 — 95% CI + 图特征归因(g_net 驱动)
- phase103 — 真图 vs 打乱图 (结构真实性主证据)
- phase92/93/95/102 — 消融(排序器/特征轴, pooled/单窗)
- phase94 — 神经 LTR vs GBDT(平台)
- phase84 — 结构+surface+encoder 主表
- phase85/86 — full LLM (deepseek/本地, 全覆盖)
- phase101 — adopt 目标(附录)
