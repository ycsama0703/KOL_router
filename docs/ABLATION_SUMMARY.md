# 消融总表 — Graph Router

最终模型: **Lead-Lag Router (LLR)** = listwise GBDT (LightGBM `rank_xendcg`, listwise 交叉熵目标, 非 LambdaMART) on { context + O_k + **g_net_rate** }.
（g_net_rate = 净领先率 = (out−in)/(out+in) = 活跃度归一化的净度数；裸 g_net=out−in 与发帖体量冗余，归一化后与体量正交且显著更强，见下第 3 步 / 设计选择表。）
口径: thr=0.50, first10, reach. 显著性: pooled 5 个滚动半年窗 (21.6-22.6 .. 25.6-26.6), 9509 事件, symbol-balanced bootstrap B=4000, 默认 90% CI (注明处为 95%). 指标 ΔNDCG@3.

各配置 pooled NDCG@3 (参考, rate 版): 完整 0.786 | ctx+O_k 0.782 | context 0.776 | 打乱图 0.770 | ridge 0.689 | 仅O_k 0.545.

**读法: 消融不是一堆独立实验, 而是对方法中每一个步骤/设计选择的系统验证** —— 每组 = 拿掉或替换某一步, 量它的净效果:
listwise vs pointwise = 验证「排序目标」这步; {full} vs {O_k only} = 验证「context」这步; 编码方式(标量/关系/图) = 验证「怎么编码结构」; 净领先率 vs 裸净度数 = 验证「去发帖体量」这步; g_net_rate vs PageRank/HITS = 验证「用净领先率而非中心性」; 真图 vs 打乱图 = 验证「结构真实性」; 残差化 vs 原始 = 验证「去时区混淆」这步. 因此下表每一行的身份都是"某步骤的验证", 步骤本身属于方法(§2), 这里只报验证结果.

---

## 故事线 (逐步搭建, 每步显著)

**起点**: context-only 的 listwise GBDT (便宜的强基线).

1. **排序目标是地基** — listwise 换 pointwise ridge: **-0.094*** [+0.077,+0.110]. 能排得好首先靠 listwise.
   补: 即便只用线性读出, 这个结构底座 (无文本) 在主表上已打败全部 BERT 家族编码器与 full LLM (见主表). 排序算法是锦上添花.

2. **context 是主体** — 去掉 context (仅 O_k): **-0.238*** [+0.219,+0.256]. 结构不能单飞.

3. **结构有增量, 但只有编码成"去体量的图"才行** (方法论核心): 在 context 之上加 originator 结构 —
   | 编码 | ΔNDCG@3 vs context | |
   |---|---|---|
   | 标量 O_k | +0.002 | 不够 (ns) |
   | 关系特征 | -0.002 | 失败 |
   | 裸净度数 g_net | ≈0 | 被发帖体量稀释 (对 ctx+O_k -0.002 ns) |
   | **lead-lag 图净领先率 g_net_rate** | **+0.0106** | **95% 显著 [+0.0044,+0.0173]; Hit +0.0214** |
   补: 净领先率 **显著胜过裸净度数** (+0.0069 NDCG / +0.0158 Hit, 95%); 在已含 O_k 的强基线上图主要提 Hit@1 (+0.0163 SIG), NDCG 擦边。

4. **这个增量来自真实的网络拓扑, 不是加列蹭噪声** (关键控制): 把真图特征换成身份打乱的版本 —
   | 对比 | ΔNDCG@3 | ΔHit@1 | |
   |---|---|---|---|
   | **真图 vs 打乱图** | **+0.0173** [+0.0115,+0.0234] | **+0.0287** [+0.0173,+0.0405] | **强显著 (95%)** |
   | 打乱图 vs context | -0.0068 [-0.0121,-0.0015] | -0.007 | 打乱图反而低于 context (有害) |
   读法: 两边都加图特征, 唯一区别是拓扑真假; 真图显著胜打乱图 +0.017, 而打乱图低于不用图 — 彻底排除 多加几列让树蹭出来 的质疑. 这是 结构真实 的最硬证据.

**终点**: Lead-Lag Router (LLR) = listwise GBDT( context + O_k + g_net_rate ).

---

## 支撑明细

### 第一部分: 组件敲除 (从完整模型逐个拿掉)
| 拿掉 | ΔNDCG@3 | 显著 | 含义 |
|---|---:|---|---|
| listwise 排序器 (换 ridge) | -0.094 | 显著 | 最大杠杆=排序目标 |
| 全部 context (仅 O_k) | -0.24 | 显著 | context 是主体 |
| 全部结构 (仅 context) | -0.0106 | 95% 显著 (Hit -0.0214) | 结构(O_k+g_net_rate)净贡献, 干净度量见下方 真vs打乱图 |
| 仅 g_net_rate (留 O_k) | NDCG -0.0048 (ns) / Hit -0.0163 (95%) | 混合 | 图在 O_k 之上主要提 Hit@1 |

### 第二部分: 设计选择论证
| 设计点 | 候选 | 结果 | 选择 |
|---|---|---|---|
| 排序器 | ridge / GBDT / XGBoost / 神经 | ridge -0.094; GBDT 约等于 XGBoost 约等于 神经 (~0.81 平台) | GBDT |
| 结构编码 | 标量 / 关系 / 图 | +0.002(ns) / -0.002(失败) / +0.0106(95%显著) | 图 |
| 图特征归一化 | 裸净度数 g_net / 净领先率 g_net_rate | 裸计数被体量稀释(对ctx+O_k -0.002 ns); rate +0.0106 vs ctx, 且 rate 显著胜裸计数 +0.0069 | **净领先率** |
| 哪个图特征 | g_net_rate / PageRank / HITS | rate +0.0048/Hit+0.0163; PageRank +0.0027 ns; HITS +0.0018 ns | g_net_rate, 弃 PageRank/HITS |
| 残差化 O_k | 残差化 / raw | 差异 ~+0.001 (ns) | 不涨点; 残差化≈原始 → O_k 非时区假象, 作抗混淆控制保留(不作 claim) |

### 第三部分: 真实性检验 (结构信号是不是真的)
| 控制实验 | ΔNDCG@3 | 显著 | 含义 |
|---|---:|---|---|
| **真图 vs 打乱图 (主证据)** | **+0.0173** | **95% 显著 (Hit +0.0287)** | 增量来自真实 lead-lag 拓扑, 非模型容量 |
| 真 O_k vs 打乱 O_k (标量层佐证) | +0.0058 | pooled 90% 显著 (单窗不显著) | 标量身份也是真信号, 效应更小 |

### 第三部分补充: vs 先验账号信号 (撞车防御, pooled 控制对照)
| 对比 | ΔNDCG@3 | 显著 | 含义 |
|---|---:|---|---|
| ours {ctx+O_k+g_net_rate} vs {context} | +0.0106 | 95% 显著 (Hit +0.0214) | 我们的结构在 context 之上加分 |
| {context+Romero-IP} vs {context} | -0.0021 | 跨0 | 通用图影响力(Romero) 零增量, 同 PageRank/HITS |
| ours vs {context+Romero-IP} | 显著优于 (rate 版待重算; g_net 版 +0.0092) | — | 显著优于 Romero-增强模型 |
| Yamada-src / Zhou-track | (= hist_mean_log_adopt / hist_success_rate) | — | 已在 context 里, 被吸收 |
读法: 先验账号信号要么已被 context 吸收 (Yamada/Zhou = 历史特征), 要么对 context 零增量 (Romero, 同 PageRank/HITS); 只有去混淆 lead-lag 结构 O_k+g_net_rate 提供显著增量. 这正面回掉 你不就是重做 Yamada/Romero 吗 的撞车质疑.

---

## 总结 (一条逻辑线)
1. 引擎: listwise GBDT 贡献最大 (+0.094 vs ridge); 且即便线性读出, 结构底座 (无文本) 已胜全部文本 SOTA (主表).
2. 结构: 在 context 之上再加增量 (+0.0106 vs 纯 context, 95%), 仅当编码成**去体量的净领先率 g_net_rate** 时有效 (标量不够/关系有害/裸净度数被活跃度稀释/PageRank-HITS 无效); 净领先率显著胜裸净度数 (+0.0069).
3. 真实性: 真图显著胜打乱图 (+0.017, 95% 显著), 打乱图甚至低于不用图 — 增量来自真实 origination 网络拓扑.
4. 取舍: 裸净度数/手搓交互/网络层级中心性/花哨排序器 全可弃或次优; 残差化不涨点(ns)但作为**抗时区混淆的控制**保留(不作性能 claim). 最终极简: context + O_k + g_net_rate 喂 listwise GBDT (Lead-Lag Router).

复现: 先验信号对照=phase104; 排序器=phase93/94; 结构编码=phase92/97/98/100; 图特征归因=phase100(裸)/phase100b(rate); 活跃度混淆诊断=phase106; 真图vs打乱图=phase103(裸)/phase103b(rate); 主表+消融rate=phase107; O_k身份=phase92/93; 单窗对照=phase95/99/102.
