# KOL 起源结构（origination axis）——发现、验证、数据、已死 payoff

**一句话**：在 16 年识别 KOL 推特面板上，存在一条**稳定、身份驱动、时区干净、正交于 follower 的"谁先发"(origination) 结构轴**。它被反复对抗验证证实**存在且真实**，但三种 payoff（收益 / 涨粉 / 级联）测下来**后果惰性**——不预测任何非平凡目标。本文沉淀该结构本身 + 用到的数据/方法/脚本，供在此之上发掘**新叙事**（结构是资产，已死的是某几个 payoff，不是结构）。
**最后更新**：2026-06-23。所有数字均本 session 在 luyao4 实测。

---

## 1. 结构的精确定义

- **事件**：`(symbol, UTC-day)`，且当天 ≥5 个不同 KOL 提及该 cashtag。
- **net-lead（每 KOL 每事件）**：事件内按首帖 `created_at` 升序排名 rank（1=最早），贡献 = `k+1-2*rank`（k=参与人数）。跨事件累加 → `s[kol]`，归一 `L_raw = s/n`（n=参与事件数，要求 n≥4）。
- **时区残差化（关键，必须做）**：把 `L_raw` 对 `[1, median_hour, median_hour²]` 回归取残差 = **origination 测度 `OLtrait`**；拟合值 = "时区/作息早晚" 成分 `SLtrait`。
  - 非参数等价版：按全局 median_hour 分十分位、组内对 L_raw 去均值——结论一致。
- **canonical 测度 = 时区残差化 net-lead**。**绝不能用 raw**（raw 有 59% 是时区，见 §3）。

---

## 2. 结构存在性（Phase 0 三闸，全过）

脚本 `phase0_origination.py`（置换检验 NPERM=1000）：
- **(a) lead-lag 层级显著**：16/17 标的置换 p=0.001（地板），仅 COIN 不显著(p=0.52)。
- **(b) 起源 ⊥ follower**：Spearman ρ = **−0.068**（641 KOL）。
- **(c) 非新闻反应伪影**：在"当日 |收益| 低于中位"的 price-quiet 子集上仍 p=0.001/标的，pooled 5959 事件；更严的**底四分位 |收益|** 静日：**17/17** p<0.01（脚本 `phase0_verify.py` E8，pooled 3179 事件）。

---

## 3. 对抗验证（试图杀它，没杀掉）

脚本 `phase0_verify.py` / `phase0_decisive.py` / `phase0_verify2.py`：

- **时区混淆（真实，已校正）**：`Spearman(L_raw, median UTC hour) = −0.768`（p~1e-126），每个 6-mo 箱的 r(L_raw,hour) 都在 −0.5~−0.7。**→ 必须残差化，否则卖的"信息轴"大半是时区轴。**
- **频率混淆**：`Spearman(OL, log#events)=+0.02`、`(OL, log#tweets)=+0.03` → 不是高频霸位。
- **bot/转发**：tweet_type 分布 `{original:374749, quote:43993, reply:40697, retweet:33}`；仅 original 重跑层级 **16/17** p<0.01。
- **事件定义稳健**：MIN_KOLS=3 → 16/17；=10 → 16/17。
- **持续性（stable-vs-lucky 核心，时区控制后）**：近期稠密期 6-mo 箱、lag-1 跨箱 pooled Spearman：
  - RAW（含时区）+0.669；**时区控制后 +0.447**（n=2104, p=6e-104）；shuffle 零分布 mean +0.001 / 95pct +0.037。lag-2 +0.345。
- **时区匹配零分布（决定性）**：把跨箱配对从"同一身份"换成"最近 median_hour 的不同人"——
  - 身份匹配 +0.42~0.47 vs **时区匹配零 +0.02~0.09**（gap≈+0.38），在 3 种时区控制（全局/逐箱中位小时、非参数十分位）+ 活动量双控制下都成立。**→ 持续性是身份驱动，不是时区。**
  - 逐对：17 个箱对，中位 +0.405，94% 为正（非单对异常）。
  - 箱长稳健：3-mo +0.440（n=3418, tz-null +0.003）、12-mo +0.390（tz-null +0.020）。
- **跨标的人特质**：groupA(9 syms) vs groupB(8 syms) 的 OL，Spearman = **+0.500**（p=5e-19, 279 KOL）。

**结论**：起源是**真实、稳定（lag-1≈0.42 / lag-2≈0.35）、身份驱动、跨标的一致（0.50）、正交于 follower（−0.07）、不被时区/活动/bot/新闻反应速度解释**的 KOL 特质。

---

## 4. 已死的 payoff（别重测，除非换 universe/分辨率）

- **收益（带符号延续 vs 反转）**，脚本 `phase1_infovsattn.py`：联合回归 `continuation ~ OL+FL+SL+VOL`（事件当日收益符号 × 前向收益，z within sym，train 估 trait/val 确认）。
  - val（n=2674）OL beta CI 全跨 0；**H 扫描 1/3/5/10/20 天，OL 的 bivariate rho 全在 ±0.026**。唯一存活：`VOL(事件规模)→反转 −0.04*`（文献已知的关注反转）。**起源不携带带符号收益信息。**
- **follower 增长（社会奖励）**，脚本 `phase1b_social.py` T1：控起始规模后 partial Spearman = **−0.064**（p=0.53, n=98 KOL）。起源不预测涨粉。
- **下游级联（次 1-2 日新 KOL）**，T2：控事件规模后 partial = **−0.023**（p=0.09, n=5340 事件）。raw −0.10 只是因 originators 带小事件；事件规模本身 +0.36 是平凡解释。

**→ 结构后果惰性：不预测收益 / 涨粉 / 扩散。**

---

## 5. 用到的数据（findata，全在 luyao4 自取，禁 ssh 传数据）

**KOL 推特**（护城河；端点见 `knowledge/FINDATA_CATALOG.md` / luyao4 用法见 `knowledge/LUYAO4_OPS.md`）：
- 端点：`/kols/tweets/by-symbol/{sym}/history`（`until` 时间戳分页，limit 500）。
- 每条字段：`tweet_id, created_at, kol_username, author_followers, author_verified, tweet_type, lang, text, cashtags`。
- 档案规模：33.8M 推文 / 3712 KOL / 2009–2026，cashtag 索引（`/kols/archive/stats`）。
- **已拉到 luyao4**：`experiments/socialenc/data/*.jsonl`，**459,472 推文 / 17 标的**：
  `AAPL MSFT NVDA TSLA AMZN META GOOGL AMD MSTR COIN HOOD PLTR SPY QQQ BTC ETH SOL`（mega-cap + crypto）。
- **密度近期偏重**（tweets/year）：2010:147, 2015:2270, 2018:7790, 2019:12422, 2020:40379, 2021:62597, 2022:71878, 2023:60771, 2024:75073, 2025:78213, 2026:35175。→ 2009–2017 太稀疏，跨16年两半分割无功效；用 **2017H2 起** 的稠密期分箱。

**价格 OHLC**：`/ohlc/{sym}`，**默认 1min（仅近125根）**；要日线必传 `interval=1d&start=2009-01-01&limit=20000`。返回 `{"symbol","interval","bars":[{ts,open,high,low,close,volume}]}`（注意 key 是 `bars`、字段 `ts`）。**crypto 需 `BTC-USD/ETH-USD/SOL-USD` 后缀**；equities 日线回到 2009（AAPL 4377 根）。

**MiniLM 句嵌入**：`experiments/socialenc/data/*.npz`（编码 gap 遗留，可复用）。

---

## 6. 脚本清单（this repository `experiments/socialenc/`）

| 脚本 | 作用 |
|---|---|
| `fetch_kol.py` | 拉 459K 推文/17标的 → data/*.jsonl |
| `phase0_origination.py` | 结构三闸 (a)(b)(c) |
| `phase0_verify.py` | 对抗 E1–E8（频率/时区/持续/跨标的/bot/事件定义/严static-c） |
| `phase0_decisive.py` | 时区控制后持续性（稠密期分箱 lag-1 + shuffle） |
| `phase0_verify2.py` | 时区匹配零分布再验证（身份 vs 时区近邻 + 多控制 + 箱长/lag） |
| `phase1_infovsattn.py` | 信息 vs 注意力/速度（收益延续回归 + H 扫描）→ 死 |
| `phase1b_social.py` | 起源→follower增长 / 下游级联 → 死 |

跑法：`python experiments/socialenc/<script>.py`（torch 不需要；纯 CPU/$0）。

---

## 7. 未死、可发掘新叙事的方向（供探索，均未测，附诚实风险）

结构是资产；死的只是上面三个 payoff。可能的新叙事（**别再撞"大盘低 SNR 收益预测"**）：

1. **内容几何刻画"为什么起源是信息空心的"**（接你偏爱的 LLM-嵌入当结构）：用推文嵌入几何刻画 originator vs amplifier 说的内容差异——起源者是否发得早但内容**泛/不定向**（解释零信息）？描述性，但 AI×fin、可证伪。风险：描述性"so what"。
2. **三轴正交本身当发现**（measurement/characterization 原型）：origination / followers / price-info 三条轴互相正交，画出 fintwit 影响力生态，**"price-info 轴几乎无人占据"这个零结果是卖点**。扩展 Cookson/Kakhbod。风险：纯实证、novelty 靠 framing。
3. **起源当方法学混淆项**：提醒别人做 KOL 信号时必须剥离时区+起源——方法 note。弱，最多 workshop。
4. **更细分辨率 / 更广 universe**：我们全是日级 UTC-day 事件 + 17 大盘/加密名。**真 intraday t0** 或**异质中小盘 universe**下，起源也许有后果——**未测**。风险：小盘在档案里稀疏（pump-probe 已死于数据），intraday 需重拉数据。
5. **起源稳定性的社会学解释**：它若纯是"作息/盯盘习惯"，可与 verified/lang/account-age 等关联做行为画像。离 finance 较远。

> 取舍提醒（来自本线教训）：任何新叙事先问"**它预测的那个目标，能否打赢笨基线、SNR 够不够**"，并**尽早跑最便宜的后果测试**——别再先验证结构、后找 payoff（本线 4 轮验证一个后果惰性结构的教训）。

---

## 8. 关联
- 数据/运维：`knowledge/FINDATA_CATALOG.md`, `knowledge/LUYAO4_OPS.md`
- 文献定位：`knowledge/SOCIAL_MEDIA_KOL_FRONTIER.md`（线2 Finfluencer / 线4 注意力扩散）
- brief：`briefs/2026-06-23-KOL-1-origination-network.md`（原始 gap + 预注册）
- 教训：拟入 `knowledge/FAILURE_PREMORTEM.md` #20「验证过的结构 ≠ 有后果的结构；后果测试应前置」
