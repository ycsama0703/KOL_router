# Introduction & Related Work — Draft (weaving in experimental results)

> Draft prose for the paper's opening. Numbers are from the thr=0.50 main window (NDCG@3) and the pooled 2021-2026 ablation. Tune tone/venue later. Citations as [Author Year]; replace with bib keys.

---

## 1. Introduction (draft)

**P1 — The bottleneck.** Agentic trading and market-research systems increasingly ingest high-volume social-media streams, where a handful of key opinion leaders (KOLs) seed financial narratives that later diffuse. The operational bottleneck is *triage*: at the moment a narrative is freshly originated — before any popularity, follower confirmation, or adoption is observable — which narratives deserve scarce, expensive downstream reasoning (LLM calls, analyst attention)? Acting *after* a narrative trends is too late; the value has diffused. This is a point-in-time (PIT) early-ranking problem under tight latency and token budgets.

**P2 — Why existing tools fall short here.** Two dominant approaches are both ill-suited. (i) Content models — text encoders (BERT, FinBERT, E5, Qwen3-Embedding) and full LLMs — read the post text; they are expensive (tens of ms and tens of tokens per item for encoders; seconds and hundreds of tokens for LLMs) and, as we show, *not even the most accurate* on this task. (ii) Cascade / popularity-prediction models (DeepCas, DeepHawkes) require an *early-adopter observation window* and therefore cannot act at origination; the one zero-observation method (CasMS) still depends on message text and an adopter-propensity signal.

**P3 — Our discovery.** We show that financial KOL streams contain a stable, identity-specific **lead-lag originator structure**: certain accounts systematically *originate* narratives before others *adopt* them, and this structure is measurable point-in-time from pre-period history and is *deconfounded* against posting-hour/timezone. The structure is strong enough that a **simple linear model over it — reading zero text — already outranks every text encoder (NDCG@3 0.745 vs ≤0.732) and every full LLM (DeepSeek 0.623, local 7-8B 0.53-0.55)**, at ~0.002 ms and 0 tokens per item. Encoding the structure as a directed lead-lag graph and ranking with a listwise gradient-boosted tree (LambdaMART) lifts this to 0.812 and yields a statistically significant increment over a strong context baseline (+0.0072 NDCG, pooled, 90% CI). Critically, replacing the real lead-lag graph with an identity-shuffled one significantly degrades the ranker (−0.011 NDCG, 95% CI), confirming the gain comes from genuine origination-network topology rather than added model capacity.

**P4 — Contributions.**
1. **An originator-structure discovery + verification.** We define a deconfounded, PIT lead-lag originator trait (scalar O_k + graph net-degree g_net) for financial cashtag KOLs and verify it is a real, identity-specific signal via shuffle controls.
2. **A cheap structure router that beats text SOTA.** A linear/tree ranker over the structure (zero text) matches or beats SOTA text encoders and full LLMs on pre-popularity narrative-reach ranking at 2-6 orders of magnitude lower latency/cost.
3. **Rigorous ablations + prior-art comparison.** We attribute the result (listwise objective is the engine; g_net not network-centrality; structure significant only when graph-encoded) and show — by re-implementing them — that prior account signals (Romero influence, Yamada source-spreader, Zhou track-record) are either subsumed by context or add nothing, while our structure adds significantly.
4. **An application.** The router doubles as an early-breakout / event-prediction layer for agentic systems (a budgeted triage stage before expensive LLM reasoning).

**P5 — Roadmap.** Section 2 positions us against prior work; Section 3 defines the data, structure, and router; Section 4 reports the main table; Section 5 the ablations; Section 6 the application.

---

## 2. Related Work (draft, organized by the audit)

### 2.1 Early / pre-popularity cascade prediction
Cascade-size models such as **DeepCas [Li 2017]** and **DeepHawkes [Cao 2017]** predict future diffusion from an *observed early cascade graph* (random-walk paths or Hawkes intensities over the first hours of adopters). They are accurate but structurally require an observation window and thus cannot fire at origination. **CasMS [Zhou 2024]** is, to our knowledge, the only model with an explicit zero-observation "message-generation" stage; however it relies on message text (BERT) plus an adopter-side retweet-propensity signal over a static social graph. We differ on all of: **zero text**, an **originator (not adopter) lead-lag trait with posting-hour deconfounding**, a **narrative-frame unit with follower-weighted reach** (not single-message retweet count), and a structure-beats-LLM cost claim. We benchmark a CasMS-style generation-stage baseline. **[Weng 2013]** shows early popularity is a weak predictor and community structure of early adopters helps, but on general memes, requiring early adopters, with no per-account identity trait.

### 2.2 Content-only popularity prediction
**[Stokowiec 2017]** predicts popularity from title text alone with a BiLSTM and argues deep text beats shallow text by ~15%. Our result is the counter-thesis in this domain: a zero-text structural model beats deep text encoders and LLMs. We cite this line as the content-only school our structure router overturns.

### 2.3 Influencers, opinion leaders, and origination traits
A long line measures per-account influence: **[Cha 2010]** (the million-follower fallacy — follower count ≠ influence; influence is stable across topics/time), **[Romero 2011]** (influence-passivity IP score on a directed retweet graph, predictive of URL reach), and recently **[Yamada 2025]** (separating *stable* from *temporal* influencers via a source-spreader-vs-broker split, zero-text GBT). These establish that identity-specific, structurally-derived account signals are stable and predictive. We extend this in four ways that prior work does not combine: a **pairwise lead-lag (posting-time) origination** trait rather than self-amplification or eigenvector influence; **residualization against posting-hour/timezone**; a **fresh-narrative reach-ranking** target (not account persistence or URL clicks); and the **financial cashtag** domain. We further re-implement Romero's IP score, Yamada's source-spreader, and a Zhou-style track-record signal on our data and show (Section 5) that they are subsumed by simple context features or add nothing over them, whereas our structure adds a significant increment.

### 2.4 Financial social media
**[Hentschel 2014]** descriptively characterizes cashtag tweets; **[Rakowski 2021]** links aggregate cashtag volume to investor attention and prices via a Twitter-outage natural experiment; **[Li 2025]** predicts abrupt price changes from conversation-level similarity. The closest is **[Zhou 2025]**, which identifies StockTwits experts by *track-record accuracy* to predict *stock returns* and motivates a cheap non-LLM pipeline. We differ on the trait (origination/lead-lag, not accuracy), the target (narrative follower-weighted reach, not returns), the graph (account lead-lag, not stock-stock correlation), and — unlike Zhou, who hedges that the approach "complements, not competes" with LLMs — we run an explicit structure-vs-LLM head-to-head.

### 2.5 Primitives we build on
We detect freshly-originated frames using streaming first-story / novelty detection **[Petrovic 2010]** (here, online cosine clustering of MiniLM embeddings) and a frame representation following **[Qin 2017]**. These are detection primitives, not predictors; they carry no originator signal and are not competitors.

---

## 3. One-paragraph positioning (for the abstract / intro close)

> Prior work establishes (a) stable, structurally-derived per-account influence/origination traits [Cha 2010; Romero 2011; Yamada 2025], (b) expert identification on financial social media by track-record accuracy [Zhou 2025], (c) zero-observation popularity prediction from message text [Stokowiec 2017; CasMS 2024], and (d) that early popularity is a weak predictor relative to structure [Weng 2013]. We are the first to combine a **deconfounded, point-in-time lead-lag originator structure** over financial cashtag KOLs with a **zero-text lightweight router** that ranks freshly-originated narratives by future follower-weighted reach before popularity is observable — matching or beating SOTA text encoders and full LLMs at orders-of-magnitude lower cost, with the structural signal verified real by shuffle controls and shown to add significantly beyond both strong context features and re-implemented prior account signals.

---

## Notes / open items
- Replace [Author Year] with bib keys; confirm CasMS authorship (Zhou et al., IJCAI 2024) vs Zhou et al. 2025 finance (distinct Zhou).
- Run CasMS generation-stage baseline before claiming the head-to-head in 2.1.
- If venue is finance (not ML), foreground 2.4 and the application; if ML/SIGIR-style, foreground 2.1/2.3 and the efficiency frontier.
