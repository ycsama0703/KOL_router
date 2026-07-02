# Novelty / Prior-Art Collision Audit — KOL Origin-Aware Graph Router

**Method.** Closest prior work in 7 areas was read in **full text** (methods, problem formulation, features, targets — not abstracts) and compared point-by-point against our five claims. Where feasible, the prior-art account signals were **re-implemented and benchmarked on our data** (phase104) rather than only argued. Some publisher PDFs returned 403 and rest on abstract + corroborating sources (flagged at the end).

## Our claims (collision targets)

- **C1 (discovery).** In financial KOL (Twitter/X cashtag) streams there is a stable, identity-specific **lead-lag originator structure** — certain accounts systematically *originate* narratives before others *adopt/repeat* them — measurable point-in-time (PIT) from pre-period history.
- **C2 (encoding).** A **deconfounded scalar trait O_k** (per-account net-lead over events, residualized against posting-hour/timezone) **+ a lead-lag directed-graph net-degree g_net** (out − in).
- **C3 (router).** A lightweight model (linear ridge / listwise cross-entropy GBDT, the Lead-Lag Router) over this structure **ranks freshly-originated narrative frames by future follower-weighted reach, before any popularity is observable, using zero text.**
- **C4 (result).** Structure (no text) matches/beats SOTA text encoders and full LLMs at orders-of-magnitude lower latency/cost.
- **C5 (application).** Early breakout / event prediction (flag high-reach narratives before popularity).

## Verdict summary

| Paper | Area | Closest to | Verdict |
|---|---|---|---|
| **CasMS** (Zhou et al., IJCAI 2024) | cascade/early popularity | C3, C5 (task slot) | **Live competitor — must benchmark** |
| **Yamada et al. 2025** (2512.17166) | influencer trait | C1, C2 | Adjacent — cite & distinguish (closest on framing) |
| **Zhou et al. 2025** (2504.10078) | financial social media | C4, finance-account framing | Adjacent — cite & distinguish (closest in finance) |
| Romero et al. 2011 (IP) | influence | C2 (directed-graph influence) | Primitive + **empirically tested (no increment)** |
| Yamada source-spreader / Zhou track-record | account trait | C2 | **Empirically == our history features (subsumed)** |
| Stokowiec et al. 2017 | content-only popularity | C4 (foil) | Adjacent — cite as opposing thesis |
| DeepCas 2017 / DeepHawkes 2017 | cascade | C3 | Adjacent — the observation-window school we remove |
| Weng et al. 2013 | virality + community | C3, C4 | Adjacent — cite & distinguish (non-finance, needs early adopters) |
| Cha et al. 2010 | influence stability | C1 (stability) | Primitive (also == our follower baseline) |
| Petrovic et al. 2010 (FSD) | first-story detection | frame detection | Primitive — cite/reuse |
| FrED / Qin et al. 2017-18 | event/frame detection (SVO) | none | NOT used — we do MiniLM-embedding cosine clustering, not SVO parsing; do not cite as our primitive |
| Rakowski 2021 / Hentschel 2014 / Li 2025 | financial social media | none | No collision |

## 1. Live competitor (must benchmark): CasMS (IJCAI 2024)

The **only** prior method that predicts popularity at the **message-generation (zero-observation) stage** — same task slot as C3/C5. Quote: *"In the message generation stage, the corresponding cascade set C_i is empty… relies solely on the content of a message, the static social graph, and the cascade model."*

**Why it is not a true collision (four deltas):**
1. **Not zero-text** — a BERT/RoBERTa message embedding is a core input even at generation stage; we use **zero text**.
2. **Different structure signal** — CasMS uses an *adopter-side* retweet-propensity scalar s_{v,i} + node2vec on a static social graph; **not** a lead-lag originator trait, **no** posting-hour deconfounding.
3. **Different unit/target** — single-message *retweet count*; we use a **narrative frame** with **follower-weighted reach**.
4. **Cost claim** — CasMS adds text; we claim and show structure-without-text beats encoders/LLMs.

**Benchmarked (phase105, best-effort).** We adapted CasMS's two zero-observation inputs — message embedding (Qwen3-4B) + originator node2vec graph-position (PIT co-occurrence graph; no follow graph available) — into the same listwise cross-entropy GBDT on our frame-reach task. Result (main window 25.6-26.6): **CasMS-style 0.695 NDCG / 0.429 Hit**, *below* text-only (Qwen 0.719; node2vec dilutes) and far below **our 0.813 (+0.118 NDCG, +0.123 Hit)**. Node2vec graph-position alone is weak (0.494). Caveat: not the full CasMS arch (no GCN / personalized-retweet module / follow graph), but its winning ingredient is text (benchmarked: Qwen 0.719, we beat) and its graph signal is shown weak (0.494) — so a fuller CasMS cannot escape that both of its zero-obs input families lose to our deconfounded lead-lag structure. (No public code.)

## 2. Empirically tested prior-art account signals (phase104) — and beaten

We re-implemented the three prior-art account signals and compared under an identical listwise cross-entropy GBDT (pooled 2021–2026, 9509 events):

- **Romero IP-influence** (directed-graph influence fixed point on our lead-lag graph): **adds nothing over context** (ΔNDCG −0.0021, ns) — same as generic PageRank/HITS centrality.
- **Yamada source-spreader** ≈ our `hist_mean_log_adopt`; **Zhou track-record** ≈ our `hist_success_rate` — both are **already in the context baseline** (subsumed).
- **Our {context + O_k + g_net}** adds **+0.0072 NDCG / +0.0135 Hit over context (90% sig)** and **+0.0092 over context+Romero (sig)**.

**Net:** prior-art account signals are either subsumed by context (Yamada/Zhou) or add nothing (Romero, like PageRank/HITS); **only the deconfounded lead-lag originator structure provides a significant increment.** This is the empirical rebuttal to "you re-did Yamada/Romero."

## 3. Adjacent — cite & distinguish

**Yamada et al. 2025 — closest on C1/C2.** Source-spreader vs broker split, stable per-account influence, zero-text GBT. Distinguish: (i) target is **account persistence** (top-10% for 6 months), not **fresh-frame future reach**; (ii) their origination = **self-amplification** (retweets received), ours = **pairwise lead-lag posting-time** order; (iii) **no deconfounding**; (iv) non-finance.
- **Internal check (load-bearing):** our O_k is computed from within-event **posting order** (rank by timestamp; net_lead = k+1−2·rank in `compute_oltrait`), i.e., genuine lead-lag **timing**, NOT retweets-received. This is what keeps us off Yamada. The paper must state this explicitly; if O_k were amplification counts, C1/C2 would collide.

**Zhou et al. 2025 — closest in finance.** Identifies experts by **track-record accuracy** → predicts **stock returns**; motivates a cheap non-LLM pipeline but **hedges** ("complements, not competes") and runs **no structure-vs-LLM head-to-head**. Distinguish: trait = origination (not accuracy); target = reach (not returns); graph = account lead-lag (not stock-stock); and **we run the head-to-head LLM benchmark they avoided** (done: DeepSeek/encoders, full coverage).

**Stokowiec et al. 2017 — content-only popularity foil.** Title-only BiLSTM, claims **deep text > shallow text**; our C4 (structure no-text > text) is the direct counter-thesis. Cite as the school we overturn.

**DeepCas / DeepHawkes 2017 — the observation-window school.** Both **require an early-adopter observation window**; inapplicable at origination. Cite as the dependency we remove.

**Weng et al. 2013 — virality + community structure.** "Early popularity is a weak predictor; structure beats it" — conceptual ancestor of C3/C4, but **general memes (non-finance), needs early adopters, no per-account identity trait, no frame-reach target.** Cite & distinguish on finance + identity-stable originator + frame level.

## 4. Primitives — cite/reuse, no collision

- **Petrovic et al. 2010 (Streaming First Story Detection)** — the fresh-frame / first-story novelty-detection primitive (our MiniLM clustering plays this role). Also reports "#users > volume" — supportive intuition for reach.
- **FrED / Qin et al. 2017-18** — SVO (subject-verb-object) semantic-frame representation + frame clustering. **We do NOT use this**: code (phase5/phase7) confirms our frame construction is MiniLM-embedding online cosine clustering (L2-normalised centroids), with no SVO / dependency / semantic-frame parsing. Listed only to disclaim collision; not cited as our primitive.
- **Cha et al. 2010 (Million-Follower Fallacy)** — foundational evidence that identity-specific influence is **stable across topics/time** (supports C1 stability); also == our follower-count baseline.
- **Romero et al. 2011** — directed-retweet-graph influence predicts content reach; the structural-influence-predicts-reach precedent (we beat its IP score empirically, see §2).

## 5. Most dangerous overlaps + defenses (summary)

| Overlap | Paper | Defense |
|---|---|---|
| C1/C2 framing | Yamada 2025 | lead-lag *timing* (not amplification) + net-degree + posting-hour residualization; fresh-frame reach target (not account persistence); finance. Empirically beats source-spreader (§2). |
| C3/C5 task slot | CasMS 2024 | zero-text; originator (not adopter-propensity); frame-reach (not message retweet count); deconfounding. Benchmark it. |
| C4 + finance-account | Zhou 2025 | origination (not accuracy); reach (not returns); we run the LLM head-to-head they avoided. |

## Bottom line

The specific combination — **zero-observation × zero-text × lead-lag originator structure (O_k, g_net) × posting-hour deconfounding × finance cashtag × follower-weighted frame-reach** — is **not occupied** by any prior work. Novelty survives. It lives in a narrow gap immediately adjacent to **CasMS** (benchmark it) and **Yamada** (distinguish on timing/deconfound/target), and is **empirically the only account signal that adds over a strong context baseline already containing prior-art traits** (phase104).

## Coverage gaps (honesty)

Full PDFs of **Rakowski 2021, Li 2025 (DSS), Weng 2013 (Nature), DeepHawkes 2017** were 403-blocked; their characterizations rest on abstracts + cross-corroboration (CasMS related-work, GitHub data specs, institutional summaries). The conclusions (no originator lead-lag trait; observation-window dependence) are robust but a library full-text pass on the DSS paper (to confirm its "social network controls" contain no account lead-lag term) would be belt-and-suspenders.

## Must-cite shortlist (for related work)
Yamada 2025; Zhou 2025; CasMS 2024; Stokowiec 2017; Romero 2011; Cha 2010; DeepCas 2017 / DeepHawkes 2017; Weng 2013; Petrovic 2010; Rakowski 2021; Hentschel 2014. (FrED/Qin 2017-18 disclaimed — not used; see above.)
