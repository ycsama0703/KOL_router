# Theory — The Lead-Lag Origination Potential

> Theory section for the KOL origin-aware **graph** router. We first show **why the obvious
> solution (let an LLM read the firehose) breaks down**, which is what makes a cheap triage layer
> necessary; then one new object, two application-coupled results, a crisp departure from classical
> influence, and a short rigor appendix. Citations as [Author Year]; bib keys to be filled.

---

## 0. The problem — LLM triage does not scale to the social-media firehose

The naive solution to "which freshly-originated narratives deserve scarce downstream reasoning?" is
to hand the candidate stream to an LLM. **This gets *worse*, not better, as the data grows** — for
two independent reasons, which is precisely why a cheap pre-filter is *required*, not merely nice.

### P1 — Selection quality degrades as the pool size `K` grows (modern-Hopfield retrieval bound)

Selecting "which of `K` candidates to route" is an **associative-retrieval** operation, and the
transformer's own attention *is* modern-Hopfield retrieval [Ramsauer et al. 2021, *Hopfield Networks
is All You Need*, ICLR/arXiv:2008.02217]. With the `K` candidates as the stored patterns `X=[x_1,…,x_K]`
and the routing query `ξ`, one attention step is their **Eq. 3**

$$\xi^{\text{new}} = X\,\mathrm{softmax}(\beta X^{\top}\xi),\tag{1}$$

i.e. exactly transformer key–value attention. Define each candidate's **separation from its nearest
competitor** (their Eq. 5)

$$\Delta_i \;=\; x_i^{\top}x_i-\max_{j\neq i} x_i^{\top}x_j,\tag{2}$$

and `M=\max_i\lVert x_i\rVert`. Their **Theorem 5** bounds the retrieval error (Eq. 9)

$$\big\lVert x_i-x_i^{*}\big\rVert \;\le\; 2e\,(N-1)\,M\,\exp(-\beta\,\Delta_i),\qquad N=K=\text{pool size}.\tag{3}$$

**Two `K`-dependences make large-`K` triage fail:** *(i)* the prefactor is **linear in `(K-1)`**;
*(ii)* more candidates shrink the separation — `\max_{j\neq i}x_i^{\top}x_j` almost surely rises as
`K` grows, so `Δ_i\downarrow` and `\exp(-\beta Δ_i)` blows up **exponentially**. When no candidate is
well-separated, the update converges to a single global fixed point `≈` the **mean of all patterns**
(metastable averaging): the model returns a blurred average instead of selecting the top item.
Reliable retrieval is only guaranteed within the capacity budget (their Thm 3) `K\le N_{\max}\sim
c^{(d-1)/4}`. The **per-step mechanism** is softmax dilution: for logits bounded by `Δ=\lVert q\rVert
\lVert k\rVert`, the weights obey [Mudarisov 2025, arXiv:2508.17821, Cor. 1]

$$\frac1K e^{-2\Delta/T}\;\le\;\alpha_i\;\le\;\frac1K e^{2\Delta/T},\qquad H(\alpha)\le\log K,\tag{4}$$

so per-candidate attention mass is `O(1/K)` and the entropy ceiling `\log K` caps how sharply a head
can target the one good item — holding `K` selection mass constant requires growing logits like
`\log K`.

*Empirical confirmation (real LLMs, not just the bound).* "Lost in the middle": multi-document QA
accuracy is U-shaped in position and **drops by >20%** as the context lengthens, with the 20-/30-document
worst case falling **below the no-document baseline (56.1%)** [Liu et al. 2024, TACL]. Listwise LLM
ranking cannot even ingest a large pool: RankGPT **"cannot manage 100 passages at a time,"** needs a
**sliding window (size 20, step 10)**, and is **"highly sensitive to the initial passage order"**
[Sun et al. 2023, EMNLP]. **This is our dilution result:** the full 30-pool gives the *worst*
reach-capture (`full_k30` 0.455), while shortlisting to `b=10` lifts the *same* LLM to ≈0.64.

*Caveats (honest).* The Hopfield bounds assume the single-step softmax–Hopfield correspondence and
continuous patterns on a sphere; real heads are trained and multi-layer, so this is the **mechanism**
(associative retrieval has `K`-dependent error that collapses to averaging), not a literal accuracy
predictor. The `(K-1)` prefactor is worst-case; the operative driver is the `Δ_i` shrinkage.

### P2 — Compute/latency scales (super)linearly with `K` (rigorous)

Encoding a `K`-candidate pool costs, **per transformer layer**, self-attention `O(K^2 d)` and the
position-wise FFN `O(K d^2)` [Vaswani et al. 2017, Table 1: self-attention complexity `O(n^2\!\cdot d)`],
and the per-token forward FLOPs are `C\approx 2N_{\text{params}} + 2\,n_{\text{layer}} n_{\text{ctx}}
d_{\text{model}}` with the context term linear in `n_{\text{ctx}}=K` *per token* [Kaplan et al. 2020,
Eq. 2.2] — summed over the `K` tokens the attention contribution is again `\Theta(K^2)`, and KV-cache
memory is `O(Kd)` per layer. Hence a per-decision LLM cost

$$\tau_{\text{LLM}}(K)=\Theta\!\big(K^2 d + K d^2\big).\tag{5}$$

*Throughput consequence.* With freshly-originated frames arriving at rate `λ`, an **LLM-only** triage
is backlog-stable only if `λ\,\tau_{\text{LLM}}\le 1`; at firehose volume `λ\,\tau_{\text{LLM}}\gg 1`.
A cheap router scoring each frame at `\tau_R\ll\tau_{\text{LLM}}` and admitting only a fraction `p`
(the shortlist) restores stability iff `λ(\tau_R+p\,\tau_{\text{LLM}})\le 1`. Empirically
`\tau_{\text{LLM}}/\tau_R\approx 2390\,\text{ms}/0.05\,\text{ms}\approx 4.8\times10^{4}` and the router
uses `0` tokens.

**Consequence (the requirement the rest of the theory must meet).** Triage must be done by a
**cheap, zero-text score available *at origination*** — before popularity, before reading every
candidate with an LLM — that nonetheless ranks candidates by future reach well enough that the
expensive model only ever sees a high-value shortlist. §1 introduces such a score; §2 (Result A)
shows it is *cheap and structurally sufficient*; §3 (Result B) shows it *predicts reach at
origination*. The two LLM failure modes (P1 quality, P2 cost) map exactly onto the two things the
score must beat: be **more accurate** than letting the LLM sift everything, at **negligible cost**.

---

## 1. The solution, in one object

We introduce the **lead-lag origination potential**: for a set of accounts (KOLs), build a
directed *temporal-precedence* graph in which an edge `a→b` records how often `a` posts a
narrative **before** `b`; the potential of account `a` is the **net-degree**
`g_net(a) = out-deg(a) − in-deg(a)` of that graph (and `O_k`, its within-event scalar form).

**Claim of the paper.** This potential is a **point-in-time, zero-text *sufficient statistic*
for ranking freshly-originated narratives by their future follower-weighted reach — before any
popularity is observable.** Read with a cheap ranker and *no text*, it matches or beats text
encoders and LLMs at our triage task, at ~0.05 ms and 0 tokens.

The theory has exactly two jobs, one for each half of that claim:

| Application claim | Theory result |
|---|---|
| *Why cheap structure suffices — no text, no eigensolve* | **Result A:** net-degree **is** the lead-lag potential, the sufficient structural ranking statistic (and PageRank/HITS are not). |
| *Why it predicts reach **at origination** (triage before popularity)* | **Result B:** an originator's precedence position sets the expected reach of the narrative it seeds. |

Everything else (deconfounding, point-in-time stability, ranking-loss consistency) is supporting
rigor, collected in §4.

---

## 1.1 What is genuinely new — and why this is **not** classical opinion leadership

Seventy years of influence research — two-step flow [Katz & Lazarsfeld 1955], the
million-follower fallacy [Cha 2010], influence-passivity [Romero 2011], stable-vs-temporal
influencers [Yamada 2025] — measures **who gets amplified**: a *person*-level trait read from
*received* engagement (retweets/mentions/reach received), used to rank *people* by *track record*.

Our object is the opposite end of the diffusion arrow, used at a different moment, for a different
unit:

| | Classical influence / opinion leadership | **Lead-lag origination potential (ours)** |
|---|---|---|
| **Signal** | *received* amplification (who is retweeted) | *originated* **timing** — who posts **first** (who-leads-whom) |
| **Unit & moment** | a **person**, scored on **past track record** | a **freshly-originated narrative**, scored **at origination** |
| **Observability** | needs realized engagement | **zero popularity observed**; zero text |
| **Use** | descriptive influence ranking | **real-time triage** for downstream compute allocation |

So classical influence is the literature we **depart from**, not a predecessor that subsumes us.
(This is also why we do not collide with Yamada's source-spreader, whose score is *retweets
received*: `O_k`/`g_net` are **posting-time precedence**, never amplification counts.) The
novelty is the *combination*: a **temporal-precedence potential**, deployed as a **zero-text,
point-in-time triage signal** that ranks **narratives at origination** by future reach.

---

## 2. Result A — Net-degree *is* the lead-lag potential (why cheap structure suffices — answers P1/P2)

**Setup.** From point-in-time history form the **skew-symmetric precedence flow** (`w(a\!\to\! b)` =
#events where `a` precedes `b`), and the **net-degree** as its row-sum / divergence:

$$Y_{ab}=w(a\!\to\! b)-w(b\!\to\! a)=-Y_{ba},\tag{6}$$

$$g_{\text{net}}(a)=\deg^{out}(a)-\deg^{in}(a)=\sum_b Y_{ab}.\tag{7}$$

**A.1 — Net-degree is the divergence of the precedence flow, and divergence is the consistent
global ranking.** By the combinatorial Helmholtz–Hodge decomposition [Jiang, Lim, Yao & Ye 2011],
any pairwise-comparison flow splits orthogonally into a **gradient** part (a globally-consistent
ranking induced by a potential `s`), plus **curl/harmonic** parts (cyclic inconsistency). The
consistent ranking is the potential solving the **Laplacian normal equation**

$$\Delta_0\,s=-\operatorname{div}Y,\qquad \operatorname{div}(Y)(a)=\sum_b Y_{ab}=g_{\text{net}}(a).\tag{8}$$

Its right-hand side is **exactly** the net-degree (7). On a complete, uniformly-weighted graph the
potential *equals* the net-degree (their eq. 29) — recovering **Borda count** and the **Massey
least-squares rating**.

**A.2 — Direct optimality (the load-bearing statement, no completeness needed).** Independently of
the Hodge geometry, ranking by the **row-sum `g_net` is risk-optimal among all
permutation-invariant ranking procedures, for all reasonable losses, under a Bradley–Terry /
stochastic-ordering model** [Huber 1963; cf. Shah et al. 2017, minimax-optimal counting estimators].
So a *zero-cost* statistic — out-minus-in degree — is the right structural summary of a precedence
flow. **This is the theoretical license for "no text, no eigensolve, O(1) per node."**

**A.3 — Why PageRank / HITS are the wrong tool (explains the ablation).** PageRank, HITS and
eigenvector centrality compute a **different operator** — the dominant eigenvector / stationary
distribution of a non-negative matrix, i.e. *recursive endorsement mass* — which is **not an
estimator of the precedence potential** `s = −Δ₀⁺ div Y`. (We do not claim they ignore edge
direction; we claim the quantity they compute is not the potential.) Hence, once `g_net` is
present, **PageRank/HITS add nothing** — our `g_pr`, `g_hub` null increments. And the **shuffle
control** (permuting which identity carries which graph feature) breaks the *identity↔potential*
correspondence, collapsing the gain — our real-vs-shuffled `+0.011` NDCG (95%): the signal is the
genuine potential, not model capacity or the marginal `g_net` distribution.

**Honest scope.** Our graph is incomplete and weighted, so strictly `g_net = div Y` (the RHS); the
exact potential is the Laplacian-smoothed solve `−Δ₀⁺ div Y` — `g_net` is its leading term (a
*testable* claim: report `g_net` vs the full solve, and the Hodge residual `‖R*‖` as a
flow-consistency certificate). The "PageRank adds nothing" claim is specific to a reach target that
is a precedence quantity. Robustness fallback when `g_net` saturates under heavy noise: SerialRank
[Fogel et al. 2016].

---

## 3. Result B — Originator precedence sets expected reach (why it predicts at origination)

**Claim.** The earlier (more upstream) the originator, the larger the expected reach of the
narrative it seeds — so a precedence score predicts reach *before* popularity, with no text.

**The mechanism, assembled from three exact pieces.**
1. **Origination captures amplification.** In a self-exciting (Hawkes) diffusion with branching
   ratio `n* < 1`, one seed's expected cluster size is `1/(1−n*)` [Hawkes 1971; Rizoiu et al. 2017].
   The **originator is the immigrant** of the cluster and so claims its entire downstream
   amplification; later participants are descendants already counted. (*Exact in expectation.*)
2. **Reach is monotone in upstream position.** Under Independent-Cascade / Linear-Threshold, the
   single-seed expected reach equals the expected reachable-set size in a random live-edge subgraph
   [Kempe, Kleinberg & Tardos 2003] — **monotone over the diffusion DAG's reachability order**.
3. **Upstream position is net-degree.** The leader↔lagger ranking of timing data is exactly the
   net-degree of the precedence graph [Bennett, Cucuringu & Reinert 2022, leader score
   `L(i) ∝ Σ_m (A_im − A_mi)` ≡ our `g_net`] — i.e. Result A.

**Status — honest.** This composition is a **well-motivated hypothesis, not a single theorem**:
there is no clean result "`E[reach]` monotone in `g_net`". Per-item virality is intrinsically
high-variance [Bakshy et al. 2011], and early *raw popularity* is a weak predictor while early
*structure* is strong [Weng et al. 2013] — which is **why we predict from origination structure,
not observed counts, and claim *ranking lift*, not point prediction** (hence NDCG, not R²).
Follower-weighting (not event-count weighting) needs a marked Hawkes kernel [SEISMIC, Zhao et al.
2015] — a modeling extension we name, not assume.

**B in one line.**

$$\mathbb E[\text{reach}\mid\text{originator }k]\ \approx\ \underbrace{(\text{origination magnitude})}_{\text{marked size}}\times\underbrace{\big(\text{amplification}\uparrow\text{ in }k\text{'s upstream precedence}\big)}_{\text{via (1)–(8): }g_{\text{net}}}.\tag{9}$$

Ranking by `g_net`/`O_k` is therefore *expected* to order narratives by future reach — a hypothesis
our experiments confirm.

---

## 4. Supporting rigor (appendix-level; the method stays honest)

These three under-write the estimator but are not the headline.

**(R1) `O_k` is deconfounded against timezone.** `O_k` = raw net-lead **residualized on a quadratic
in median UTC posting-hour**. By Frisch–Waugh–Lovell [1933/1963] this is *numerically* the
posting-hour-partialled net-lead (orthogonal to the hour basis); under a Robinson partially-linear /
double-ML model [Robinson 1988; Chernozhukov et al. 2018] it *identifies* the structural lead net of
timezone **if** posting-hour is the sole confounder and is correctly modeled. *Caveat:* FWL is
algebraic not causal, and hour-of-day is circular — report a sin/cos (cyclical) robustness check.

**(R2) The potential is point-in-time stable enough to apply forward.** We estimate the trait on
pre-cutoff history and rank the next period; this needs influence-role autocorrelation, which holds
empirically at our regime: cross-topic ρ ≈ 0.5–0.68 and marginal 8-month drift [Cha 2010];
network-only PIT features predict role-stability at AUC ≈ 0.89, with 74% six-month persistence
[Yamada 2025]. *Caveat:* stability is for coarse, head-account, short-horizon ranking — not general
stationarity [Pena et al. 2025].

**(R3) The listwise objective is NDCG-consistent.** Our ranker is LightGBM with objective
**`rank_xendcg`** — the **XE-NDCG cross-entropy listwise loss** [Bruch et al. 2021], a
Plackett–Luce/softmax-family objective (**not** LambdaMART; prior "LambdaMART" labels are a
misnomer). Minimizing it is NDCG-consistent — convex NDCG-consistent surrogates exist as Bregman
divergences against the normalized gain with √-rate regret transfer [Ravikumar, Tewari & Yang 2011],
and softmax cross-entropy bounds log-NDCG [Bruch et al. 2019]. The convex-calibration impossibility
of [Calauzènes et al. 2012] is for MAP/ERR, **not** NDCG — we deliberately target NDCG. Because the
loss we actually minimize is XE-NDCG, this consistency attaches *directly* to our model.

---

## 5. How the theory matches the experiments

| Empirical result | Explained by |
|---|---|
| **LLM reach-capture is *worst* on the full 30-pool (`full_k30` 0.455); a cheap shortlist lifts the same LLM to ~0.64** | §0 P1 (listwise dilution / lost-in-the-middle) |
| **LLM routing ~2390 ms / ~5.8k tok vs structure ~0.05 ms / 0 tok** | §0 P2 (LLM cost scales with volume) |
| `g_net` adds a significant increment; scalar-only / relational encodings do not | A.1–A.2 (net-degree = sufficient potential) |
| **PageRank/HITS add ≈ 0 over `g_net`** | A.3 (eigenvector centrality ≠ potential) |
| **Real vs shuffled graph: +0.011 NDCG, 95% sig** | A.3 (shuffle breaks identity↔potential) |
| Zero-text structure ranks reach ≥ text encoders / LLMs, at ~0.05 ms | A.2 (cheap row-sum is the right statistic) + B |
| Triage works *before* popularity is observable | B (precedence ⇒ expected reach at origination) |
| `O_k` residualized > raw `O_k` | R1 (FWL deconfounding) |
| PIT train-past / test-next-year holds across rolling windows | R2 (influence-role autocorrelation) |
| Listwise > pointwise (+0.094); ranker plateaus across GBDT/XGB/NN | R3 (XE-NDCG NDCG-consistency) |
| Per-frame reach noisy; we win on *ranking*, not point prediction | B caveat (rank lift, not point pred) |

---

## 6. Must-cite bibliography

**Problem / §0 (why LLM triage doesn't scale).** *P1 (quality):* Ramsauer et al. 2021 (*Hopfield
Networks is All You Need*, ICLR; arXiv:2008.02217 — Eq. 3, 5, Thm 3/5 retrieval-error bound);
Mudarisov 2025 (*Limitations of Normalization in Attention*, arXiv:2508.17821 — Cor. 1, `O(1/K)`
dilution); Liu et al. 2024 (*Lost in the Middle*, TACL; arXiv:2307.03172); Sun et al. 2023 (*RankGPT*,
EMNLP; arXiv:2304.09542). *P2 (cost):* Vaswani et al. 2017 (*Attention Is All You Need*, NeurIPS;
arXiv:1706.03762, Table 1 `O(n²d)`); Kaplan et al. 2020 (*Scaling Laws*, arXiv:2001.08361, Eq. 2.2).
**Result A (encoding).** Jiang, Lim, Yao, Ye 2011 (arXiv:0811.1067); Huber 1963 (Ann. Math. Stat.
34); Shah et al. 2017 (JMLR 18:199); Massey 1997 (arXiv:1701.03363); Fogel, d'Aspremont, Vojnovic
2016 (SerialRank, JMLR 17); Kleinberg 1999 (HITS); Brin–Page 1998 (PageRank); Bonacich 1972.
**Result B (mechanism).** Kempe, Kleinberg, Tardos 2003/2015; Hawkes 1971; Rizoiu et al. 2017
(arXiv:1708.06401); Bennett, Cucuringu, Reinert 2022 (arXiv:2201.08283); Weng, Menczer, Ahn 2013
(Sci. Rep. 3:2522); Bakshy et al. 2011 (WSDM); Zhao et al. 2015 (SEISMIC, KDD).
**Departure / related (§0.1).** Katz & Lazarsfeld 1955; Cha et al. 2010 (ICWSM); Romero et al. 2011
(WWW/ECML); Yamada, Tsugawa, Yoshida 2025 (arXiv:2512.17166); Pena et al. 2025 (PLoS ONE).
**Rigor appendix.** Frisch–Waugh 1933 / Lovell 1963; Robinson 1988; Chernozhukov et al. 2018
(arXiv:1608.00060); Ravikumar, Tewari, Yang 2011 (AISTATS); Bruch et al. 2019 (ICTIR) & 2021
(XE-NDCG, WWW); Calauzènes, Usunier, Gallinari 2012 (NIPS).

---

*Provenance: theorem statements verified against primary sources (HodgeRank eq. 29/34; KKT
live-edge; Huber 1963 row-sum optimality; Bennett 2022 leader score; Ravikumar Thms 6/9/10; Bruch
Prop. 3; FWL/Robinson/DML; Cha/Romero/Yamada primary PDFs).*
