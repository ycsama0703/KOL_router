# Theoretical Framing (V2): From LLM Selection Inefficiency To KOL-Origin Routing

This is the revised theoretical framing. It is self-contained. Formulas contain
only placeholders; empirical numbers are substituted into the final derived
results to strengthen the conclusions, never into the derivations themselves.

## 1. Setup

For each event-day `e`, the agent observes a pool of social-media narrative
origin candidates:

```text
C_e = {i = 1, ..., K_e}
```

Each candidate `i` has an unobserved future value:

```text
Y_i = future follower-weighted reach of candidate i
```

The agent can send only a small subset to expensive downstream reasoning:

```text
A_e subset C_e,    |A_e| = r
```

The ideal decision is an attention-allocation problem, not a return-prediction
problem:

```text
max_{A_e}  E[ sum_{i in A_e} Y_i | I_e ] - lambda * Cost(A_e)   s.t. |A_e| = r
```

where `I_e` is the point-in-time information set available before later
follower confirmation is observable. The router is valuable if it improves
which items enter memory, retrieval, LLM reasoning, or strategy research.

## 2. Full-Context LLM As A Noisy Selector

A full-context LLM reads all `K` candidates and selects `r`. We model its
selection as a noisy utility estimate:

```text
tilde Y_i = Y_i + epsilon_i(K)
```

where `epsilon_i(K)` is effective selection error (bias and variance
unseparated) induced by context length, distractors, and position effects. The
argument needs one property of `epsilon`, established next.

## 3. The Error-to-Capture Link

Under a Plackett-Luce noisy selector the selection probability is:

```text
P_sel(i | C) = exp(Y_i / tau) / sum_{j in C} exp(Y_j / tau)
```

where `tau` scales effective error (larger error = larger `tau`; Luce 1959;
Plackett 1975). Let `Capture(K, tau)` be the expected fraction of oracle top-r
reach captured by the selector. In the standard utility-ranking case:

```text
larger tau flattens selection probabilities toward uniform choice
Capture(K, tau -> 0)   ->  1          (near-oracle selection)
Capture(K, tau -> inf) ->  r / K      (near-random selection)
```

So rising effective error compresses expected capture toward the random
baseline `r / K`. This is a modeling bridge, not a claim that the real LLM
literally samples from a PL distribution or that all possible utility
configurations satisfy a strict monotonic theorem. It formalizes the intuition
that noisier listwise selection captures less of the oracle top-r future reach.

## 4. LLM Effective Error Need Not Decrease With K

We use the following weaker and falsifiable condition:

```text
tau_LLM(K) is not guaranteed to decrease with K.
```

This is anchored in three lines of LLM evidence on listwise / multi-document
behavior: long-context degradation (Liu et al. 2023; Levy et al. 2024),
distractor scaling with length controlled (Modarressi et al. 2025; Hsieh et al.
2024), and listwise position bias that worsens with list length (Wang et al.
2023; Qiao et al. 2026; Sun et al. 2023). Our setting is listwise selection
over comparable candidate narratives, not retriever-filtered RAG where position
bias can be marginal (Cuconasu et al. 2025), so these channels apply directly.
We model end-to-end listwise difficulty through effective error and do not
separate its bias and variance. (A human choice-overload analogue exists —
Iyengar & Lepper 2000; Chernev et al. 2015 — but is not used as evidence for
LLMs.)

The consequence we use is not that the LLM becomes worse in absolute terms as
K grows, but that it need not convert additional candidates into
proportionally more usable selection. If the effective error fails to fall
enough as K grows, the marginal information from a larger pool can be absorbed
by selection complexity, so the marginal capture gain per additional candidate
diminishes. The claim is about utilization, not universal degradation: more
candidates need not help proportionally, and part of the additional data can be
wasted.

## 5. Empirical Discovery: A K-Weakly-Dependent Originator Structure

Section 4 implies a useful first-stage selector should avoid the LLM-specific
error channels that arise from long prompts, distractors, and listwise position
effects. We need a K-weakly-dependent signal that is also real, stable,
point-in-time, and not a trivial proxy. We summarize an empirical
discovery in large-scale financial KOL data that provides such a signal. The
full evidence is in `docs/KOL_ORIGINATOR_STRUCTURE_DISCOVERY.md`; here we state
the premise the router requires, with the measurement and tests formalized.

### 5.1 Representation And Measurement

Each candidate is actor-time-frame data, not a bare text string:

```text
i = (text_i, frame_i, time_i, actor_i)
```

For each event `e = (symbol, UTC day)` with `k_e` participating KOLs, order
KOLs by the timestamp of their first post in `e` (`rank = 1` is earliest). The
net-lead contribution and raw lead score are:

```text
net_lead_{e,k} = k_e + 1 - 2 * rank_{e,k}
L_raw(k) = (1 / |E_k|) * sum_{e in E_k} net_lead_{e,k},    |E_k| >= 4
```

where `E_k` is the set of events KOL `k` joined. `L_raw(k) > 0` means `k`
originates earlier than average.

### 5.2 Residualization (Frisch-Waugh-Lovell)

Raw lead is confounded by posting schedule. Regress out the median UTC posting
hour `h_k` and its square, and keep the residual as the canonical trait:

```text
L_raw(k) = a + b1 * h_k + b2 * h_k^2 + O_k
O_k = L_raw(k) - Proj_{[1, h, h^2]} L_raw(k)
```

`O_k` is the identity-driven originator role with the schedule component removed
(FWL residual; Frisch-Waugh 1933; Lovell 1963).

### 5.3 Existence Test (Permutation)

Whether a lead-lag hierarchy exists is tested against a permutation null that
reassigns the lead multiset within each event. Define the hierarchy statistic
and p-value:

```text
T(S) = sum_{k in S} O_k^2 / n_k
p_e = P_{pi ~ Perm} [ T(pi(O)) >= T(O) ]
```

where `pi` permutes KOL-to-role assignments within events, preserving the
marginal lead distribution. The structure exists on symbol `s` if `p_e <
alpha`.

### 5.4 Orthogonality To Trivial Proxies

The trait must not reduce to account scale or activity. With `rho = Spearman`:

```text
| rho(O, follower_k) |      small
| rho(O, log |E_k|) |       small
| rho(O, log #tweets_k) |   small
```

### 5.5 Not News-Reaction Speed

To rule out "earliest = fastest to react to market news", retest the hierarchy
on the price-quiet sub-sample `S_q` (events on days with `|return|` below a
quantile), where the news-reaction channel is weakened:

```text
T(S_q) significant under the same permutation null
```

### 5.6 Stability And Identity Versus Timezone

A routing trait must persist. Let `O^{(t)}` denote the trait estimated in
time bin `t`; persistence is the lag-1 cross-bin Spearman:

```text
rho_persist = rho( O^{(t)}, O^{(t+1)} )
```

against a shuffle null `rho(pi(O^{(t)}), O^{(t+1)})` that destroys KOL-label
identity. The stricter test separates identity from timezone: let `m(k)` be a
different KOL with median hour nearest to `k`. Persistence is identity-driven
iff:

```text
rho_id  = rho( O^{(t)}_k,    O^{(t+1)}_k )        >>    (same KOL)
rho_tz  = rho( O^{(t)}_k,    O^{(t+1)}_{m(k)} )         (nearest-hour KOL)
```

### 5.7 Cross-Asset Consistency

Splitting symbols into disjoint groups `A`, `B`, the trait should agree:

```text
rho( O^{(A)}, O^{(B)} ) > 0
```

### 5.8 K-Weak Dependence (Bridge To The Router)

The decisive property for Section 7 is that `O_k` is a KOL-level attribute
estimated from history, not a function of the current candidate-pool size `K`:

```text
O_k = f(history of k),    partial O_k / partial K = 0
```

This does not imply the router has zero error. It means the originator trait
itself does not introduce a prompt-length, distractor, or position-bias channel.
Whatever K-dependence the router has comes from competition among candidates
and feature discriminability, not from reading a longer prompt. This yields the
design expectation used in Section 7:

```text
the OL router is K-weakly dependent relative to full-context LLM selection.
```

This relative property is not assumed as a theorem; it is motivated by the
construction and then tested through routing quality and selector latency.

### 5.9 Empirical Substitution

Substituting the measured quantities into the tests above:

```text
confound (5.2)         : rho(L_raw, h) = -0.768                  -> residualization required
existence (5.3)        : 16/17 symbols, p = 0.001
orthogonality (5.4)    : rho(O, followers) = -0.068  (n = 641)
                         rho(O, log |E|) = +0.02 ;  rho(O, log #tweets) = +0.03
not news-reaction (5.5): price-quiet below median |return|: p = 0.001 by symbol, pooled 5,959 events
                         bottom |return| quartile: 17/17 symbols, p < 0.01, pooled 3,179 events
stability (5.6)        : rho_persist = +0.447  (n = 2,104, p = 6e-104)
                         shuffle null mean +0.001, 95th pct +0.037
identity vs tz (5.6)   : rho_id = +0.42 to +0.47  vs  rho_tz = +0.02 to +0.09  (gap +0.38)
                         holds under 3 timezone controls + activity double-control
cross-asset (5.7)      : rho = +0.500  (p = 5e-19, n = 279)
robustness             : original-tweet-only 16/17 ; min-KOLs {3,10} both 16/17
```

The trait is real, stable, identity-driven, orthogonal to trivial proxies, and
point-in-time measurable. Its construction gives the router a K-weakly
dependent structural signal, which is the premise Section 7 requires.

## 6. Two Capture Channels And Why Two Metrics Are Reported

Capture can move with `K` through two channels acting on opposite sides of the
ratio:

```text
Capture = E[ sum_{i in A_e} Y_i ] / E[ sum_{i in Top_r(C_e; Y)} Y_i ]
```

```text
(a) selection quality :  tau(K) rises  ->  the selected set drifts from the oracle.
(b) pool reachability :  larger K      ->  higher-Y candidates become selectable.
```

Channel (a) lowers the numerator (the selected set worsens relative to the
oracle). Channel (b) raises the numerator (a larger pool lets the selector
reach higher-Y items) and, when the oracle is defined over the seen pool, also
raises the denominator. The two metrics isolate them differently:

```text
shown-pool oracle :  denominator = oracle of the seen pool, which rises with K.
                     (a) lowers the numerator; (b) raises the denominator
                     -> capture tends to fall. Dominated by (a) plus a size effect.
Kmax-oracle       :  denominator = a fixed global oracle, independent of K.
                     (a) lowers the numerator; (b) raises it (information gain)
                     -> capture may rise or fall depending on which dominates.
```

The dilution experiment reports both because they separate the two channels.
The paper's claim concerns utilization, not absolute performance: channel (b)
may let a strong LLM extract some net information from a larger pool (visible
on the Kmax metric), but channel (a) prevents that gain from being
proportional to the extra candidates (visible on the shown metric). The joint
signature is diminishing marginal capture per added candidate, not monotone
decline.

## 7. The Router Attenuates The LLM-Specific Channel (a)

The OL router uses the originator structure discovered in Section 5 as a linear
scorer over the pre-estimated frozen trait:

```text
s_i = beta' X_i + gamma O_{k(i)} + delta_1 O_{k(i)} vis_i + delta_2 O_{k(i)} nov_i
```

where `O_{k(i)}` is the residualized originator trait of the originating KOL,
`X_i` are non-OL origin-time controls, and `vis`, `nov` are originator
visibility and semantic novelty. It does not read the full `K` candidate texts
inside a single prompt and does not ask an LLM to compare all K candidates at
the first stage. Therefore it bypasses the LLM-specific context-length,
distractor, and position-bias channels in (a). We use the relative design
expectation:

```text
the OL selector is K-weakly dependent compared with full-context LLM selection.
```

This is not a claim that `tau_OL(K)` is mathematically non-increasing. First,
`O_k` is independent of the current pool size `K` (`partial O_k / partial K =
0`, Section 5.8). Second, the router does not introduce the full-context LLM's
prompt-length, distractor, or position-bias channels. These facts motivate the
K-weak dependence claim, which is evaluated empirically rather than assumed.

Routing also reduces the LLM's problem from `K` to a shortlist `b << K`, so the
downstream LLM operates on a smaller comparison set. If effective LLM selection
error is lower on the shorter list, then `tau_LLM(b) < tau_LLM(K)`. Under
approximate independence of the two stages, routed capture admits the heuristic
decomposition:

```text
E[Capture_routed] approx Enrichment(OL, b, K) * E[Capture_LLM(b)]
```

This factorization is a heuristic, not an exact identity; it identifies the two
channels through which routing beats full-context selection: a first stage
whose effective error is K-weakly-dependent, and a second stage whose error is
reduced by shrinking the pool from `K` to `b`.

## 8. Empirical Substitution Into The Derived Results

The formulas above contain only placeholders. We now substitute the measured
quantities into the derived results to strengthen the conclusions.

### 8.1 Channel (a): LLM selection quality does not improve with K

Section 6 predicts that the shown-pool oracle, dominated by channel (a) plus
the seen-pool size effect, should fall as `K` grows if `tau_LLM(K)` does not
fall. The dilution experiment supports this: full-context shown-pool capture is
lower at `K = 30` than at `K = 10` for all four LLMs:

```text
DeepSeek V4 Flash :  0.368 -> 0.184
Claude Sonnet 4.6  :  0.339 -> 0.279
GPT-5.4            :  0.365 -> 0.308
Gemini 2.5 FL      :  0.280 -> 0.227
```

Within the pool each LLM actually saw, more candidates did not improve
selection quality. This is the empirical instance of channel (a).

### 8.2 Channel (b) is real but the marginal utilization is poor

Section 6 also predicts that the Kmax-oracle, mixing (a) and (b), can rise if
the information gain from a larger pool exceeds the error gain. The data
confirm this — but the gain is far from proportional to the extra candidates.
Tripling the pool (`K = 10 -> 30`) yields the following Kmax-oracle capture
changes:

```text
DeepSeek V4 Flash :  0.272 -> 0.184   (-32%; net waste)
Claude Sonnet 4.6  :  0.214 -> 0.279   (+30%)
GPT-5.4            :  0.252 -> 0.308   (+22%)
Gemini 2.5 FL      :  0.206 -> 0.227   (+10%)
```

Even on the Kmax metric, the metric most favorable to the LLM, tripling the
candidate pool produces only a 10-30% capture gain for three of four models
(and a loss for the fourth). The majority of the additional candidates are
wasted: channel (b) lets the LLM extract some information, but channel (a)
prevents proportional utilization. This is the paper's claim — not that more
data makes the LLM worse, but that more data is largely unused.

### 8.3 The router wins on the channel it is designed to attenuate

Section 7 predicts that a first stage that bypasses the LLM-specific part of
channel (a), followed by an LLM operating on a shortlist, can beat full-context
selection on the common Kmax-oracle metric — the metric on which strong LLMs
otherwise benefit from more candidates. The dilution experiment supports this
across all four LLMs:

```text
ol_b capture  >  full_K capture   (Kmax-oracle, 4/4 LLMs)
```

Routing attenuates the LLM-specific first-stage error channel and shrinks the
LLM's pool from K to b at the second stage. This beats full-context selection
even where strong LLMs net-benefit from larger pools.

### 8.4 The linear form suffices, locating value in structure

The router of Section 7 is a deliberate linear ridge model. Its ablation shows
the originator structure does real work but only when coupled with origin-time
context:

```text
OL Only (O_k alone)            :  NDCG@3 = 0.650
No-OL Strong (context only)    :  NDCG@3 = 0.712
OL-Origin Full (context + O)   :  NDCG@3 = 0.755   (best)
```

`O_k` alone is insufficient (below context-only); the full linear combination
beats both. That a K-weakly-dependent linear selector suffices is itself the
result: it locates the exploitable value in the KOL-derived structure (the
data), not in model capacity (the architecture). Had a high-capacity model been
necessary, it would be harder to attribute the gain cleanly to the discovered
KOL structure rather than to representation capacity.

## 9. Full Logical Chain

```text
Observation      : large social-media streams are expensive and inefficient
                   for direct LLM consumption.
Diagnosis        : direct LLM consumption asks the model to solve first-stage
                   routing under effective error that need not decrease with K
                   (Section 4).
Representation   : social-media observations are actor-time-frame data, not
                   only raw text.
Structural       : KOL streams contain a stable originator role O_k that is
discovery          point-in-time, residualized against posting hour, and not
                   reducible to follower scale, frequency, or news-reaction
                   speed (Section 5).
Model implication: use O_k as a low-latency, K-weakly-dependent selector over
                   newly originated frames.
Model design     : linear OL-Origin router with origin-time controls,
                   residualized originator role, and role-context interactions
                   (Section 7).
Validation       : channel (a) is real and measured (Section 8.1); the router
                   attenuates it and beats full-context selection on the common
                   Kmax metric (Section 8.3); the linear form suffices, so
                   value lies in structure, not capacity (Section 8.4).
```

The central claim:

```text
The bottleneck is not the absence of social-media information, but the absence
of a point-in-time structure that tells the agent which social-media
information deserves expensive reasoning.
```

## 10. Literature Anchors

- Luce, R.D. (1959), *Individual Choice Behavior*. — choice axiom / PL basis.
- Plackett, R.L. (1975), "The Analysis of Permutations," *JRSS-C*. — listwise PL.
- Frisch & Waugh (1933); Lovell (1963). — FWL residualization.
- Liu et al. (2023), "Lost in the Middle," arXiv:2307.03172. — long-context degradation.
- Levy, Jacoby & Goldberg (2024), arXiv:2402.14848. — length-isolated degradation.
- Modarressi et al. (2025), NoLiMa, arXiv:2502.05167. — distractor scaling.
- Hsieh et al. (2024), RULER, arXiv:2404.06654. — distractor/aggregate scaling.
- Wang et al. (2023), "LLMs are not Fair Evaluators," arXiv:2305.17926. — position bias.
- Qiao et al. (2026), listwise reranking position bias, arXiv:2604.03642. — listwise position bias.
- Sun et al. (2023), RankGPT, arXiv:2304.09542. — listwise does not scale with K.
- Cuconasu et al. (2025), arXiv:2505.15561. — position bias marginal in RAG (honest counterpoint).
- Iyengar & Lepper (2000), "When Choice Is Demotivating," *JPSP*. — overload analogue (not used as LLM evidence).
- Chernev, Bockenholt & Goodman (2015), *J. Consumer Psychology*. — overload meta (moderated).
