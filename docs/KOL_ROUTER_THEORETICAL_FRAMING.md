# Theoretical Framing: From LLM Selection Inefficiency To KOL-Origin Routing

This note states the theoretical logic behind the KOL Origin-Aware Narrative
Router. The paper is not primarily a new neural architecture paper. Its central
claim is that large-scale KOL social-media data contain a point-in-time
originator structure, and that this structure can route agent attention with a
minimal, low-latency linear score.

The key distinction is:

```text
available social-media data != usable agent information
```

Directly passing many social-media candidates to an LLM gives the model access
to more text, but it also asks the model to solve a difficult first-stage
selection problem. The router is designed to decide which candidates deserve
downstream memory, retrieval, LLM reasoning, or strategy research.

## 1. Attention Allocation Setup

For each event-day `e`, the agent observes a pool of originated social-media
narrative candidates:

```text
C_e = {1, ..., K_e}
```

Each candidate `i` has an unobserved future value:

```text
Y_i = future follower-weighted reach of candidate i
```

The agent can send only a small subset to expensive downstream reasoning:

```text
A_e subset C_e,    |A_e| = r
```

The ideal decision is:

```text
max_A  E[ sum_{i in A_e} Y_i | I_e ] - lambda * Cost(A_e)
subject to A_e subset C_e, |A_e| = r
```

where `I_e` is the point-in-time information set before later diffusion and
follower confirmation are observable. This is an attention-allocation problem,
not a direct market-return prediction problem.

The oracle set is:

```text
A_e^* = Top_r(C_e; Y)
```

and the empirical capture metric is:

```text
Capture(S)
  = E[sum_{i in S(C_e)} Y_i]
    / E[sum_{i in A_e^*} Y_i]
```

## 2. Full-Context LLM As A Noisy Listwise Selector

A full-context LLM selector reads the whole candidate pool and chooses `r`
items:

```text
A_full = S_LLM(C_e)
```

This asks one model call to solve several subproblems at once:

```text
semantic parsing
deduplication
origin detection
actor reliability inference
cross-candidate comparison
importance ranking
```

We model this as noisy utility estimation:

```text
tilde Y_i^LLM(K) = Y_i + epsilon_i(K)
```

and:

```text
S_LLM(C_e) = Top_r({tilde Y_i^LLM(K) : i in C_e})
```

Here `epsilon_i(K)` is effective selection error. It includes bias and variance
induced by context length, distractors, formatting, position effects, and the
difficulty of comparing many similar financial narratives in one prompt. The
paper does not need to separately identify these components; it only needs the
weaker claim that more candidates are not automatically converted into better
first-stage selection.

## 3. Error-To-Capture Link

One way to connect selection error to capture is a Plackett-Luce temperature
model:

```text
P_sel(i | C) = exp(Y_i / tau) / sum_{j in C} exp(Y_j / tau)
```

where `tau` is an effective error scale. Smaller `tau` gives sharper selection
toward high-value candidates; larger `tau` flattens selection probabilities
toward uniform choice. In the standard utility-ranking case:

```text
tau -> 0     implies near-oracle selection
tau -> inf   implies near-uniform selection
```

Thus rising effective error compresses expected capture toward the random
baseline:

```text
random top-r baseline = r / K
```

This is a modeling bridge, not a claim that the real LLM literally samples from
a PL distribution. It formalizes the intuition that noisier listwise selection
captures less of the oracle top-r future reach.

The same point can be expressed pairwise. If candidate `i` is truly better than
candidate `j`:

```text
Delta_ij = Y_i - Y_j > 0
```

the LLM ranks `j` above `i` when:

```text
epsilon_j(K) - epsilon_i(K) > Delta_ij
```

Under approximately independent Gaussian errors with variance
`sigma_LLM^2(K)`, the pairwise ranking error is:

```text
P(error)
  = 1 - Phi( Delta_ij / (sqrt(2) * sigma_LLM(K)) )
```

Both views give the same qualitative condition:

```text
selection quality improves when signal gaps rise or effective selection noise falls.
```

## 4. Candidate-Pool Size And Data Utilization

The paper does not claim that more social-media data always makes an LLM worse.
The defensible claim is:

```text
more candidates do not automatically become more usable information.
```

Formally:

```text
sigma_LLM(K) is not guaranteed to decrease with K
```

or, in the PL notation:

```text
tau_LLM(K) is not guaranteed to decrease with K
```

This is consistent with long-context, multi-document, distractor, and listwise
position-bias evidence: LLMs can fail to use long contexts robustly, can be
sensitive to where relevant information appears, and can degrade under larger
multi-document or listwise comparison settings.

The relevant failure mode is not:

```text
more text always hurts the LLM
```

but rather:

```text
larger candidate pools can add useful opportunities while also increasing
selection complexity, so marginal data utilization can be poor.
```

## 5. Two Capture Channels And Two Oracle Metrics

Capture can move with `K` through two channels:

```text
Capture = E[ sum_{i in A_e} Y_i ] / E[ sum_{i in Top_r(C_e; Y)} Y_i ]
```

```text
(a) selection quality:
    larger K can increase effective selection error, lowering the numerator.

(b) pool information:
    larger K can contain better oracle candidates, raising the opportunity set.
```

These two channels motivate two reporting metrics in the listwise experiment.

Shown-pool oracle:

```text
selected reach / oracle top-r reach inside the candidate pool shown to the LLM
```

This mostly isolates channel (a): within the pool the model saw, did it choose
the high-value candidates?

Kmax-oracle:

```text
selected reach / oracle top-r reach inside the fixed K=30 candidate pool
```

This mixes channel (a) with channel (b): it asks how much of the full
opportunity set the model captured. The paper's claim concerns utilization, not
monotone decline. A strong LLM may extract some net information from a larger
pool, but the gain need not be proportional to the additional candidates.

## 6. Social Media Has Actor-Time-Frame Structure

A financial social-media candidate is not only text. It is:

```text
i = (text_i, frame_i, time_i, actor_i)
```

where:

```text
text_i   = what is being said
frame_i  = which semantic narrative the post belongs to
time_i   = when the narrative appears inside the event
actor_i  = which KOL originated or amplified it
```

The pre-popularity setting removes later diffusion signals from the online
information set. Therefore the routing question is:

```text
Which newly originated narrative frames should receive scarce agent attention
before follower confirmation is observable?
```

This reframes the task. A text-only selector scores what is said. A KOL-aware
router also uses who originated it, when it appeared, and under what origin-time
context.

## 7. Empirical Structure: Stable Originator Role

Let `k(i)` be the KOL associated with candidate `i`. Define:

```text
O_k = residualized originator role of KOL k
```

In the implementation, `O_k` is estimated only from pre-validation history using
event-order lead-lag behavior, then residualized against median UTC posting
hour and hour squared. It is therefore a point-in-time, pre-estimated actor
trait rather than an ex-post validation/test label.

Let `X_i` denote ordinary origin-time controls:

```text
X_i = {
  follower visibility,
  verified status,
  origin rank,
  elapsed event time,
  semantic novelty,
  sentiment,
  historical activity,
  other non-OL context controls
}
```

The structural condition is:

```text
I(Y_i ; O_{k(i)} | X_i) > 0
```

Equivalently:

```text
E[Y_i | X_i, O_{k(i)}] != E[Y_i | X_i]
```

This says that, after controlling for ordinary origin-time context, the stable
originator role contains incremental information about future follower-weighted
reach.

The falsification conditions are:

```text
O_k is not merely follower scale
O_k is not merely posting frequency
O_k is not merely timezone or median posting hour
O_k is not merely news-reaction speed
O_k is not a bot/retweet artifact
O_k is not an ex-post popularity label
```

These are tested through residualization, shuffling, follower replacement,
no-OL controls, raw-OL comparison, and robustness checks.

## 8. Router Design As Structure Identification

If:

```text
I(Y_i ; O_{k(i)} | X_i) > 0
```

then the first-stage selector should not ignore `O_k`:

```text
O_{k(i)} should enter the first-stage attention decision.
```

This does not uniquely determine a functional form. Many models could use this
signal:

```text
nonparametric ranking
tree model
neural scorer
LLM prompt feature
linear router
```

The paper chooses a lightweight ridge router with role-context interactions as
a deployment-oriented and identification-oriented design. The claim is not that
linear ridge is the most expressive selector. The claim is stronger and cleaner:

```text
If a structure discovered from large-scale KOL data can improve routing through
a simple ridge score, then the value is more likely to come from the discovered
structure than from hidden model capacity.
```

In other words, the model is intentionally minimal. Neural scorers,
transformer fine-tuning, and LLM-based selectors are useful baselines, but they
are not the conceptual contribution. The contribution is the point-in-time
originator structure and its use as an agent routing signal.

The modeling target is:

```text
m(I_i) = E[Y_i | X_i, O_{k(i)}]
```

The OL-Origin router uses a first-order ridge approximation:

```text
s_i =
  beta' X_i
  + gamma O_{k(i)}
  + delta_1 O_{k(i)} * visibility_i
  + delta_2 O_{k(i)} * novelty_i
```

with coefficients estimated by:

```text
min_theta  sum_{i in train} (Y_i - theta' Z_i)^2 + lambda ||theta||_2^2
```

where:

```text
Z_i = [X_i, O_{k(i)}, O_{k(i)} * visibility_i, O_{k(i)} * novelty_i]
```

The terms have separate roles:

```text
beta' X_i
  strong non-OL baseline; controls for visibility, timing, sentiment, novelty,
  and history

gamma O_{k(i)}
  tests whether the discovered originator role adds information after controls

O_{k(i)} * visibility_i and O_{k(i)} * novelty_i
  test whether the role is useful conditional on current audience scale and
  semantic novelty
```

The model is not a global KOL leaderboard:

```text
high O_k does not imply every post by k is important
```

The useful object is conditional:

```text
who originated the frame
+ when it was originated
+ how visible and novel the frame is at origin time
```

## 9. How Routing Helps Downstream LLM Use

The router first selects a shortlist:

```text
B_e = Top_b({s_i : i in C_e}),    r <= b < K
```

The downstream LLM then reranks only the shortlist:

```text
A_routed = S_LLM(B_e)
```

Routing can help through two channels.

First, the shortlist is enriched:

```text
P(i in A_e^* | i in B_e)
  >
P(i in A_e^* | i in C_e)
```

Second, the LLM operates on a smaller comparison set:

```text
sigma_LLM(b) < sigma_LLM(K)
```

or:

```text
tau_LLM(b) < tau_LLM(K)
```

Under the pairwise view, routing helps when:

```text
Delta_b / sigma_LLM(b) > Delta_K / sigma_LLM(K)
```

where `Delta_b` is the effective value gap among candidates seen after routing.
Under the channel view, the routed system is:

```text
structural router over K candidates -> shortlist b -> LLM selection over b
```

The router does not eliminate all selection error. It bypasses the
LLM-specific first-stage error channel created by long prompts, distractors,
and listwise position effects, then asks the LLM to solve a smaller problem.

Heuristically:

```text
E[Capture_routed]
  approx Enrichment(OL, b, K) * E[Capture_LLM(b)]
```

This factorization is not an exact identity. It identifies the two mechanisms
tested empirically: shortlist enrichment and reduced downstream listwise
selection burden.

## 10. Empirical Substitution

The formulas above define the logic. The measured results are then substituted
into the derived conditions.

### 10.1 Full LLM Selection Quality Under Larger K

The shown-pool oracle isolates whether the LLM selects well inside the pool it
actually saw. In the current listwise experiment, shown-pool full-context
capture is lower at `K=30` than at `K=10` for all four tested LLMs:

```text
DeepSeek V4 Flash :  0.368 -> 0.184
Claude Sonnet 4.6 :  0.339 -> 0.279
GPT-5.4           :  0.365 -> 0.308
Gemini 2.5 FL     :  0.280 -> 0.227
```

This does not prove universal monotone degradation. It supports the narrower
claim that, in this task, larger candidate pools were not converted into better
within-pool selection quality.

### 10.2 Kmax-Oracles Show Partial Utilization, Not Full Utilization

The fixed Kmax-oracle metric is more favorable to the LLM because a larger pool
can contain better candidates. Tripling the pool from `K=10` to `K=30` gives:

```text
DeepSeek V4 Flash :  0.272 -> 0.184   (-32%)
Claude Sonnet 4.6 :  0.214 -> 0.279   (+30%)
GPT-5.4           :  0.252 -> 0.308   (+22%)
Gemini 2.5 FL     :  0.206 -> 0.227   (+10%)
```

Thus the conclusion is not "more data always hurts." Three LLMs extract some
net information from the larger pool, while one loses capture. The central
point is that additional candidates are only partially utilized: the extra
opportunity set does not translate into proportional capture gains.

### 10.3 OL Routing Beats Full K=30 Selection Across Backends

The same Kmax-oracle metric compares routed selection against full-context
selection over the same `K=30` opportunity set. In the current results, at
least one OL-Origin shortlist beats full `K=30` for every tested LLM:

```text
DeepSeek V4 Flash : full K30 0.184, OL b10 0.315, OL b20 0.310
Gemini 2.5 FL     : full K30 0.227, OL b10 0.320, OL b20 0.300
Claude Sonnet 4.6 : full K30 0.279, OL b10 0.382, OL b20 0.350
GPT-5.4           : full K30 0.308, OL b10 0.364, OL b20 0.369
```

This is the small experiment's main implication: a structural entry-layer
router can improve how the LLM uses the large candidate pool.

### 10.4 Originator Role Is Stable And Orthogonal

The router requires `O_k` to be a real, stable, point-in-time trait rather than
a proxy for follower scale, posting hour, or news-reaction speed. The structure
diagnostics support this:

```text
raw lead vs median posting hour    : Spearman = -0.768
OLtrait vs followers               : Spearman = -0.068 across 641 KOLs
lead-lag hierarchy                 : 16/17 symbols, permutation p = 0.001
price-quiet / bottom-quartile days : 17/17 symbols, p < 0.01
original-tweet-only rerun          : 16/17 symbols significant
temporal persistence, lag-1        : Spearman about +0.447
timezone-matched null, lag-1       : Spearman about +0.02 to +0.09
cross-asset group split            : Spearman about +0.500
```

These facts justify treating `O_k` as an identity-driven originator role rather
than a timezone, scale, frequency, or news-reaction artifact.

### 10.5 Linear Form Suffices

The router is deliberately simple. The ablation shows that `O_k` is useful only
when coupled with origin-time context:

```text
OL Only (O_k alone)          : NDCG@3 = 0.650
No-OL Strong (context only)  : NDCG@3 = 0.712
OL-Origin Full              : NDCG@3 = 0.755
```

And:

```text
Full - No-OL Strong:
  Delta NDCG@3 = +0.043, 90% CI [+0.019, +0.069]
  Delta Hit@1  = +0.072, 90% CI [+0.013, +0.156]
  JS improve   = +0.039, 90% CI [+0.022, +0.058]
```

This is the identification result. Since a minimal ridge router can exploit the
trait, the value is easier to attribute to the KOL-derived structure than to
hidden representation capacity.

## 11. Latency And Component Cost

If the LLM itself performs first-stage selection over all `K` candidates:

```text
Cost_full_selector = T_LLM(K)
```

If OL-Origin performs first-stage selection:

```text
Cost_OL_selector = T_OL(K)
```

The routed pipeline then costs:

```text
Cost_routed_total = T_OL(K) + T_LLM(b)
```

The component comparison is:

```text
T_LLM(K) / T_OL(K)
```

In the current small experiment with `K=30`:

```text
T_OL(30)  approximately 0.048 ms
T_LLM(30) approximately 1.22 s to 7.09 s, depending on backend
```

Thus OL-Origin is not only a quality component. It is a low-latency first-stage
selector that decides what deserves expensive reasoning.

## 12. Full Logical Chain

```text
Observation:
  Large social-media streams are expensive and inefficient for direct LLM
  consumption.

Diagnosis:
  Direct LLM consumption asks the model to solve first-stage routing under
  effective selection error.

Representation:
  Financial social media is actor-time-frame data, not only raw text.

Structural discovery:
  KOL streams contain a stable originator role O_k that is point-in-time,
  residualized against posting hour, and not reducible to follower scale,
  frequency, timezone, bot/retweet behavior, or news-reaction speed.

Model implication:
  Use O_k as a low-latency structural signal in the first-stage attention
  decision.

Model design:
  Use a minimal ridge utility-index router with origin-time controls,
  residualized originator role, and role-context interactions.

Validation:
  Main experiment tests early alert quality.
  Ablation tests whether the discovered structure is doing real work.
  Listwise small experiment tests whether routing improves downstream LLM
  attention allocation under large candidate pools.
  Latency analysis tests selector-component cost.
```

The central claim is:

```text
The bottleneck is not a lack of social-media data. The bottleneck is the lack
of a point-in-time structure that tells the agent which social-media items
deserve expensive reasoning.
```

## 13. Literature Anchors

Random utility and noisy choice:

```text
Thurstone (1927), "A Law of Comparative Judgment"
Luce (1959), Individual Choice Behavior
McFadden (1974), "Conditional Logit Analysis of Qualitative Choice Behavior"
Plackett (1975), "The Analysis of Permutations"
```

These support utility-index and noisy-choice views of first-stage selection.

Regularized linear scoring:

```text
Hoerl and Kennard (1970), "Ridge Regression: Biased Estimation for
Nonorthogonal Problems"
```

This supports a stable regularized linear estimator for correlated origin-time
controls and interaction terms.

Learning to rank:

```text
Joachims (2002), "Optimizing Search Engines Using Clickthrough Data"
```

This supports treating first-stage routing as candidate ranking rather than
open-ended generation.

Rational inattention and costly information processing:

```text
Sims (2003), "Implications of Rational Inattention"
Mackowiak, Matejka, and Wiederholt (2021), "Rational Inattention: A Review"
```

These justify distinguishing available information from processed information.

Interpretable and minimal models:

```text
Rudin (2019), "Stop Explaining Black Box Machine Learning Models for High
Stakes Decisions and Use Interpretable Models Instead"
Grinsztajn, Oyallon, and Varoquaux (2022), "Why do tree-based models still
outperform deep learning on tabular data?"
```

These support not making model capacity the paper's main contribution when the
input is a structured tabular routing signal and latency/interpretability are
central.

Long-context, distractor, and listwise LLM limitations:

```text
Liu et al. (2024), "Lost in the Middle: How Language Models Use Long Contexts"
Levy, Jacoby, and Goldberg (2024), "Same Task, More Tokens: the Impact of Input
Length on the Reasoning Performance of Large Language Models"
Levy et al. (2025), "More Documents, Same Length: Isolating the Challenge of
Multiple Documents in RAG"
Hsieh et al. (2024), "RULER: What's the Real Context Size of Your Long-Context
Language Models?"
Sun et al. (2023), "RankGPT: Leveraging ChatGPT for Text Ranking"
```

These support the assumption that larger contexts and larger candidate pools
are not automatically converted into better decisions.

Reference links:

```text
Lost in the Middle:
https://aclanthology.org/2024.tacl-1.9/

Same Task, More Tokens:
https://arxiv.org/abs/2402.14848

More Documents, Same Length:
https://aclanthology.org/2025.findings-emnlp.1064/

RULER:
https://arxiv.org/abs/2404.06654

RankGPT:
https://arxiv.org/abs/2304.09542

Rational Inattention review:
https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp2570~a3979fbfa5.en.pdf

McFadden conditional logit:
https://eml.berkeley.edu/reprints/mcfadden/zarembka.pdf

Ridge regression:
https://www.tandfonline.com/doi/abs/10.1080/00401706.1970.10488634

Learning to rank:
https://www.cs.cornell.edu/people/tj/publications/joachims_02c.pdf

Interpretable models:
https://www.nature.com/articles/s42256-019-0048-x

Tabular deep learning comparison:
https://arxiv.org/abs/2207.08815
```
