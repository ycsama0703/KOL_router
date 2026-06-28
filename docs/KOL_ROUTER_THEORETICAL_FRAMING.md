# Theoretical Framing: From Inefficient Social-Media Attention To KOL Routing

This note states the theoretical logic behind the KOL Origin-Aware Narrative
Router. The goal is not to prove that the discovered originator feature must
work from first principles. The goal is to make the paper's argument precise:
large social-media streams create a first-stage attention-allocation problem;
direct LLM consumption is a noisy and costly way to solve that problem; the
observed KOL-origin structure provides a point-in-time signal that can be used
as a cheaper and more stable routing layer before expensive reasoning.

The most important distinction is:

```text
available social-media data != usable agent information
```

The router is designed to turn a large candidate pool into a small, enriched
set of narratives that deserve downstream memory, retrieval, LLM reasoning, or
strategy research.

## 1. Attention Allocation Problem

For each event-day `e`, the agent observes a candidate pool:

```text
C_e = {1, ..., K_e}
```

Each candidate `i` is an originated social-media narrative frame. It has an
unobserved future value:

```text
Y_i = future follower-weighted reach of candidate i
```

At decision time, the agent only has point-in-time information:

```text
I_i = information available before later diffusion is observed
```

The agent cannot send all `K_e` candidates to expensive downstream reasoning.
It must select a small attention set:

```text
A_e subset C_e,    |A_e| = r
```

The ideal first-stage decision is:

```text
max_A  E[ sum_{i in A_e} Y_i | I_e ] - lambda * Cost(A_e)
subject to A_e subset C_e, |A_e| = r
```

where `I_e = {I_i : i in C_e}` and `lambda` prices downstream reasoning cost.
This makes the problem a budgeted attention-allocation problem, not a direct
market-return prediction problem.

The oracle set is:

```text
A_e^* = Top_r(C_e; Y)
```

and the empirical evaluation asks how much future reach a selector captures
relative to this oracle:

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

We model this as a noisy listwise selector. For each candidate:

```text
tilde Y_i^LLM(K) = Y_i + epsilon_i(K)
```

and the LLM chooses:

```text
S_LLM(C_e) = Top_r({tilde Y_i^LLM(K) : i in C_e})
```

Here `epsilon_i(K)` is not random noise in the LLM weights. It is effective
selection noise induced by context length, number of candidates, distractors,
formatting, position effects, and the difficulty of comparing many similar
social-media narratives in one prompt.

This is a random-utility abstraction: a selector behaves as if it ranks a latent
utility plus an error term. The paper uses this abstraction because our task is
listwise selection, not open-ended generation.

## 3. Why More Candidates Need Not Improve LLM Selection

The paper does not claim that more social-media data always makes an LLM worse.
The weaker and more defensible claim is:

```text
More candidates do not automatically become more usable information.
```

Formally, define the LLM's effective selection-noise variance:

```text
Var[epsilon_i(K)] = sigma_LLM^2(K)
```

When the LLM compares two candidates `i` and `j`, with true gap:

```text
Delta_ij = Y_i - Y_j > 0
```

the LLM ranks `j` above `i` when:

```text
tilde Y_j^LLM(K) > tilde Y_i^LLM(K)
```

Equivalently:

```text
epsilon_j(K) - epsilon_i(K) > Delta_ij
```

If the two error terms are approximately independent Gaussian errors with
variance `sigma_LLM^2(K)`, then:

```text
P(LLM ranks j above i)
  = 1 - Phi( Delta_ij / (sqrt(2) * sigma_LLM(K)) )
```

Therefore, if a larger candidate pool increases effective selection noise, or
if it fails to reduce noise enough to exploit the additional candidates, the
probability of selecting the truly better candidate does not necessarily
improve.

This gives the paper's key LLM-side theoretical condition:

```text
sigma_LLM(K) is not guaranteed to decrease with K.
```

This condition is consistent with long-context and multi-document LLM evidence:
models can fail to robustly use information in long contexts, can be sensitive
to where relevant information appears, and can find multi-document comparison
hard even when total context length is controlled.

In our setting, the relevant failure mode is not "the LLM sees too much text and
therefore becomes worse." The relevant failure mode is:

```text
the LLM sees many plausible social-media candidates but does not reliably
convert the larger candidate set into better first-stage attention allocation.
```

## 4. Social Media Has Actor-Time-Frame Structure

A financial social-media candidate is not only a text string. It can be
represented as:

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
information set. Thus the routing question becomes:

```text
Which newly originated narrative frames should receive scarce agent attention
before follower confirmation is observable?
```

This reframes the task. A text-only selector scores what is said. A KOL-aware
router also uses who originated it, when it was originated, and under what
origin-time context.

## 5. Empirical Structure: Stable Originator Role

Let `k(i)` be the KOL associated with candidate `i`. Define a stable originator
role:

```text
O_k = residualized originator role of KOL k
```

In the implementation, `O_k` is estimated only from pre-validation history using
event-order lead-lag behavior, and then residualized against median UTC posting
hour and hour squared. It is therefore a point-in-time, pre-estimated actor
trait rather than an ex-post label from the validation/test period.

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

The structural condition needed for the router is:

```text
I(Y_i ; O_{k(i)} | X_i) > 0
```

Equivalently:

```text
E[Y_i | X_i, O_{k(i)}] != E[Y_i | X_i]
```

In words, after controlling for ordinary origin-time context, the stable
originator role still contains incremental information about future
follower-weighted reach.

The falsification tests are as important as the positive result:

```text
O_k is not merely follower scale
O_k is not merely posting frequency
O_k is not merely timezone or median posting hour
O_k is not merely news-reaction speed
O_k is not a bot/retweet artifact
O_k is not an ex-post popularity label
```

The empirical design therefore uses residualization, shuffling, follower
replacement, no-OL controls, raw-OL comparison, and robustness checks to test
whether `O_k` is a real structural signal rather than a proxy for a simpler
surface variable.

## 6. Router Design Implied By The Structure

If:

```text
I(Y_i ; O_{k(i)} | X_i) > 0
```

then a first-stage selector should expose `O_k` directly instead of requiring an
LLM to rediscover the originator structure from raw prompt context on every
event-day.

The target conditional value function is:

```text
m(I_i) = E[Y_i | X_i, O_{k(i)}]
```

The OL-Origin router approximates this value with a lightweight linear score:

```text
s_i =
  beta' X_i
  + gamma O_{k(i)}
  + delta_1 O_{k(i)} * visibility_i
  + delta_2 O_{k(i)} * novelty_i
```

where:

```text
X_i                       = non-OL origin-time controls
O_{k(i)}                  = residualized originator role
O_{k(i)} * visibility_i   = originator role under current audience scale
O_{k(i)} * novelty_i      = originator role under semantic novelty
```

The interaction terms are central. The model is not a global KOL leaderboard:

```text
high O_k does not imply every post by k is important
```

Instead, the useful object is conditional:

```text
who originated the frame
+ when it was originated
+ how visible and novel the frame is at origin time
```

Thus the router is a structural attention allocator, not an alternative
foundation model.

## 7. Why Routing Can Improve Downstream LLM Use

The router first selects a shortlist:

```text
B_e = Top_b({s_i : i in C_e}),    r <= b < K
```

The downstream LLM then performs listwise reranking only over the shortlist:

```text
A_routed = S_LLM(B_e)
```

Routing can improve LLM use through two mechanisms.

First, the shortlist is enriched:

```text
P(i in A_e^* | i in B_e)
  >
P(i in A_e^* | i in C_e)
```

Second, the LLM faces a smaller and less noisy listwise selection problem:

```text
sigma_LLM(b) < sigma_LLM(K)
```

Under the noisy-selector model, both effects reduce the probability that the
LLM spends its final `r` selections on low-value candidates. In pairwise form,
if routing raises the average value gap among candidates seen by the LLM from
`Delta_K` to `Delta_b`, and reduces selection noise from `sigma_LLM(K)` to
`sigma_LLM(b)`, then the pairwise ranking error changes from:

```text
1 - Phi( Delta_K / (sqrt(2) * sigma_LLM(K)) )
```

to:

```text
1 - Phi( Delta_b / (sqrt(2) * sigma_LLM(b)) )
```

Routing helps when:

```text
Delta_b / sigma_LLM(b) > Delta_K / sigma_LLM(K)
```

This is the cleanest theoretical statement for the small experiment:

```text
the router should improve downstream LLM attention if it increases the
signal-to-selection-noise ratio of the candidate set shown to the LLM.
```

Therefore:

```text
E[Capture(A_routed)] > E[Capture(A_full)]
```

is not assumed. It is the empirical implication tested by the listwise small
experiment.

## 8. Latency And Component-Cost Logic

If the LLM itself performs first-stage selection over all `K` candidates, the
selector cost is:

```text
Cost_full_selector = T_LLM(K)
```

If OL-Origin performs first-stage selection, the selector cost is:

```text
Cost_OL_selector = T_OL(K)
```

and the total routed system cost is:

```text
Cost_routed_total = T_OL(K) + T_LLM(b)
```

The component comparison is:

```text
T_LLM(K) / T_OL(K)
```

In the current small experiment with `K = 30`:

```text
T_OL(30) approximately 0.048 ms
T_LLM(30) approximately 1.22 s to 7.09 s, depending on backend
```

Thus the first-stage selector implemented by OL-Origin is orders of magnitude
cheaper than asking the tested LLMs to perform the same first-stage selection
directly.

The important latency claim is component-level:

```text
OL-Origin is cheap as a router.
```

It does not claim that every complete routed pipeline is always faster than
every complete full-LLM pipeline, because total wall-clock time also depends on
API backend, prompt construction, output decoding, batching, and rate limits.

## 9. Testable Claims And Experiment Mapping

The theory implies four empirical claims.

Claim 1: Social-media routing is an attention-allocation task.

```text
Metric: oracle-normalized future reach capture
Experiment: main early narrative alert benchmark
```

Claim 2: KOL originator role contains incremental information.

```text
Condition: I(Y_i ; O_{k(i)} | X_i) > 0
Experiment: no-OL controls, shuffled OL, follower replacement, raw OL,
            residualization checks
```

Claim 3: A structural router can enrich the candidate set shown to an LLM.

```text
Condition: P(i in A_e^* | i in B_e) > P(i in A_e^* | i in C_e)
Experiment: listwise small experiment, routed LLM vs full LLM
```

Claim 4: The router is a low-latency first-stage component.

```text
Condition: T_OL(K) << T_LLM(K)
Experiment: component latency analysis
```

These claims intentionally separate model quality from system cost. The paper's
argument is strongest when the same structural component both improves capture
and reduces first-stage selection latency.

## 10. Full Logical Chain

The paper's logic is:

```text
Observation:
  Large social-media streams are not automatically usable information for an
  agent. Directly passing many candidates to an LLM often fails to convert the
  larger pool into better first-stage attention allocation.

Diagnosis:
  The LLM is being asked to solve a noisy listwise routing problem before it can
  perform downstream reasoning.

Theoretical abstraction:
  Full-context LLM selection can be modeled as noisy random-utility ranking:
    tilde Y_i^LLM(K) = Y_i + epsilon_i(K)
  where effective selection noise need not decrease with candidate-pool size.

Structural discovery:
  Financial social media is actor-time-frame data. KOL streams contain a stable
  originator role O_k that has conditional information about future reach after
  controlling for ordinary origin-time context.

Model implication:
  If I(Y_i ; O_{k(i)} | X_i) > 0, expose this structure directly through a
  point-in-time selector rather than asking the LLM to rediscover it from raw
  context.

Model design:
  Fit a lightweight OL-Origin router using origin-time controls, residualized
  originator role, and role-context interactions.

Prediction:
  Routing improves downstream LLM use when it raises the candidate-set
  signal-to-selection-noise ratio:
    Delta_b / sigma_LLM(b) > Delta_K / sigma_LLM(K)

Validation:
  Main experiment tests alert quality.
  Latency analysis tests selector-component cost.
  Ablation tests whether the discovered originator structure is doing real work.
  Listwise small experiment tests whether routing improves downstream LLM
  attention allocation under large candidate pools.
```

The central theoretical claim is:

```text
The bottleneck is not a lack of social-media data. The bottleneck is the lack
of a point-in-time structure that tells the agent which social-media items
deserve expensive reasoning.
```

## 11. Literature Anchors

This document uses three existing theoretical and empirical anchors.

Random utility and noisy choice:

```text
Thurstone (1927), "A Law of Comparative Judgment"
McFadden (1974), "Conditional Logit Analysis of Qualitative Choice Behavior"
```

These justify modeling a selector as ranking latent value plus an effective
error term.

Rational inattention and costly information processing:

```text
Sims (2003), "Implications of Rational Inattention"
Mackowiak, Matejka, and Wiederholt (2021), "Rational Inattention: A Review"
```

These justify distinguishing available information from processed information.

Long-context and multi-document LLM limitations:

```text
Liu et al. (2024), "Lost in the Middle: How Language Models Use Long Contexts"
Levy et al. (2025), "More Documents, Same Length: Isolating the Challenge of
Multiple Documents in RAG"
```

These support the assumption that larger contexts and larger document/candidate
pools are not automatically converted into better model decisions.

Reference links:

```text
Lost in the Middle:
https://aclanthology.org/2024.tacl-1.9/

More Documents, Same Length:
https://aclanthology.org/2025.findings-emnlp.1064/

Rational Inattention review:
https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp2570~a3979fbfa5.en.pdf
```
