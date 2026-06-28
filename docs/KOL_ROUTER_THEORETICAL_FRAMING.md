# Theoretical Framing: From Social-Media Inefficiency To KOL-Origin Routing

This note formalizes the logic behind the KOL Origin-Aware Narrative Router.
The goal is not to prove from first principles that the empirical OL-Origin
feature must work. The goal is to state a clean theoretical framework that
explains why the observed KOL structure naturally implies a low-latency routing
model, and which empirical conditions the experiments are designed to test.

## 1. Agent Attention As A Budgeted Selection Problem

For each event-day `e`, the agent observes a pool of social-media narrative
origin candidates:

```text
C_e = {i = 1, ..., K_e}
```

Each candidate `i` has an unobserved future value:

```text
Y_i = future follower-weighted reach of candidate i
```

The agent cannot send every candidate to expensive downstream reasoning. It
must choose a small subset:

```text
A_e subset C_e,    |A_e| = r
```

The ideal decision problem is:

```text
max_A  E[ sum_{i in A_e} Y_i | I_e ] - lambda * Cost(A_e)
subject to |A_e| = r
```

where `I_e` is the point-in-time information set available before later follower
confirmation is observed.

This makes the task an attention-allocation problem rather than a direct return
prediction problem. The router is valuable if it improves which social-media
items enter memory, retrieval, LLM reasoning, or strategy research.

## 2. Why Full-Context LLM Selection Can Be Inefficient

A full-context LLM selector reads the entire candidate pool and chooses `r`
items:

```text
A_full = S_LLM(C_e)
```

This selector must solve several subproblems at once:

```text
semantic parsing
deduplication
origin detection
actor reliability inference
importance ranking
```

We can represent its internal candidate-value estimate as:

```text
Yhat_i^LLM(K) = Y_i + epsilon_i(K)
```

where `epsilon_i(K)` is selection noise induced by the complexity of reading and
ranking a candidate pool of size `K`.

The paper does not require the strong claim that more data always hurts. The
weaker and more realistic condition is:

```text
LLM selector efficiency does not necessarily improve with K.
```

Equivalently, larger candidate pools can increase the complexity of the first
selection step:

```text
Var[epsilon_i(K)] may increase with K
```

or at least may fail to decrease enough to exploit the extra candidates. The
measured object is the capture ratio:

```text
Capture_full(K)
  = E[sum_{i in S_LLM(C_e)} Y_i]
    / E[sum_{i in Top_r(C_e; Y)} Y_i]
```

where `Top_r(C_e; Y)` is the oracle top-`r` set under realized future reach.

The key issue is therefore not whether the LLM has access to more text, but
whether it converts the larger candidate pool into better selected candidates.

## 3. Social Media As Structured Actor-Time-Frame Data

A financial social-media observation is not only text. Each candidate can be
represented as:

```text
i = (text_i, frame_i, time_i, actor_i)
```

where:

```text
text_i   = what is being said
frame_i  = which semantic narrative the post belongs to
time_i   = when it appears inside the event
actor_i  = which KOL originated or amplified it
```

The pre-popularity setting removes later diffusion and follower confirmation
from the online information set. Thus the relevant question becomes:

```text
Which newly originated narrative frames should the agent attend to before
popularity is observable?
```

This reframes the modeling problem. The model should not only score text; it
should exploit the structure of who originated a frame and when.

## 4. Empirical Structure: Stable Originator Role

Let `k(i)` be the KOL who originates candidate `i`. Define a KOL-level
originator role:

```text
O_k = residualized stable originator trait of KOL k
```

In the implementation, `O_k` is estimated from pre-validation history using
event-order lead-lag behavior and then residualized against median UTC posting
hour and hour squared.

The structural condition needed for the router is:

```text
I(Y_i ; O_{k(i)} | X_i) > 0
```

where `X_i` includes ordinary origin-time controls such as follower visibility,
verified status, origin rank, elapsed time, semantic novelty, sentiment, and
historical activity.

In words:

```text
After controlling for ordinary context, the stable originator role contains
incremental information about future follower-weighted reach.
```

The empirical validation also needs falsification conditions:

```text
O_k is not merely follower scale
O_k is not merely posting frequency
O_k is not merely timezone or median posting hour
O_k is not merely news-reaction speed
O_k is not a bot/retweet artifact
O_k is not an ex-post popularity label
```

These are not assumed. They are tested through residualization, permutation,
shuffling, follower-replacement, and robustness checks.

## 5. Router Design Implied By The Structure

If `O_k` contains conditional information about future reach, then the first
selection stage should expose this structure directly instead of asking an LLM
to infer it from raw context each time.

For candidate `i`, define the point-in-time information set:

```text
I_i = {
  origin text and semantic novelty,
  origin time and within-event rank,
  originator identity k(i),
  pre-estimated originator role O_{k(i)},
  basic originator visibility and context
}
```

The OL-Origin router uses a linear score:

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
O_{k(i)} * visibility_i   = stable originator role under current audience scale
O_{k(i)} * novelty_i      = stable originator role under semantic novelty
```

The interaction terms are conceptually important. The model is not a global KOL
leaderboard:

```text
high O_k does not imply every post by k is important
```

Instead, the useful object is conditional:

```text
who originates + when it is originated + how novel/visible the originated frame is
```

## 6. Why Routing Can Improve Downstream LLM Use

The router selects a shortlist:

```text
B_e = Top_b({s_i : i in C_e}),    b < K
```

The downstream LLM then reranks only the shortlist:

```text
A_routed = S_LLM(B_e)
```

Routing improves downstream LLM use under two conditions.

First, the router enriches the shortlist:

```text
P(i in Top_r(C_e; Y) | i in B_e)
  >
P(i in Top_r(C_e; Y) | i in C_e)
```

Second, the LLM has lower effective selection noise on the smaller, enriched
candidate pool:

```text
Var[epsilon_i(b)] < Var[epsilon_i(K)]
```

Under these conditions:

```text
E[Capture_routed] > E[Capture_full]
```

This statement should be read as a conditional theoretical implication. The
main and small experiments test whether the enrichment and capture conditions
hold in the observed KOL data.

## 7. Latency And Component-Cost Logic

If the LLM itself performs first-stage selection over all `K` candidates, the
selector cost is:

```text
Cost_full_selector = T_LLM(K)
```

If OL-Origin performs first-stage selection, the selector cost is:

```text
Cost_OL_selector = T_OL(K)
```

and the routed downstream call costs:

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

## 8. Full Logical Chain

The paper's logic can be summarized as:

```text
Observation:
  Large social-media streams are expensive and inefficient for direct LLM
  consumption.

Diagnosis:
  Direct LLM consumption asks the model to solve first-stage routing before it
  can solve downstream reasoning.

Representation:
  Social-media observations should be represented as actor-time-frame data, not
  only as raw text.

Structural discovery:
  KOL streams contain a stable originator role O_k that is measurable before the
  validation period and is not reducible to follower count, posting frequency,
  timezone, or ex-post popularity.

Model implication:
  Use O_k as a point-in-time low-latency selector over newly originated
  semantic frames.

Model design:
  Fit a lightweight OL-Origin router with origin-time controls, residualized
  originator role, and role-context interactions.

Experimental validation:
  Main experiment tests early reach-alert quality.
  Latency analysis tests component cost.
  Ablation tests whether the discovered originator structure is doing real work.
  Listwise small experiment tests whether routing improves downstream LLM
  attention allocation in large candidate pools.
```

The central theoretical claim is therefore:

```text
The bottleneck is not the absence of social-media information, but the absence
of a point-in-time structure that tells the agent which social-media information
deserves expensive reasoning.
```
