# Empirical Discovery: KOL Originator Structure

This note documents the empirical structure that motivates the KOL
Origin-Aware Narrative Router. The purpose is to separate the data discovery
from the routing model. The router is not introduced first; it is derived from
a repeated empirical finding in large-scale financial KOL data:

```text
Financial KOL streams contain a stable, identity-driven originator role.
```

This originator role is point-in-time measurable, not reducible to follower
scale, not explained by posting hour or activity frequency, and useful as a
routing signal once coupled with origin-time context.

## 1. Data Scale

The discovery is not based on a hand-collected small sample. It is based on a
large financial KOL archive and a derived multi-year working panel.

```text
findata KOL archive:
  33.8M tweets
  3,712 KOL accounts
  2009-2026 coverage

paper working panel:
  459,472 tweets
  17 equity / ETF / crypto symbols
  AAPL MSFT NVDA TSLA AMZN META GOOGL AMD MSTR COIN HOOD PLTR SPY QQQ BTC ETH SOL

main origin-alert panel:
  12,822 origin candidates
  2,868 symbol-day events
  6,907 train candidates
  5,915 validation candidates

main ranking evaluation:
  785 validation events
  16 symbols

point-in-time OLtrait universe:
  106 KOLs with sufficient pre-2020 history
```

The full archive provides the broad source universe. The paper working panel is
a dense slice over highly followed equity, ETF, and crypto narratives. The
main validation uses only point-in-time information available before later
follower confirmation.

## 2. Representation: Actor-Time-Frame Data

The first empirical step is to stop treating financial social media as an
unordered text pile. Each candidate is represented as:

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

This representation makes the discovery possible. The central question becomes:

```text
Which KOLs systematically originate narrative frames before other KOLs attach
to them?
```

## 3. Event And Originator Measurement

An event is defined as:

```text
event = (symbol, UTC day) with enough distinct KOL participants
```

For each event, KOLs are ordered by the timestamp of their first post inside
the event:

```text
rank = order of each KOL's first post, rank 1 = earliest
```

For an event with `k` participating KOLs, the net-lead contribution of a KOL is:

```text
net_lead = k + 1 - 2 * rank
```

This gives positive contribution to early originators and negative
contribution to late participants. Across events:

```text
L_raw(kol) = average net_lead contribution across events
```

KOLs must have sufficient event participation to receive a stable estimate:

```text
minimum events for OL estimation: n >= 4
```

## 4. Timezone Confound And Residualized OLtrait

Raw lead-lag behavior is strongly confounded by posting schedule and timezone.
The raw score is highly correlated with median UTC posting hour:

```text
Spearman(L_raw, median UTC hour) = -0.768
```

Therefore the canonical originator trait is not raw lead score. The paper uses
a residualized originator trait:

```text
L_raw(k) = a + b1 * median_hour_k + b2 * median_hour_k^2 + residual_k
```

and defines:

```text
OLtrait_k = residual_k
```

Equivalently:

```text
O_k = residualized stable originator role of KOL k
```

This residualization is essential. It separates identity-driven origination
from schedule-driven early posting.

## 5. Structural Existence Tests

The first set of tests asks whether a lead-lag hierarchy exists at all.

```text
Lead-lag hierarchy:
  16/17 symbols have permutation p = 0.001
  COIN is the only non-significant symbol in the phase-0 test
```

The structure is not a follower-count effect:

```text
Spearman(OLtrait, followers) = -0.068 across 641 KOLs
```

The structure is not a posting-frequency effect:

```text
Spearman(OLtrait, log #events) = +0.02
Spearman(OLtrait, log #tweets) = +0.03
```

The structure is not merely fast reaction to large market-moving news:

```text
price-quiet days:
  pooled 5,959 events still show p = 0.001 by symbol

bottom absolute-return quartile days:
  17/17 symbols have p < 0.01
  pooled 3,179 events
```

These tests show that the originator axis is present even when obvious
market-news reaction channels are weakened.

## 6. Robustness And Falsification

The discovery was repeatedly tested against alternative explanations.

Timezone and schedule:

```text
raw lead vs median UTC hour:
  Spearman = -0.768

after residualization:
  OLtrait is designed to remove the schedule component
```

Activity frequency:

```text
Spearman(OLtrait, log #events) = +0.02
Spearman(OLtrait, log #tweets) = +0.03
```

Bot / retweet artifact:

```text
tweet_type distribution:
  original = 374,749
  quote    = 43,993
  reply    = 40,697
  retweet  = 33

original-tweet-only rerun:
  16/17 symbols remain significant at p < 0.01
```

Event-definition robustness:

```text
minimum KOLs per event = 3:
  16/17 symbols significant

minimum KOLs per event = 10:
  16/17 symbols significant
```

These checks indicate that the structure is not an artifact of a narrow event
definition, retweets, or account activity volume.

## 7. Stability Over Time

A useful routing trait must be stable enough to estimate before validation and
use later. The originator trait passes this test.

Using recent dense-period 6-month bins:

```text
raw lag-1 cross-bin Spearman:
  +0.669

timezone-residualized lag-1 cross-bin Spearman:
  +0.447
  n = 2,104
  p = 6e-104

timezone-residualized lag-2 cross-bin Spearman:
  +0.345
```

The shuffle null is near zero:

```text
shuffle null mean = +0.001
shuffle null 95th percentile = +0.037
```

This shows that the originator trait is persistent, not a one-period accident.

## 8. Identity Signal Versus Timezone Persistence

Residualization removes the main timezone component, but the stricter question
is whether persistence is driven by identity or by similar posting-hour groups.

The decisive comparison matches each KOL's future trait either to the same KOL
or to a different KOL with nearby median UTC posting hour.

```text
identity-matched lag-1 persistence:
  about +0.42 to +0.47

timezone-matched null:
  about +0.02 to +0.09

gap:
  about +0.38
```

This result holds across multiple timezone controls:

```text
global median-hour residualization
within-bin median-hour residualization
nonparametric median-hour decile adjustment
activity-volume double controls
```

Additional robustness:

```text
17 bin-pairs:
  median identity advantage = +0.405
  94% of pairs positive

3-month bins:
  identity persistence = +0.440
  timezone null = +0.003

12-month bins:
  identity persistence = +0.390
  timezone null = +0.020
```

The conclusion is that persistence is identity-driven rather than merely
schedule-driven.

## 9. Cross-Asset Consistency

The originator trait is not isolated to one symbol family. Splitting the 17
symbols into two groups gives:

```text
group A: 9 symbols
group B: 8 symbols

Spearman(OLtrait_groupA, OLtrait_groupB):
  +0.500
  p = 5e-19
  n = 279 KOLs
```

This supports treating originator role as a cross-asset KOL trait rather than
a symbol-specific artifact.

## 10. From Structure To Routing Target

The structural discovery alone is not yet the final model. The paper next asks
whether the discovered trait helps route newly originated semantic frames
before popularity is observable.

The routing setting is:

```text
semantic threshold = 0.55
origin window = first10
target = future follower-weighted Reach
validation events = 785
symbols = 16
```

The full router is a linear ridge model using:

```text
originator role
origin-time context controls
originator-role interactions
```

The discovered structure enters as:

```text
origin_ol
ol_x_visibility = origin_ol * origin_logfoll
ol_x_novelty    = origin_ol * novelty_global
```

The non-OL controls include:

```text
origin_logfoll
origin_verified
log_origin_rank
elapsed_hours
prior_frame_count
origin_stance
origin_stance_abs
novelty_global
novelty_event
hist_log_origin_count
hist_mean_log_adopt
hist_success_rate
```

These controls ensure that the OL channel is tested against account scale,
timing, sentiment, semantic novelty, and historical activity.

## 11. Routing Ablation Results

The ablation tests whether the discovered originator role is doing real work
inside the router.

| Variant | Role | NDCG@3 | Hit@1 | Mass@3 | JS ↓ |
|---|---|---:|---:|---:|---:|
| No-OL Strong | context controls only | 0.712 | 0.421 | 0.891 | 0.301 |
| OL Only | stable originator role only | 0.650 | 0.292 | 0.854 | 0.314 |
| **OL-Origin Full** | context + OLtrait + OL interactions | **0.755** | **0.493** | <u>0.901</u> | **0.263** |
| Shuffled OL-Origin | shuffled KOL-to-OL mapping | 0.724 | 0.459 | 0.892 | 0.295 |
| Follower Replacement | follower-scale interaction analogues | 0.717 | 0.431 | 0.895 | 0.296 |
| Raw OL-Origin | non-residualized raw OLtrait | <u>0.749</u> | <u>0.489</u> | **0.905** | <u>0.268</u> |

Symbol-balanced bootstrap comparisons:

| Comparison | ΔNDCG@3 | 90% CI | ΔHit@1 | 90% CI | JS improvement | 90% CI |
|---|---:|---:|---:|---:|---:|---:|
| Full - No-OL Strong | +0.043 | [+0.019, +0.069] | +0.072 | [+0.013, +0.156] | +0.039 | [+0.022, +0.058] |
| Full - OL Only | +0.105 | [+0.056, +0.149] | +0.201 | [+0.101, +0.308] | +0.052 | [+0.029, +0.076] |
| Full - Shuffled OL | +0.031 | [+0.009, +0.057] | +0.034 | [-0.014, +0.093] | +0.033 | [+0.017, +0.051] |
| Full - Follower Replacement | +0.038 | [+0.013, +0.067] | +0.063 | [+0.001, +0.134] | +0.033 | [+0.017, +0.053] |
| Full - Raw OL | +0.007 | [-0.009, +0.025] | +0.004 | [-0.027, +0.035] | +0.006 | [+0.001, +0.011] |

Interpretation:

```text
1. OLtrait alone is not sufficient.
2. Non-OL context alone is weaker than the full origin-aware router.
3. The true KOL-to-OLtrait mapping matters: shuffling the mapping reduces
   performance.
4. The result is not merely a large-account effect: follower replacement is
   weaker than OL-Origin Full.
5. Residualized OLtrait is cleaner for interpretation because it removes the
   posting-hour confound.
```

The useful signal is therefore conditional:

```text
who originates
+ when it is originated
+ how visible and novel the originated frame is
```

## 12. Main Discovery Claims

The empirical discovery can be summarized as five claims.

Claim 1: Financial KOL data are actor-time-frame data.

```text
The key observation is not only what is said, but who originates a frame and
when.
```

Claim 2: KOL streams contain a stable originator role.

```text
The role is measured by event-order lead-lag behavior and persists across time.
```

Claim 3: The role is not a simple surface proxy.

```text
It is not follower scale, not posting frequency, not timezone, not retweet
activity, and not news-reaction speed.
```

Claim 4: The role is point-in-time usable.

```text
It can be estimated from pre-validation history and used later without future
popularity leakage.
```

Claim 5: The role becomes useful when coupled with origin-time context.

```text
The full OL-Origin router beats context-only, OL-only, shuffled-OL, and
follower-replacement ablations.
```

## 13. Reproducibility Artifacts

Primary scripts and result files:

```text
experiments/socialenc/phase7_origin_alert.py
experiments/socialenc/phase7_origin_alert_result.json
experiments/socialenc/phase33_origin_alert_ablation.py
experiments/socialenc/phase33_origin_alert_ablation_result.json
```

Supporting structure-discovery scripts are preserved in the broader working
history and summarized in `docs/KOL_ORIGINATION_STRUCTURE.md`.

## 14. Relationship To The Theory Document

This document establishes the empirical premise:

```text
O_k is a real, stable, point-in-time originator structure.
```

The theory document then uses this premise to justify the router:

```text
docs/KOL_ROUTER_THEORETICAL_FRAMING.md
```

In short:

```text
Empirical discovery:
  KOL streams contain stable originator structure.

Theoretical implication:
  a first-stage agent router should expose this structure before expensive LLM
  reasoning.
```
