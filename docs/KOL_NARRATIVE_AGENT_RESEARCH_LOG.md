# KOL Origin-Aware Narrative Router for Financial Agents

Last updated: 2026-06-28

This is the clean working document for the current paper direction. It is no
longer a running experiment log. The main text only keeps the current research
narrative, adopted experimental setup, final core results, necessary diagnostics,
and reproducibility artifacts. Deprecated intermediate experiments are excluded
from the main narrative.

## 0. Introduction Example

```text
Recent advances in agentic quantitative trading have enabled increasingly automated pipelines for market research, data extraction, feature construction, model selection, and portfolio decision-making. A key motivation behind these systems is their ability to acquire and process information at a scale and speed far beyond human analysts. However, this capability also creates a new bottleneck: when high-volume, time-sensitive social media data are directly fed into large agentic systems, the resulting pipeline often becomes both slow and inefficient. The model must spend substantial computation and token budget processing noisy streams, while many actionable signals may already be absorbed by the market before the system reaches a trading decision.
```

```text
This issue is especially severe for financial social media. Unlike structured market data, social media signals are sparse, noisy, and highly dependent on who first originates a narrative, when it appears, and whether it is likely to diffuse before broad follower confirmation becomes visible. Treating all posts as raw textual evidence can dilute positive and negative signals, causing the agent to produce generic conclusions despite consuming more data, more computation, and more time.
```

```text
Motivated by this observation, we study the entry point of social media information in agentic trading systems. We find that financial KOL data exhibit a stable originator structure: certain users are systematically more informative when they originate new narratives, and this signal is not reducible to follower count or textual salience alone. Based on this structure, we propose a KOL-origin-aware narrative router, a lightweight linear routing layer that identifies which newly originated financial narratives should receive the agent’s attention at the moment they first appear. Rather than replacing downstream reasoning models, the router allocates attention before expensive inference, memory writing, retrieval, or strategy research is triggered.
```

```text
Empirically, this simple routing layer achieves competitive or superior early narrative alert quality compared with substantially more complex text-encoder and LLM-based baselines, while requiring almost no online token consumption and incurring orders-of-magnitude lower latency. These results suggest that effective use of social media in agentic trading does not only require larger models or more data; it also requires a point-in-time mechanism for deciding which signals deserve to enter the agent’s reasoning pipeline in the first place.
```

## 1. Paper Motivation

Background:

```text
Agentic quantitative trading systems are becoming increasingly capable of
automating information acquisition, data ETL, feature construction, strategy
research, model fitting, and trading decisions.
```

One reason these systems are attractive is their ability to process information
far faster than human quantitative researchers. In principle, agents can compress
the full pipeline of information acquisition, data cleaning, feature engineering,
model building, and strategy optimization into a much shorter cycle.

However, social-media data remains a weak entry point for many agentic trading
frameworks:

```text
They often treat social media streams as large unstructured text piles and feed
too much raw content directly into LLMs or text encoders.
```

This creates two practical problems.

First, latency and throughput become binding:

```text
Social-media signals are highly time-sensitive. If an agent spends too much
time reading, summarizing, embedding, or querying API models before routing the
event, the practical value of the signal may already be gone.
```

Second, signal dilution becomes severe:

```text
More social-media data does not necessarily mean better decisions. Mixing many
weak, noisy, repeated, or contradictory posts can flatten sharp signals and
push the model toward generic conclusions.
```

The paper therefore does not ask agents to read more tweets. Instead, it adds a
low-latency and selective routing layer at the social-media entry point: when a
financial narrative first appears, the system decides whether it deserves
expensive downstream agent attention.

## 2. Current Claim

Core claim:

```text
Stable KOL originator structure can serve as a low-latency routing signal that
helps financial AI agents identify high-reach narrative origins before
popularity is observable.
```

Method position:

```text
KOL Origin-Aware Narrative Router
```

This is not a large model that replaces the downstream trading agent, nor is it
a direct return-prediction alpha model. It is a point-in-time social-media entry
component:

```text
KOL stream -> semantic origin detection -> origin alert ranking ->
agent watchlist / memory write / RAG routing -> downstream research or trading agent
```

Core contributions:

```text
Find: financial KOL streams contain stable originator structure.
Build: a simple linear origin-aware router using this structure.
Show: it can route early high-reach narratives faster and more accurately than
      heavier text encoders and LLM scorers under the pre-popularity setting.
```

The main task is therefore not direct return prediction, and it is not replacing
engagement statistics after a topic has already become popular. The main task is
to decide whether a newly originated narrative deserves early agent attention
before follower confirmation or diffusion evidence is visible.

### 2.1 Empirical Foundation: KOL Origination Structure

The empirical structure comes before the router. The key discovery is not that
a linear model works; it is that the KOL stream contains a stable originator
axis that can be measured before the validation period and then used as
point-in-time routing information.

Data scale:

```text
findata KOL archive: 33.8M tweets, 3,712 KOL accounts, 2009-2026
paper working panel: 459,472 tweets over 17 equity/crypto symbols
main origin-alert panel: 12,822 origin candidates from 2,868 symbol-day events
train/validation candidates: 6,907 train, 5,915 validation
main ranking evaluation: 785 validation events across 16 symbols
point-in-time OLtrait universe: 106 KOLs with sufficient pre-2020 history
```

The working panel is not a hand-collected small sample. It is a dense
multi-year slice of the findata KOL archive, selected to cover highly followed
equity, ETF, and crypto narratives while preserving point-in-time history for
trait estimation and validation.

Origination is measured from event order:

```text
event = (symbol, UTC day) with enough distinct KOL participants
rank  = order of each KOL's first post inside the event
net-lead contribution = k + 1 - 2 * rank
raw lead score = average net-lead contribution across events
OLtrait = raw lead score residualized on median UTC posting hour and hour^2
```

The residualization is essential. Raw lead-lag behavior is heavily entangled
with timezone and posting schedule:

```text
Spearman(raw lead, median UTC hour) = -0.768
```

After removing the schedule component, the remaining originator trait has the
properties needed for a paper-level mechanism:

| Fact | Evidence |
|---|---|
| Lead-lag hierarchy exists | 16/17 symbols have permutation p=0.001 in the phase-0 test |
| Not follower scale | Spearman(OLtrait, followers) = -0.068 across 641 KOLs |
| Not posting frequency | Spearman with log event count/tweet count is about +0.02/+0.03 |
| Not merely news-reaction speed | price-quiet days still show p=0.001; bottom absolute-return quartile gives 17/17 p<0.01 |
| Not bot/retweet artifact | original-tweet-only rerun still gives 16/17 significant symbols |
| Stable identity signal | lag-1 residualized cross-bin Spearman about +0.447; lag-2 about +0.345 |
| Not timezone persistence | identity-matched persistence about +0.42 to +0.47 vs timezone-matched null about +0.02 to +0.09 |
| Cross-asset trait | group-A vs group-B symbol split Spearman about +0.500 |

These facts justify treating `origin_ol` as a stable KOL role rather than as a
proxy for account size, posting hour, activity level, or ex-post popularity.
The current main experiment then asks whether this discovered role helps route
newly originated semantic frames before popularity is observable.

## 3. Method Definition

The method can be defined as:

```text
An originator-aware, low-latency narrative routing layer for financial agents.
```

Online inputs:

```text
new KOL tweet
current semantic frame state
pre-estimated OLtrait of the originating KOL
basic origin-time context
```

Online output:

```text
a ranking score for whether the newly originated semantic frame should enter
the agent's watchlist / memory / RAG queue
```

Why this matches agentic trading systems:

```text
The router is cheap enough to run before expensive reasoning. It does not ask
the agent to read every tweet deeply. It decides which newly originated frames
deserve scarce downstream reasoning and execution resources.
```

### 3.1 Methodology Narrative Chain

The method starts from the structure of the KOL data rather than from a large
language model. A financial KOL stream is not just a pile of independent text
documents. Each observation carries three distinct signals:

```text
text:  what narrative is being expressed
time:  whether this expression appears early in the narrative's life
actor: which KOL originated the expression
```

The pre-popularity setting removes follower confirmation and later diffusion
evidence from the online information set. The relevant online question is
therefore:

```text
Given a newly originated narrative frame, should the downstream agent spend
scarce attention, memory, retrieval, or research budget on it now?
```

The method follows this chain:

```text
Raw KOL tweet stream
-> semantic origin detection
-> point-in-time originator trait estimation
-> origin-context interaction model
-> low-latency agent attention router
```

First, the raw tweet stream is converted into semantic origin candidates. Tweets
within the same `(symbol, day)` event are grouped by semantic similarity, and the
router only scores early origin candidates rather than every repeated post. This
turns the problem from generic text scoring into an early attention-allocation
task.

Second, the method estimates a KOL-level stable originator trait:

```text
o_k = residualized stable originator propensity for KOL k
```

This trait is estimated only from the pre-2020 history. It measures whether a
KOL has historically acted as an originator of narratives that later diffuse,
and is residualized against mechanical timing patterns such as median posting
hour. It is therefore not intended to be a proxy for follower count, posting
frequency, or timezone.

Third, the KOL trait is combined with current origin-time context. A strong
originator is not assumed to make every post important. The signal matters when
the current narrative origin is also visible, novel, or otherwise relevant at
the moment it appears. The router therefore uses both main effects and
interactions:

```text
s_i = beta_0
    + beta_c' c_i
    + beta_o o_{k(i)}
    + beta_v (o_{k(i)} * visibility_i)
    + beta_n (o_{k(i)} * novelty_i)
```

where:

```text
i              = newly originated narrative frame
k(i)           = originating KOL
c_i            = origin-time controls
o_{k(i)}       = stable originator trait of the originating KOL
visibility_i   = originator visibility, such as log followers
novelty_i      = semantic novelty of the origin frame
s_i            = downstream attention-routing score
```

The fitted router is intentionally lightweight:

```text
theta_hat = argmin_theta sum_i (y_i - s_i)^2 + lambda ||theta||_2^2
```

The target `y_i` is future follower-weighted reach during training. At validation
time, the router only observes point-in-time origin text, origin-time context,
and the pre-estimated KOL trait. The output is a ranking score for deciding
which emerging narratives enter the downstream agent's watchlist, memory, RAG
queue, or research workflow.

The methodological contribution is therefore not a complex neural architecture.
It is the conversion of noisy KOL social-media data into a point-in-time
origin-aware routing problem, and the use of a residualized KOL role signal as a
cheap first-stage attention allocator before expensive text models or agentic
reasoning are invoked.

This also defines the downstream usage model. In an agentic trading system, the
router is not only evaluated as a standalone ranker inside isolated events. It
can also be used as a budgeted queue controller: among all newly originated
narratives arriving during a trading day, only a small number are allowed to
trigger memory writes, RAG retrieval, deeper LLM scoring, or strategy research.
This budgeted interpretation is the direct bridge between the KOL data
structure and the agent pipeline described in the introduction.

## 4. Data And Point-In-Time Setup

Asset universe:

```text
AAPL, MSFT, NVDA, TSLA, AMZN, META, GOOGL, AMD, MSTR,
COIN, HOOD, PLTR, SPY, QQQ, BTC, ETH, SOL
```

Data files:

```text
data/socialenc/
{SYM}.jsonl      tweet / KOL metadata
{SYM}.npz        MiniLM embeddings, 384 dimensions
```

Time split:

```text
OLtrait estimation: before 2020-01-01
model train:        2020-01-01 to 2021-06-01
validation:         2021-06-01 to 2022-06-01
```

Event definition:

```text
event = (symbol, UTC day)
within each event, each KOL contributes only the first tweet
```

Stable originator trait:

```text
n_ol_kols = 106
OLtrait is estimated only from pre-2020 data
OLtrait is residualized against median UTC hour to reduce timezone/hour confounding
```

## 5. Semantic Frame Construction

For each `(symbol, UTC-day)` event:

1. Sort each KOL's first tweet by timestamp.
2. Use MiniLM embeddings to construct semantic frames online.
3. A new tweet joins an existing frame if its maximum cosine similarity to that
   frame exceeds the semantic threshold.
4. Otherwise, it creates a new frame.

Current main setting:

```text
origin window = first10
semantic threshold = 0.55
```

Threshold interpretation:

```text
0.35: extremely loose boundary; frame definition collapses
0.45: loose stress test
0.50: low-threshold supplement
0.55: main coherent early-frame setting
0.65: narrow robustness setting
0.75: extremely narrow, text-encoder-favorable boundary
```

## 6. Main Experiment: Pre-Popularity Origin Alert

Task:

```text
A semantic frame has just been originated by one KOL.
No follower confirmation, retweet/like growth, or later KOL adoption is visible.
The agent ranks newly originated frames by future follower-weighted Reach.
```

This experiment directly matches the paper's core application:

```text
The router acts before popularity exists. Once popularity is visible, simple
engagement and diffusion statistics are already strong; the hard problem is
early attention allocation.
```

Why Reach is the main target:

```text
Reach measures whether the originated frame will later be amplified by
high-visibility KOLs. This is closer to agent attention allocation than raw
adoption count.
```

Main coverage:

```text
validation events = 785
symbols = 16
evaluation = event-level candidate ranking
metrics = symbol-balanced validation metrics
```

Metrics:

```text
NDCG@3: top-3 watchlist ranking quality
Hit@1: top-1 alert hit rate
Mass@3: future reach mass captured by top-3
JS: predicted distribution vs realized reach distribution; lower is better
```

## 7. Main Result: Reach Alert At Threshold 0.55

All numeric columns are measured outputs or direct calculations from measured
outputs. `Delta` is relative to `No-OL Strong`. `Input Len` and `Latency` are
measured at the online scoring stage and exclude the upstream frame construction
shared by all methods.

| Family | Method | Events | Symbols | NDCG@3 | Hit@1 | Mass@3 | JS ↓ | ΔNDCG | ΔHit | Input Len | Latency ms/q |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Scale | Follower | 785 | 16 | 0.724 | 0.477 | 0.870 | 0.292 | +0.012 | +0.056 | 0.0 | 0.002 |
| Scale | Visibility | 785 | 16 | 0.723 | 0.477 | 0.870 | 0.292 | +0.011 | +0.056 | 0.0 | 0.002 |
| Context | Rank/Time | 785 | 16 | 0.678 | 0.372 | 0.872 | 0.320 | -0.034 | -0.049 | 0.0 | 0.002 |
| Context | Sentiment | 785 | 16 | 0.687 | 0.423 | 0.841 | 0.314 | -0.025 | +0.002 | 0.0 | 0.002 |
| Context | Novelty | 785 | 16 | 0.673 | 0.362 | 0.857 | 0.328 | -0.039 | -0.059 | 0.0 | 0.002 |
| Context | History | 785 | 16 | 0.632 | 0.339 | 0.811 | 0.323 | -0.079 | -0.083 | 0.0 | 0.002 |
| Context | No-OL Strong | 785 | 16 | 0.712 | 0.421 | 0.891 | 0.301 | +0.000 | +0.000 | 0.0 | 0.003 |
| Origin Role | OL Only | 785 | 16 | 0.650 | 0.292 | 0.854 | 0.314 | -0.062 | -0.129 | 0.0 | 0.002 |
| Origin Role | **OL-Origin** | 785 | 16 | **0.755** | **0.493** | **0.901** | **0.263** | **+0.043** | **+0.072** | 0.0 | 0.003 |
| Surface Text | Symbol one-hot | 785 | 16 | 0.678 | 0.372 | 0.872 | 0.319 | -0.034 | -0.049 | 0.0 | 0.001 |
| Surface Text | Text surface | 785 | 16 | 0.741 | 0.463 | 0.896 | 0.293 | +0.029 | +0.042 | 28.8 | 0.041 |
| Surface Text | Symbol + surface | 785 | 16 | 0.735 | 0.462 | 0.886 | 0.288 | +0.023 | +0.041 | 28.8 | 0.043 |
| Text Encoder | BERT-origin text | 785 | 16 | 0.745 | 0.440 | <u>0.901</u> | 0.298 | +0.033 | +0.019 | 51.6 | 0.645 |
| Text Encoder | FinBERT-origin text | 785 | 16 | 0.742 | 0.447 | 0.896 | 0.295 | +0.030 | +0.026 | 51.6 | 0.534 |
| Text Encoder | Qwen3-4B-origin text | 785 | 16 | <u>0.750</u> | 0.474 | 0.900 | <u>0.279</u> | +0.038 | +0.053 | 59.3 | 11.862 |
| Text Encoder | BGE-origin text | 785 | 16 | 0.742 | <u>0.476</u> | 0.888 | 0.287 | +0.030 | +0.054 | 51.6 | 0.452 |
| Text Encoder | E5-Mistral-7B-origin text | 785 | 16 | 0.684 | 0.385 | 0.863 | 0.379 | -0.028 | -0.037 | 65.2 | 25.994 |
| Local LLM | Llama3.1-8B | 785 | 16 | 0.619 | 0.291 | 0.827 | 0.474 | -0.093 | -0.131 | 309.7 | 172.487 |
| Local LLM | Qwen2.5-7B | 785 | 16 | 0.612 | 0.273 | 0.818 | 0.493 | -0.100 | -0.149 | 329.0 | 190.410 |
| Commercial API | GPT Chat Latest | 785 | 16 | 0.742 | 0.480 | 0.897 | 0.347 | +0.030 | +0.059 | 178.0 | 227.281 |
| Commercial API | Claude Sonnet 4.5 | 785 | 16 | 0.733 | 0.442 | 0.897 | 0.343 | +0.021 | +0.021 | 197.3 | 496.733 |
| Commercial API | Gemini 2.5 Flash | 759 | 16 | 0.711 | 0.394 | 0.882 | 0.396 | -0.001 | -0.028 | 198.5 | 195.458 |
| Commercial API | DeepSeek V4 Flash | 753 | 16 | 0.707 | 0.410 | 0.885 | 0.394 | -0.005 | -0.011 | 182.2 | 600.658 |
| Commercial API | Qwen3.7 Plus | 785 | 16 | 0.718 | 0.477 | 0.863 | 0.367 | +0.006 | +0.056 | 199.8 | 751.179 |

Table notes:

```text
Input Len:
  scalar rows: no text-token input
  surface rows: mean words
  encoder rows: mean tokenizer tokens
  LLM/API rows: mean prompt tokens

Latency:
  scalar/surface rows: measured scoring time
  encoder/local LLM rows: batch-amortized measured scoring time
  commercial API rows: batch-amortized API wall-clock time, batch size 10

New encoder probes:
  Qwen3-4B quality metrics come from Phase39; cost metrics come from Phase42.
  E5-Mistral-7B quality metrics come from Phase41; cost metrics come from Phase42.
  Phase42 reuses the Phase31 validation sample construction and runs large
  encoders in separate processes to avoid GPU memory carryover.
```

Main interpretation:

```text
At the main 0.55 threshold, OL-Origin remains the best method on Reach NDCG@3,
Reach Hit@1, and JS after adding the Qwen3/E5-Mistral encoder probes. Qwen3-4B
is the SOTA embedding row and is the strongest text encoder on NDCG@3 and JS
point estimate; under the current text-encoder roster, BGE has the strongest
Hit@1 point estimate and BERT has the strongest Mass@3 point estimate. OL-Origin
still gives better top-rank alert quality and distributional calibration with
near-zero online cost.

Uncertainty for the main table is reported in the statistical support section.
Method-level bootstrap support for Qwen3-4B and E5-Mistral is included below
from their Phase39/Phase41 probes; family-average text-encoder support remains a
legacy phase28 diagnostic until refreshed for the current roster. The robust
claim is that OL-Origin is statistically better than the strong non-OL structural
model and competitive with the best text encoder at much lower latency.
```

## 8. Latency Analysis: Why Low-Cost Routing Matters

Latency is not merely an engineering detail; it is part of the task definition:

```text
Origin alerts are useful only if they arrive before the narrative is already
absorbed by the market or by competing agents.
```

In an agentic trading pipeline, the social-media entry layer usually sits before
expensive reasoning. If the first routing stage already requires long prompts,
embedding models, local LLMs, or commercial API calls, the system faces two
problems:

```text
1. routing latency increases before the agent can even decide whether to care;
2. scarce downstream reasoning budget is spent on many low-value events.
```

The advantages of OL-Origin are:

```text
zero prompt tokens
no online text-model inference
near-zero scoring latency
point-in-time availability before follower confirmation
```

Representative online scoring latency from the main 0.55 Reach-alert table:

| Method | Family | Input Len | Latency ms/q | Multiple vs OL-Origin |
|---|---|---:|---:|---:|
| OL-Origin | Origin Role | 0.0 | 0.003 | 1.0x |
| Text surface | Surface Text | 28.8 | 0.041 | 13.7x |
| BGE-origin text | Text Encoder | 51.6 | 0.452 | 150.7x |
| BERT-origin text | Text Encoder | 51.6 | 0.645 | 215.0x |
| Qwen3-4B-origin text | Text Encoder | 59.3 | 11.862 | 3954.0x |
| E5-Mistral-7B-origin text | Text Encoder | 65.2 | 25.994 | 8664.7x |
| Llama3.1-8B | Local LLM | 309.7 | 172.487 | 57495.7x |
| GPT Chat Latest | Commercial API | 178.0 | 227.281 | 75760.3x |
| DeepSeek V4 Flash | Commercial API | 182.2 | 600.658 | 200219.3x |
| Qwen3.7 Plus | Commercial API | 199.8 | 751.179 | 250393.0x |

Latency scope:

```text
The main-table latency column measures online scoring latency for each baseline
after shared frame construction. It is not a claim about end-to-end trading
latency, and commercial API rows are backend-dependent batch-amortized
measurements. The relevant comparison is whether the first social-media routing
stage requires text-model or API inference before deciding whether the agent
should pay downstream reasoning cost.
```

This should be written as part of the method's value, not merely as a runtime
appendix:

```text
The method is not merely cheaper after achieving similar performance; its low
latency is what makes it suitable for the pre-popularity social-media routing
problem faced by trading agents.
```

## 9. Small Experiment: Listwise LLM Routing Under Large Candidate Pools

Setup:

```text
Decision unit: same-day cross-asset validation candidate pool
Eligible days: 28 validation days with at least 30 candidates
Candidate sizes: K = 10, 20, 30
Router shortlists: b = 10, 20 from the K=30 pool
Final LLM selection: r = 3
Label: future follower-weighted log reach
Metric: selected future log reach / oracle top-r future log reach
LLM mode: listwise reranking and attention allocation
Note: K=30 is used because only one validation day has at least 50 candidates.
```

Full LLM candidate-size sweep, fixed Kmax-oracle capture:

| LLM | Candidate pool K | Capture |
|---|---:|---:|
| DeepSeek V4 Flash | 10 | 0.272 |
| DeepSeek V4 Flash | 20 | 0.227 |
| DeepSeek V4 Flash | 30 | 0.184 |
| Gemini 2.5 Flash Lite | 10 | 0.206 |
| Gemini 2.5 Flash Lite | 20 | 0.259 |
| Gemini 2.5 Flash Lite | 30 | 0.227 |
| Claude Sonnet 4.6 | 10 | 0.214 |
| Claude Sonnet 4.6 | 20 | 0.271 |
| Claude Sonnet 4.6 | 30 | 0.279 |
| GPT-5.4 | 10 | 0.252 |
| GPT-5.4 | 20 | 0.262 |
| GPT-5.4 | 30 | 0.308 |

Large-pool routed LLM results, fixed Kmax-oracle capture:

| LLM | Router / selector | Candidates sent to LLM | Selector latency | Capture |
|---|---|---:|---:|---:|
| DeepSeek V4 Flash | Full LLM | 30 | 2.270000 s | 0.184 |
| DeepSeek V4 Flash | Random | 10 | ~0 s | 0.177 |
| DeepSeek V4 Flash | Random | 20 | ~0 s | 0.218 |
| DeepSeek V4 Flash | Follower | 10 | 0.000012 s | 0.320 |
| DeepSeek V4 Flash | Follower | 20 | 0.000012 s | 0.266 |
| DeepSeek V4 Flash | No-OL Strong | 10 | 0.000039 s | 0.309 |
| DeepSeek V4 Flash | No-OL Strong | 20 | 0.000039 s | 0.305 |
| DeepSeek V4 Flash | OL-Origin | 10 | 0.000048 s | 0.315 |
| DeepSeek V4 Flash | OL-Origin | 20 | 0.000048 s | 0.310 |
| DeepSeek V4 Flash | Qwen3-4B readout | 10 | 0.355860 s | 0.300 |
| DeepSeek V4 Flash | Qwen3-4B readout | 20 | 0.355860 s | 0.241 |
| Gemini 2.5 Flash Lite | Full LLM | 30 | 1.220000 s | 0.227 |
| Gemini 2.5 Flash Lite | Random | 10 | ~0 s | 0.143 |
| Gemini 2.5 Flash Lite | Random | 20 | ~0 s | 0.207 |
| Gemini 2.5 Flash Lite | Follower | 10 | 0.000012 s | 0.262 |
| Gemini 2.5 Flash Lite | Follower | 20 | 0.000012 s | 0.198 |
| Gemini 2.5 Flash Lite | No-OL Strong | 10 | 0.000039 s | 0.266 |
| Gemini 2.5 Flash Lite | No-OL Strong | 20 | 0.000039 s | 0.284 |
| Gemini 2.5 Flash Lite | OL-Origin | 10 | 0.000048 s | 0.320 |
| Gemini 2.5 Flash Lite | OL-Origin | 20 | 0.000048 s | 0.300 |
| Gemini 2.5 Flash Lite | Qwen3-4B readout | 10 | 0.355860 s | 0.307 |
| Gemini 2.5 Flash Lite | Qwen3-4B readout | 20 | 0.355860 s | 0.231 |
| Claude Sonnet 4.6 | Full LLM | 30 | 7.090000 s | 0.279 |
| Claude Sonnet 4.6 | Random | 10 | ~0 s | 0.215 |
| Claude Sonnet 4.6 | Random | 20 | ~0 s | 0.224 |
| Claude Sonnet 4.6 | Follower | 10 | 0.000012 s | 0.304 |
| Claude Sonnet 4.6 | Follower | 20 | 0.000012 s | 0.326 |
| Claude Sonnet 4.6 | No-OL Strong | 10 | 0.000039 s | 0.330 |
| Claude Sonnet 4.6 | No-OL Strong | 20 | 0.000039 s | 0.367 |
| Claude Sonnet 4.6 | OL-Origin | 10 | 0.000048 s | 0.382 |
| Claude Sonnet 4.6 | OL-Origin | 20 | 0.000048 s | 0.350 |
| Claude Sonnet 4.6 | Qwen3-4B readout | 10 | 0.355860 s | 0.379 |
| Claude Sonnet 4.6 | Qwen3-4B readout | 20 | 0.355860 s | 0.327 |
| GPT-5.4 | Full LLM | 30 | 2.280000 s | 0.308 |
| GPT-5.4 | Random | 10 | ~0 s | 0.178 |
| GPT-5.4 | Random | 20 | ~0 s | 0.219 |
| GPT-5.4 | Follower | 10 | 0.000012 s | 0.362 |
| GPT-5.4 | Follower | 20 | 0.000012 s | 0.361 |
| GPT-5.4 | No-OL Strong | 10 | 0.000039 s | 0.350 |
| GPT-5.4 | No-OL Strong | 20 | 0.000039 s | 0.344 |
| GPT-5.4 | OL-Origin | 10 | 0.000048 s | 0.364 |
| GPT-5.4 | OL-Origin | 20 | 0.000048 s | 0.369 |
| GPT-5.4 | Qwen3-4B readout | 10 | 0.355860 s | 0.373 |
| GPT-5.4 | Qwen3-4B readout | 20 | 0.355860 s | 0.354 |

Selector-component latency:

```text
Scope: latency of the component that performs the first selection over the K=30
candidate pool.

For local routers, this is direct measured local scoring latency over K=30.
For Full LLM, there is no separate local router; the LLM itself is the selector.
Therefore the Full LLM selector latency is inferred as:

  Full K=30 listwise latency - local router latency
  = Full K=30 listwise latency - 0
  = Full K=30 listwise latency

This is the relevant comparison for the entry-layer decision: using OL/Follower/
Qwen as the selector versus asking the LLM itself to read all K=30 candidates.
```

| LLM backend | Full LLM selector, K=30 | Follower selector, K=30 | No-OL selector, K=30 | OL-Origin selector, K=30 | Qwen3-4B selector, K=30 | Full / OL |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek V4 Flash | 2.270 s | 0.000012 s | 0.000039 s | 0.000048 s | 0.355860 s | 47291.7x |
| Gemini 2.5 Flash Lite | 1.220 s | 0.000012 s | 0.000039 s | 0.000048 s | 0.355860 s | 25416.7x |
| Claude Sonnet 4.6 | 7.090 s | 0.000012 s | 0.000039 s | 0.000048 s | 0.355860 s | 147708.3x |
| GPT-5.4 | 2.280 s | 0.000012 s | 0.000039 s | 0.000048 s | 0.355860 s | 47500.0x |

Operational reading:

```text
If the first selector is Full LLM, the selector itself costs seconds per
day-level K=30 decision. If the first selector is OL-Origin, the selector costs
about 0.000048 seconds. Thus the OL selector is roughly 25,000x to 148,000x
faster than using the tested LLMs themselves as the first-stage selector, before
considering the later downstream reasoning call.
```

Routing-component latency:

```text
Scope: router-side local scoring only.
For routed policies, the component scores the full K=30 candidate pool before
shortlisting b=10 or b=20 for the downstream LLM.
Excluded: downstream LLM API latency, prompt construction, output decoding, and
LLM token count. Those are framework/backend costs rather than router-component
costs.
```

| Router component | Local latency / candidate | K=10 scoring | K=20 scoring | K=30 scoring | vs OL K=30 | Notes |
|---|---:|---:|---:|---:|---:|---|
| Follower | 0.0004 ms | 0.004 ms | 0.008 ms | 0.012 ms | 0.25x | scalar popularity baseline |
| No-OL Strong | 0.0013 ms | 0.013 ms | 0.026 ms | 0.039 ms | 0.81x | scalar structural baseline without OL |
| OL-Origin | 0.0016 ms | 0.016 ms | 0.032 ms | 0.048 ms | 1.00x | proposed scalar router |
| Qwen3-4B readout | 11.862 ms | 118.620 ms | 237.240 ms | 355.860 ms | 7413.75x | supervised embedding readout |

Latency interpretation:

```text
OL-Origin adds essentially no operational delay at the routing stage: scoring
the entire K=30 pool costs about 0.048 ms before the LLM is called. This is
orders of magnitude below the embedding-readout router while still producing
top-ranked or second-ranked routed LLM performance across tested LLM backends.
```

Run status:

| Phase | LLM | New API calls | Parse failures | Artifacts |
|---|---|---:|---:|---|
| Phase50 | DeepSeek V4 Flash | 303 + 56 | 0 | `phase50_deepseek_listwise_dilution_*`, `phase50_deepseek_listwise_cache.jsonl` |
| Phase53 | Claude Sonnet 4.6 | 361 | 0 | `phase53_claude_sonnet_4_6_listwise_dilution_*`, `phase53_claude_sonnet_4_6_listwise_cache.jsonl` |
| Phase54 | GPT-5.4 | 361 | 0 | `phase54_gpt_5_4_listwise_dilution_*`, `phase54_gpt_5_4_listwise_cache.jsonl` |
| Phase55 | Gemini 2.5 Flash Lite | 361 | 0 | `phase55_gemini_2_5_flash_lite_listwise_dilution_*`, `phase55_gemini_2_5_flash_lite_listwise_cache.jsonl` |

## 10. Statistical Support

This section is part of the main-table experiment, not the ablation study. The
main table reports point estimates; this table reports paired statistical
support for those main-table comparisons.

We use symbol-balanced bootstrap so high-volume assets do not dominate the
conclusion. All deltas are `OL-Origin - baseline`; for JS, the value is
`baseline JS - OL-Origin JS`, so positive is better. The reported p-value is a
one-sided diagnostic p-value inferred from the bootstrap delta distribution.

| Family | Baseline | ΔNDCG@3 [90% CI], p | ΔHit@1 [90% CI], p | ΔMass@3 [90% CI], p | JS improvement [90% CI], p |
|---|---|---:|---:|---:|---:|
| Context | No-OL Strong | +0.043 [+0.019, +0.068], p=0.002 | +0.072 [+0.013, +0.153], p=0.045 | +0.011 [-0.007, +0.033], p=0.191 | +0.039 [+0.021, +0.059], p<0.001 |
| Scale | Follower | +0.032 [-0.022, +0.087], p=0.169 | +0.016 [-0.086, +0.120], p=0.399 | +0.032 [-0.017, +0.084], p=0.153 | +0.030 [+0.005, +0.054], p=0.023 |
| Surface Text | Text surface | +0.014 [-0.044, +0.061], p=0.326 | +0.030 [-0.076, +0.135], p=0.320 | +0.006 [-0.035, +0.045], p=0.408 | +0.030 [-0.013, +0.078], p=0.138 |
| Surface Text | Symbol + surface | +0.020 [-0.030, +0.077], p=0.271 | +0.031 [-0.089, +0.125], p=0.315 | +0.016 [-0.030, +0.061], p=0.286 | +0.025 [-0.015, +0.068], p=0.157 |
| Text Encoder | BERT-origin text | +0.011 [-0.038, +0.060], p=0.361 | +0.053 [-0.059, +0.184], p=0.236 | +0.000 [-0.034, +0.034], p=0.497 | +0.035 [-0.011, +0.084], p=0.111 |
| Text Encoder | FinBERT-origin text | +0.013 [-0.029, +0.060], p=0.313 | +0.046 [-0.043, +0.143], p=0.209 | +0.006 [-0.029, +0.042], p=0.396 | +0.033 [-0.010, +0.082], p=0.121 |
| Text Encoder | Qwen3-4B-origin text | +0.005 [-0.052, +0.062], p=0.442 | +0.019 [-0.084, +0.128], p=0.382 | +0.001 [-0.037, +0.044], p=0.478 | +0.017 [-0.031, +0.062], p=0.277 |
| Text Encoder | E5-Mistral-7B-origin text | +0.071 [+0.026, +0.114], p=0.004 | +0.109 [+0.025, +0.197], p=0.019 | +0.039 [-0.000, +0.077], p=0.049 | +0.117 [+0.068, +0.159], p<0.001 |
| Text Encoder | BGE-origin text | +0.013 [-0.033, +0.061], p=0.327 | +0.018 [-0.082, +0.122], p=0.387 | +0.013 [-0.026, +0.055], p=0.300 | +0.024 [-0.023, +0.069], p=0.194 |
| Local LLM | Llama3.1-8B | +0.136 [+0.082, +0.186], p<0.001 | +0.203 [+0.110, +0.304], p<0.001 | +0.074 [+0.014, +0.139], p=0.026 | +0.212 [+0.161, +0.261], p<0.001 |
| Local LLM | Qwen2.5-7B | +0.143 [+0.088, +0.198], p<0.001 | +0.221 [+0.111, +0.357], p=0.002 | +0.083 [+0.040, +0.128], p=0.001 | +0.230 [+0.168, +0.296], p<0.001 |
| Commercial API | GPT Chat Latest | +0.013 [-0.040, +0.073], p=0.348 | +0.014 [-0.103, +0.127], p=0.423 | +0.005 [-0.029, +0.035], p=0.404 | +0.084 [+0.016, +0.145], p=0.016 |
| Commercial API | Claude Sonnet 4.5 | +0.022 [-0.017, +0.059], p=0.171 | +0.051 [-0.048, +0.165], p=0.214 | +0.004 [-0.022, +0.030], p=0.405 | +0.081 [+0.025, +0.138], p=0.010 |
| Commercial API | Gemini 2.5 Flash | +0.042 [-0.001, +0.083], p=0.050 | +0.094 [-0.008, +0.201], p=0.068 | +0.018 [-0.020, +0.054], p=0.210 | +0.131 [+0.082, +0.187], p<0.001 |
| Commercial API | DeepSeek V4 Flash | +0.045 [-0.024, +0.106], p=0.125 | +0.079 [-0.090, +0.228], p=0.208 | +0.013 [-0.027, +0.045], p=0.273 | +0.131 [+0.046, +0.205], p=0.003 |
| Commercial API | Qwen3.7 Plus | +0.037 [-0.012, +0.088], p=0.111 | +0.016 [-0.073, +0.115], p=0.388 | +0.038 [-0.009, +0.082], p=0.085 | +0.104 [+0.042, +0.161], p=0.002 |

Family-average support:

This summary averages OL-Origin paired deltas across methods within each
baseline family. It is a descriptive family-level diagnostic, not a replacement
for the method-level paired bootstrap table above. The text-encoder and
commercial family-average rows are retained as legacy diagnostics and are not
yet refreshed for the current Qwen3-4B / E5-Mistral and phase43 commercial API
replacements.

| Family average baseline | n methods | ΔNDCG@3 [90% CI], p | ΔHit@1 [90% CI], p | ΔMass@3 [90% CI], p | JS improvement [90% CI], p |
|---|---:|---:|---:|---:|---:|
| Scale | 2 | +0.032 [-0.006, +0.069], p=0.083 | +0.016 [-0.056, +0.088], p=0.357 | +0.032 [-0.004, +0.067], p=0.070 | +0.030 [+0.012, +0.047], p=0.003 |
| Context | 5 | +0.078 [+0.048, +0.108], p<0.001 | +0.109 [+0.053, +0.165], p<0.001 | +0.046 [+0.018, +0.074], p=0.003 | +0.054 [+0.040, +0.069], p<0.001 |
| Surface Text | 3 | +0.037 [-0.002, +0.076], p=0.059 | +0.061 [-0.008, +0.130], p=0.073 | +0.017 [-0.010, +0.043], p=0.148 | +0.037 [+0.012, +0.063], p=0.009 |
| Text Encoder (phase28 legacy) | 4 | +0.011 [-0.014, +0.035], p=0.235 | +0.032 [-0.025, +0.089], p=0.176 | +0.003 [-0.016, +0.023], p=0.383 | +0.029 [+0.005, +0.052], p=0.022 |
| Local LLM | 2 | +0.140 [+0.102, +0.178], p<0.001 | +0.212 [+0.133, +0.291], p<0.001 | +0.079 [+0.040, +0.117], p<0.001 | +0.221 [+0.179, +0.263], p<0.001 |
| Commercial API (phase32 legacy) | 4 | +0.058 [+0.020, +0.095], p=0.006 | +0.094 [+0.027, +0.162], p=0.011 | +0.029 [+0.002, +0.055], p=0.038 | +0.128 [+0.086, +0.170], p<0.001 |

Main-table statistical interpretation:

```text
The strongest statistically supported result is OL-Origin over the main
non-OL structural control: NDCG@3, Hit@1, and JS all have positive bootstrap
support. This validates that the KOL originator structure adds information
beyond ordinary origin-time context.

Against LLM-based scorers, OL-Origin is strongly supported against local LLMs
and has a robust JS advantage against the refreshed commercial API rows. GPT
Chat Latest is the strongest refreshed API baseline: OL-Origin has better point
estimates, but only the JS advantage is statistically supported.

Against older BERT-family text encoders, OL-Origin has better point estimates
on most metrics but is not statistically separated. Against E5-Mistral-7B,
OL-Origin is statistically stronger on NDCG@3, Hit@1, and JS in the paired
bootstrap table. Against Qwen3-4B, the correct claim remains top-tier / best
point-estimate quality with much lower latency, not statistical domination.

At the family-average level, OL-Origin is significantly better than Context and
Local LLM families across all four metrics. The Text Encoder and Commercial API
family-average rows are retained as legacy diagnostics until those family
summaries are refreshed for the current main-table roster.
```

## 11. Ablation Study: What Makes OL-Origin Work

The ablation study uses the same main setting as the primary experiment:

```text
semantic threshold = 0.55
origin window = first10
target = future follower-weighted Reach
events = 785
symbols = 16
```

The goal is to test whether the stable originator role is doing real work, and
whether it needs to be coupled with origin-time context.

### 11.1 Feature Taxonomy

The full OL-Origin router is a linear ridge model. It is not a standalone KOL
score. Its online score is a linear combination of three feature families:

```text
score(originated frame) =
  originator role
  + origin-time context controls
  + originator-role interactions
```

The discovered structure enters the model through:

```text
origin_ol
ol_x_visibility = origin_ol * origin_logfoll
ol_x_novelty    = origin_ol * novelty_global
```

`origin_ol` is the core discovered structure. It measures whether a KOL has
historically acted as a stable narrative originator. It is estimated only from
pre-2020 data and residualized against median posting hour, so the validation
experiment uses a point-in-time trait rather than future information.

The two interaction terms embed this discovered structure into the routing
problem:

```text
stable originator role x account visibility
stable originator role x semantic novelty
```

This reflects the intended mechanism: a stable originator is especially useful
when the originated frame is visible and/or semantically novel.

The non-OL context controls are:

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

These controls are not the paper's core discovery. They are included to make the
comparison fair by controlling for account scale, timing, event context,
sentiment, semantic novelty, and historical activity.

### 11.2 Ablation Design

Each ablation answers a specific identification question:

| Variant | Feature construction | Question answered |
|---|---|---|
| No-OL Strong | all non-OL controls, no `origin_ol` and no OL interactions | Does OL structure add value beyond strong origin-time context? |
| OL Only | `origin_ol` only | Is the stable originator trait sufficient by itself? |
| OL-Origin Full | No-OL controls + `origin_ol` + OL interactions | Does the discovered structure help when coupled with context? |
| Shuffled OL-Origin | No-OL controls + shuffled `origin_ol` + shuffled OL interactions | Does the true KOL-to-role mapping matter, or only the marginal OL distribution? |
| Follower Replacement | No-OL controls + follower-scale interaction analogues | Is the effect merely a large-account effect? |
| Raw OL-Origin | No-OL controls + non-residualized raw OLtrait + raw OL interactions | Is hour residualization driving or suppressing the result? |

The most important falsification check is `Shuffled OL-Origin`. It preserves the
distribution of OLtrait values but destroys which KOL receives which trait. If
the full model beats this variant, the result depends on the learned KOL-role
mapping rather than on an arbitrary extra numeric feature.

The second key falsification check is `Follower Replacement`. It gives the
follower-scale baseline comparable interaction capacity:

```text
logfoll_x_visibility = origin_logfoll * origin_logfoll
logfoll_x_novelty    = origin_logfoll * novelty_global
```

This tests whether the OL channel is simply a disguised version of follower
count or account scale.

### 11.3 Ablation Results

| Variant | Role | NDCG@3 | Hit@1 | Mass@3 | JS ↓ |
|---|---|---:|---:|---:|---:|
| No-OL Strong | remove OLtrait from the full router | 0.712 | 0.421 | 0.891 | 0.301 |
| OL Only | use only stable originator role | 0.650 | 0.292 | 0.854 | 0.314 |
| **OL-Origin Full** | full origin-aware router | **0.755** | **0.493** | <u>0.901</u> | **0.263** |
| Shuffled OL-Origin | shuffle OLtrait across KOLs | 0.724 | 0.459 | 0.892 | 0.295 |
| Follower Replacement | replace OL channel with follower-scale analogues | 0.717 | 0.431 | 0.895 | 0.296 |
| Raw OL-Origin | use non-residualized OLtrait | <u>0.749</u> | <u>0.489</u> | **0.905** | <u>0.268</u> |

Symbol-balanced bootstrap comparisons:

| Comparison | ΔNDCG@3 | 90% CI | ΔHit@1 | 90% CI | JS improvement | 90% CI |
|---|---:|---:|---:|---:|---:|---:|
| Full - No-OL Strong | +0.043 | [+0.019, +0.069] | +0.072 | [+0.013, +0.156] | +0.039 | [+0.022, +0.058] |
| Full - OL Only | +0.105 | [+0.056, +0.149] | +0.201 | [+0.101, +0.308] | +0.052 | [+0.029, +0.076] |
| Full - Shuffled OL | +0.031 | [+0.009, +0.057] | +0.034 | [-0.014, +0.093] | +0.033 | [+0.017, +0.051] |
| Full - Follower Replacement | +0.038 | [+0.013, +0.067] | +0.063 | [+0.001, +0.134] | +0.033 | [+0.017, +0.053] |
| Full - Raw OL | +0.007 | [-0.009, +0.025] | +0.004 | [-0.027, +0.035] | +0.006 | [+0.001, +0.011] |

### 11.4 Interpretation

```text
The full router is not just a context model: removing OLtrait reduces ranking
quality and calibration. OLtrait alone is also not sufficient, which means the
originator role must be coupled with origin-time context.

The true KOL-to-OLtrait mapping matters: shuffling OLtrait across KOLs preserves
the marginal distribution but loses a significant amount of NDCG and JS.

The result is not just a large-account effect: replacing the OL channel with
follower-scale analogues is weaker than the full origin-aware router.

Raw OLtrait remains strong, but residualized OLtrait gives slightly better JS
and provides a cleaner defense against timezone/hour confounding.
```

The ablation supports the central modeling claim:

```text
The discovered KOL originator structure is useful, but it is useful as a
component inside an origin-time routing model, not as an isolated KOL ranking.
```

In other words, the signal is not simply "who has many followers" or "who has a
large score." The useful object is:

```text
who originates + when it is originated + how novel the originated frame is
```

## 12. Threshold Robustness

OL-Origin minus No-OL Strong:

| Threshold | Adopt NDCG Δ | Adopt Hit@1 Δ | Adopt JS Δ | Reach NDCG Δ | Reach Hit@1 Δ | Reach JS Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 0.45 | +0.022 | +0.007 | +0.001 | +0.022 | +0.010 | -0.012 |
| 0.50 | +0.009 | +0.023 | +0.001 | +0.008 | +0.021 | +0.007 |
| 0.55 | +0.033 | +0.055 | +0.004 | +0.043 | +0.072 | +0.039 |
| 0.60 | +0.038 | +0.092 | +0.002 | +0.050 | +0.120 | +0.021 |
| 0.65 | +0.060 | +0.073 | +0.002 | +0.070 | +0.067 | +0.023 |

Boundary checks:

| Threshold | Key Result | Interpretation |
|---:|---|---|
| 0.35 | OL-Origin underperforms No-OL Strong | too loose; frame definition collapses into broad clusters |
| 0.50 | OL-Origin remains competitive but BGE leads | low-threshold supplement, not main setting |
| 0.75 | OL-Origin beats No-OL Strong but BGE dominates | narrow text-encoder-favorable stress test |

Why 0.55 is the main setting:

```text
0.55 is the first stable coherent-frame region: it is not so loose that
narratives collapse, and not so narrow that the task becomes mostly text
matching.
```

## 13. Why Text Encoders Are Strong

Text encoders are strong mainly because the origin tweet exposes visible
surface and topic salience, not because of simple ticker leakage.

Diagnostic result, averaged over 0.55 / 0.60 / 0.65:

| Method | Adopt NDCG@3 | Adopt Hit@1 | Reach NDCG@3 | Reach Hit@1 |
|---|---:|---:|---:|---:|
| Symbol one-hot | 0.634 | 0.365 | 0.635 | 0.352 |
| Text surface | 0.751 | 0.542 | 0.750 | 0.508 |
| Symbol + text surface | 0.750 | 0.541 | 0.749 | 0.503 |

Interpretation:

```text
Text surface / BERT-family: content-form salience of the originated frame
OL-Origin: stable KOL role structure of the originator
No-OL Strong: non-OL origin-time context
```

Positive surface/topic cues:

```text
uppercase ratio
earnings / revenue / guidance language
percent sign or numeric-performance language
AI-related language
```

Negative or suppressive cues:

```text
many cashtags
many mentions / hashtags
option-flow language
macro / legal language in this origin-alert setup
excessive token count in several thresholds
```

## 14. Commercial API Baseline

Commercial models are evaluated through OpenRouter:

```text
openai/gpt-chat-latest
anthropic/claude-sonnet-4.5
google/gemini-2.5-flash
deepseek/deepseek-v4-flash
qwen/qwen3.7-plus
```

Input boundary:

```text
anonymized origin text + origin-time non-OL context
no OLtrait
no KOL identity
no symbol
no date
no follower confirmation
no future text
```

Run status:

```text
batch size = 10
phase32 elapsed = 10,754.6 seconds
phase43 GPT Chat Latest elapsed = 1,362.6 seconds
phase43 DeepSeek V4 Flash elapsed = 3,455.1 seconds
phase43 Qwen3.7 Plus elapsed = 4,324.5 seconds
```

Parsing coverage:

```text
GPT Chat Latest:     5785 / 5785 API items parsed, 785 events
Claude Sonnet 4.5:   5781 / 5781 API items parsed, 785 events
Gemini 2.5 Flash:    5561 / 5781 API items parsed, 759 events
DeepSeek V4 Flash:   5425 / 5675 API items parsed, 753 events
Qwen3.7 Plus:        5695 / 5695 API items parsed, 785 events
```

Additional notes:

```text
openai/gpt-5-mini was smoke-tested but rejected as default because it returned
responses with failed JSON parsing.
google/gemini-3.5-flash was run as a diagnostic but is excluded from the main
table because OpenRouter credit/max-token failures left only 334 events covered.
google/gemma-4-31b-it:free is excluded because the upstream provider returned
429 rate-limit errors during the run.
```

## 15. Current Table Plan

Recommended main-text tables:

```text
Table 1: Main 0.55 Reach origin-alert result
  rows = scale / context / origin-role / surface / encoder / local LLM / commercial API
  columns = coverage, Reach metrics, delta vs No-OL, input length, latency

Table 2: Statistical support for low-latency routing
  rows = OL-Origin vs No-OL Strong and OL-Origin vs commercial LLMs
  columns = delta metrics and 90% confidence intervals

Table 3: Latency analysis
  rows = representative structural, text, local LLM, and commercial API methods
  columns = Reach quality, latency, input length, latency multiple

Table 4: Agent-facing listwise routing small experiment
  rows = LLM backend x full/routed policy
  columns = K or b, capture, selector-component latency, routed LLM wall-clock diagnostics

Table 5: Ablation study
  rows = No-OL, OL-only, full OL-Origin, shuffled OL, follower replacement, raw OL
  columns = Reach metrics and bootstrap deltas

Table 6: Threshold robustness
  rows = semantic thresholds
  columns = adoption and reach deltas

Table 7: Text-surface diagnostic
  rows = symbol one-hot, text surface, symbol + surface
  purpose = explain why BERT-family baselines are strong
```

Appendix:

```text
Additional model probes, excluded OpenRouter diagnostics, and old intermediate
experiments can remain in appendix only if needed for auditability.
```

## 16. Reproducibility Artifacts

Main structural experiment:

```text
experiments/socialenc/phase7_origin_alert.py
experiments/socialenc/phase7_origin_alert_result.json
experiments/socialenc/phase7_origin_alert_first10_supplement.py
experiments/socialenc/phase7_origin_alert_first10_supplement_result.json
```

Text encoder baselines:

```text
experiments/socialenc/phase28_origin_alert_encoder_baselines.py
experiments/socialenc/phase28_origin_alert_encoder_baselines_result.json
experiments/socialenc/phase28_origin_alert_encoder_baselines_thr050_result.json
experiments/socialenc/phase28_origin_alert_encoder_baselines_thr035_075_result.json
experiments/socialenc/phase39_qwen3_origin_alert_encoder_probe.py
experiments/socialenc/phase39_qwen3_origin_alert_encoder_probe_result.json
experiments/socialenc/phase41_e5_mistral_origin_alert_encoder_probe.py
experiments/socialenc/phase41_e5_mistral_origin_alert_encoder_probe_result.json
experiments/socialenc/phase42_new_encoder_cost_benchmark.py
experiments/socialenc/phase42_new_encoder_cost_benchmark_result.json
```

Text surface diagnostic:

```text
experiments/socialenc/phase29_origin_alert_text_surface_diagnostic.py
experiments/socialenc/phase29_origin_alert_text_surface_diagnostic_result.json
```

Local LLM baselines:

```text
experiments/socialenc/phase18_origin_alert_llm_baselines.py
experiments/socialenc/phase18_origin_alert_llm_baselines_result.json
```

Latency analysis:

```text
experiments/socialenc/phase31_origin_alert_cost_benchmark.py
experiments/socialenc/phase31_origin_alert_cost_benchmark_result.json
experiments/socialenc/phase34_latency_quality_frontier.py
experiments/socialenc/phase34_latency_quality_frontier_result.json
experiments/socialenc/phase34_latency_quality_frontier_table.csv
experiments/socialenc/phase34_latency_quality_frontier_table.md
experiments/socialenc/phase34_latency_quality_frontier_ndcg.png
experiments/socialenc/phase42_new_encoder_cost_benchmark.py
experiments/socialenc/phase42_new_encoder_cost_benchmark_result.json
```

Agent-facing listwise routing small experiment:

```text
experiments/socialenc/phase50_deepseek_listwise_dilution_result.json
experiments/socialenc/phase50_deepseek_listwise_dilution_table.md
experiments/socialenc/phase50_deepseek_listwise_cache.jsonl
experiments/socialenc/phase53_claude_sonnet_4_6_listwise_dilution_result.json
experiments/socialenc/phase53_claude_sonnet_4_6_listwise_dilution_table.md
experiments/socialenc/phase53_claude_sonnet_4_6_listwise_cache.jsonl
experiments/socialenc/phase54_gpt_5_4_listwise_dilution_result.json
experiments/socialenc/phase54_gpt_5_4_listwise_dilution_table.md
experiments/socialenc/phase54_gpt_5_4_listwise_cache.jsonl
experiments/socialenc/phase55_gemini_2_5_flash_lite_listwise_dilution_result.json
experiments/socialenc/phase55_gemini_2_5_flash_lite_listwise_dilution_table.md
experiments/socialenc/phase55_gemini_2_5_flash_lite_listwise_cache.jsonl
```

Main-table bootstrap support:

```text
experiments/socialenc/phase36_main_table_bootstrap_support.py
experiments/socialenc/phase36_main_table_bootstrap_support_result.json
experiments/socialenc/phase36_main_table_bootstrap_support_table.md
experiments/socialenc/phase37_family_average_bootstrap_support.py
experiments/socialenc/phase37_family_average_bootstrap_support_result.json
experiments/socialenc/phase37_family_average_bootstrap_support_table.md
```

OL-Origin ablation:

```text
experiments/socialenc/phase33_origin_alert_ablation.py
experiments/socialenc/phase33_origin_alert_ablation_result.json
```

Commercial API baselines:

```text
experiments/socialenc/phase32_openrouter_origin_alert_baselines.py
experiments/socialenc/phase32_openrouter_origin_alert_baselines_result.json
experiments/socialenc/phase32_openrouter_origin_alert_baselines_progress.json
experiments/socialenc/phase43_openai__gpt-chat-latest_result.json
experiments/socialenc/phase43_deepseek__deepseek-v4-flash_result.json
experiments/socialenc/phase43_qwen__qwen3.7-plus_result.json
```

## 17. Current Conclusion

The current results support the following paper narrative:

```text
Agentic trading systems need a selective, low-latency social-media entry layer,
not just larger context windows and heavier text models. Stable KOL originator
traits provide such a layer: they help route newly originated financial
narratives before popularity is observable.

At the main 0.55 setting, OL-Origin has the best point estimates on Reach
NDCG@3, Hit@1, and JS. Its gains over the strong non-OL structural control are
supported by symbol-balanced bootstrap. Against the current E5 representative
(E5-Mistral-7B), OL-Origin is statistically stronger on NDCG@3, Hit@1, and JS.
Against Qwen3-4B, the main claim is competitive top-tier quality with zero
prompt tokens and near-zero online latency. This makes it suitable as a
first-stage attention router for downstream research, memory, RAG, and trading
agents.
```
