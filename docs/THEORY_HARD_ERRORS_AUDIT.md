# Theory Hard-Error Audit

Scope: this note only lists hard mathematical / logical errors in `docs/THEORY.md`.
It intentionally excludes softer issues such as "the claim is too strong", "the caveat should be
expanded", or "the wording invites reviewer pushback".

## Summary

There are four hard issues that should be fixed before this theory section is reused:

1. HodgeRank sign convention is inconsistent with the paper's `g_net = out - in` definition.
2. Transformer cost uses candidate count `K` where the correct variable is token count `n`.
3. The entropy bound `H(alpha) <= log K` is used to justify sharpness limits, but that implication is
   mathematically wrong.
4. Equation (9) cites `(1)-(8)` as the source of `g_net`, but only `(6)-(8)` are relevant.

A fifth item is not fatal but should be corrected for exactness: "potential equals net-degree" should
be "potential is proportional to net-degree / gives the same ordering" under complete uniform graphs.

---

## 1. HodgeRank Sign Error

Location: `docs/THEORY.md`, Eq. (6)-(8).

The current definitions are:

```math
Y_{ab}=w(a\to b)-w(b\to a)
```

and

```math
g_{\text{net}}(a)=\sum_b Y_{ab}.
```

Under this convention, a leader/originator has positive `g_net`: it precedes others more often than
others precede it.

The document then writes:

```math
\Delta_0 s=-\operatorname{div}Y,\qquad
\operatorname{div}(Y)(a)=\sum_b Y_{ab}=g_{\text{net}}(a).
```

This reverses the score direction. If `div(Y)(a)=g_net(a)`, then the normal equation should use the
same sign:

```math
\Delta_0 s=\operatorname{div}Y.
```

Alternative fix: keep the negative sign, but redefine divergence as:

```math
\operatorname{div}(Y)(a)=-\sum_b Y_{ab}.
```

Preferred fix for this paper: keep `g_net = out - in` as the positive originator score and change Eq.
(8) to:

```math
\Delta_0 s=\operatorname{div}Y,\qquad
\operatorname{div}(Y)(a)=\sum_b Y_{ab}=g_{\text{net}}(a).
```

## 2. Candidate Count vs Token Count in Transformer Complexity

Location: `docs/THEORY.md`, P2 / Eq. (5).

The document currently writes self-attention cost as:

```math
O(K^2 d)
```

where `K` is the number of candidates.

This is not the transformer complexity variable. The correct variable is token length `n`, not
candidate count. If each candidate has average length `\bar L`, then:

```math
n \approx K\bar L.
```

So the per-layer cost should be written as:

```math
O(n^2 d + n d^2)
= O((K\bar L)^2 d + K\bar L d^2).
```

Only under fixed average candidate length `\bar L` can we simplify the scaling to quadratic in `K`.

Recommended replacement:

```math
n(K)\approx K\bar L,\qquad
\tau_{\text{LLM}}(K)=\Theta(n(K)^2d+n(K)d^2)
=\Theta((K\bar L)^2d+K\bar Ld^2).
```

## 3. Entropy Bound Is Used Incorrectly

Location: `docs/THEORY.md`, Eq. (4) and the sentence immediately after it.

The current bound is:

```math
\frac1K e^{-2\Delta/T}\le \alpha_i\le \frac1K e^{2\Delta/T},
\qquad H(\alpha)\le \log K.
```

The first inequality can support a dilution argument: if the logit range is bounded, then the maximum
attention mass on any one candidate is at most `O(1/K)`.

But the entropy statement:

```math
H(\alpha)\le \log K
```

does not imply that attention cannot be sharp. It is only the standard maximum-entropy upper bound on
a distribution over `K` items. A one-hot distribution has entropy `0`, so this inequality does not
cap sharpness.

Recommended fix: remove the entropy clause from the argument and use only the bounded-logit mass
bound:

```math
\alpha_i\le \frac1K e^{2\Delta/T}.
```

Then state:

> With bounded logit range, keeping constant mass on a single target as `K` grows requires the logit
> separation to grow like `log K`.

## 4. Wrong Equation Reference in Eq. (9)

Location: `docs/THEORY.md`, Eq. (9).

The current text says the reach mechanism uses:

```text
via (1)-(8): g_net
```

This is incorrect because Eq. (1)-(5) are about LLM attention and transformer compute, not about
lead-lag graph scoring.

The graph score is introduced only in Eq. (6)-(8). The reference should be:

```text
via (6)-(8): g_net
```

or:

```text
via Result A: g_net
```

## 5. Exact Equality Should Be Replaced by Proportionality

Location: `docs/THEORY.md`, HodgeRank discussion after Eq. (8).

The document says that on a complete uniformly weighted graph:

```text
the potential equals the net-degree
```

The exact statement is generally proportionality / same ranking, not literal equality. With a
complete graph and a zero-sum constraint, the Laplacian introduces a scale factor such as `1/n`.

Recommended fix:

```text
On a complete, uniformly-weighted graph, the Hodge potential is proportional to net-degree and
therefore induces the same ranking.
```

This is an exactness correction rather than a fatal logic error.

---

## Not Counted Here

The following issues are real but are not "hard mathematical errors" under this audit scope:

- Calling `g_net` a "sufficient statistic".
- Saying LLM performance necessarily gets worse as data grows.
- Saying row-sum is optimal "for all reasonable losses".
- Treating Hawkes / IC / LT mechanisms as a proof of frame-level reach.
- Saying `rank_xendcg` consistency attaches directly to a finite LightGBM model.
- Saying residualized `O_k` is better than raw `O_k` when the ablation says the difference is not
  meaningful.

These should still be softened in the theory document, but they are mainly claim-strength or
scope-control problems rather than algebraic errors.

---

## Resolution (2026-07-01, THEORY.md updated)

All five confirmed valid and fixed in `docs/THEORY.md`:

1. **Sign** — kept the correct least-squares normal equation `Δ₀ s = −div Y` (the audit's suggested
   `Δ₀ s = div Y` would misstate it), and added Eq. (8a) `s* = −(1/n) g_net`, stating the potential is
   **proportional to net-degree up to the −1/n orientation-scale**; we rank by `g_net = −n·s*`.
2. **K vs n** — added Eq. (5a) `n(K) ≈ K·L̄`; Eq. (5) now `Θ(n²d + nd²) = Θ((K L̄)²d + K L̄ d²)`,
   quadratic in K only for fixed `L̄`.
3. **Entropy** — removed `H(α) ≤ log K` from the sharpness argument; sharpness limit now rests only on
   the bounded-logit mass bound `α_i ≤ e^{2Δ/T}/K`, with an explicit note that entropy ≤ log K does not
   cap peak sharpness (one-hot has entropy 0).
4. **Eq. (9) ref** — changed "via (1)–(8)" to "via (6)–(8) / Result A".
5. **Equality → proportionality** — folded into fix 1 (Eq. 8a; "proportional … same ranking").

(The "Not Counted Here" claim-strength items remain as-is; they are honesty/scope hedges, addressable
separately if desired.)
