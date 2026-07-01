# Theory Hard-Error Recheck

Date: 2026-07-01

Scope: this is a second-pass check of `docs/THEORY.md` after the first hard-error audit. It only
tracks hard mathematical / logical errors. It does not discuss claim-strength issues such as
"sufficient statistic", "LLM gets worse as data grows", or whether the theoretical framing should be
softened for reviewers.

## Current Status

Most hard errors from the first audit have been fixed.

| Issue | Current status |
|---|---|
| Candidate count `K` used as token count | Fixed |
| Entropy bound used to justify sharpness limit | Fixed |
| Eq. (9) cites `(1)-(8)` instead of graph equations | Fixed |
| "potential equals net-degree" literal equality | Mostly fixed |
| HodgeRank sign / leadership direction | Still not fully clean |

The only remaining hard issue is the HodgeRank sign convention and how it maps to the paper's
positive leader score `g_net = out - in`.

---

## 1. Fixed: Candidate Count vs Token Count

`THEORY.md` now correctly distinguishes candidate count `K` from token length `n`:

```math
n(K)\approx K\bar L
```

and writes transformer cost as:

```math
\tau_{\text{LLM}}(K)
=\Theta(n^2d+nd^2)
=\Theta((K\bar L)^2d+K\bar Ld^2).
```

This is mathematically clean: quadratic scaling is in token length `n`; it becomes quadratic in `K`
only when the average candidate length `\bar L` is treated as fixed.

## 2. Fixed: Entropy Bound

`THEORY.md` no longer uses:

```math
H(\alpha)\le \log K
```

as a sharpness argument. It now relies on the bounded-logit mass bound:

```math
\alpha_i\le \frac{1}{K}e^{2\Delta/T}.
```

This is the correct mathematical route: with bounded logit range, keeping constant mass on one target
requires logit separation to grow like `log K`.

## 3. Fixed: Eq. (9) Reference

Eq. (9) now cites:

```text
via (6)-(8) / Result A
```

instead of `(1)-(8)`. This is correct because `g_net` is introduced only in the lead-lag graph
equations, not in the LLM attention / compute equations.

## 4. Mostly Fixed: Equality vs Proportionality

The text no longer says the Hodge potential literally equals net-degree. It now writes:

```math
s^{*}_a=-\frac{1}{n}g_{\text{net}}(a).
```

This correctly introduces a scale factor. However, this equation also exposes the remaining sign
problem: under the current convention, `s*` is proportional to **negative** `g_net`, not positive
`g_net`.

---

## 5. Remaining Hard Issue: HodgeRank Sign / Ranking Direction

### Current text

`THEORY.md` currently defines:

```math
Y_{ab}=w(a\to b)-w(b\to a),
```

and:

```math
g_{\text{net}}(a)=\sum_b Y_{ab}.
```

So a leader/originator has positive `g_net`.

The Hodge equation then says:

```math
\Delta_0 s=-\operatorname{div}Y,\qquad
\operatorname{div}(Y)(a)=\sum_bY_{ab}=g_{\text{net}}(a).
```

Therefore the right-hand side is:

```math
-\operatorname{div}Y=-g_{\text{net}}.
```

But the prose currently says:

```text
whose right-hand side is exactly the net-degree
```

That sentence is mathematically false under the current convention. The right-hand side is exactly
the **negative** net-degree.

The text then writes:

```math
s^{*}_a=-\frac{1}{n}g_{\text{net}}(a).
```

This means higher-lead accounts have lower Hodge potential values. So the leadership ranking is not
"same ranking by `s*`"; it is the same ranking by `-s*`, or equivalently by `g_net`.

### Why this matters

This is not just wording. If a reader follows the equations literally, the direction of the Hodge
potential ranks leaders last unless the paper explicitly says the leadership score is `-s*` or
`g_net = -n s*`.

### Minimal fix

Keep the standard HodgeRank equation:

```math
\Delta_0s=-\operatorname{div}Y
```

and keep the paper's positive leader score:

```math
g_{\text{net}}(a)=\sum_bY_{ab}.
```

Then rewrite the prose around Eq. (8) as:

```text
Under this orientation, the right-hand side is the negative net-degree. On a complete,
uniformly-weighted graph, the minimum-norm solution satisfies

s^{*}_a = -\frac{1}{n} g_{\text{net}}(a).

Thus the Hodge potential ranks leaders in the reverse direction, and the paper's leadership score is
the sign-flipped potential, g_net = -n s*. Ranking by g_net is therefore equivalent to ranking by
-s*, not by s*.
```

### Alternative fix

If the paper wants `s*` itself to be the positive leader score, redefine either the edge flow or the
divergence:

```math
\operatorname{div}(Y)(a)=-\sum_bY_{ab}.
```

Then `\Delta_0s=-divY` gives a positive leader potential. This is also valid, but it is easier to
misread because `div(Y)` no longer equals the intuitive row-sum. For this paper, the minimal fix
above is clearer: keep `g_net=out-in` and explicitly say it is the sign-flipped Hodge potential.

---

## Patch-Ready Replacement

Replace the paragraph around Eq. (8) in `docs/THEORY.md` with the following:

```markdown
The consistent ranking is the potential solving the Laplacian normal equation

$$\Delta_0\,s=-\operatorname{div}Y,\qquad
\operatorname{div}(Y)(a)=\sum_b Y_{ab}=g_{\text{net}}(a).\tag{8}$$

Under this HodgeRank orientation, the right-hand side is the **negative** of the paper's positive
leader score. The minimum-norm solution is
`s^{*}=-\Delta_0^{\dagger}\operatorname{div}Y`. On a complete, uniformly-weighted graph this collapses
to

$$s^{*}_a=-\tfrac1n\,\operatorname{div}(Y)(a)=-\tfrac1n\,g_{\text{net}}(a),\tag{8a}$$

so the Hodge potential ranks leaders in the reverse direction. We therefore use the sign-flipped
leadership score `g_{\text{net}}=-n\,s^{*}`: ranking by `g_net` is equivalent to ranking by `-s*`.
This recovers the Borda/Massey row-sum ranking up to orientation and scale.
```

## Final Recheck Verdict

After applying the Hodge wording fix above, the current hard mathematical errors are cleared. The
remaining concerns in `THEORY.md` are not algebraic errors; they are scope and claim-strength issues
that should be handled separately if the theory section is converted into paper prose.

---

## Resolution (2026-07-01, applied)

Item 5 (HodgeRank sign) wording fixed in `THEORY.md` around Eq. (8)/(8a): removed the misleading
"right-hand side is exactly the net-degree" (the normal-equation RHS is the **negative** leader
score), and now state explicitly that `s* = −(1/n) g_net` ranks leaders in **reverse**, so the paper
uses `g_net = −n·s*` and **ranking by `g_net` ≡ ranking by `−s*` (not `s*`)**. Items 1–4 already
cleared in the prior pass. All hard mathematical errors are now resolved.
