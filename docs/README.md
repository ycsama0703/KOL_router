# docs — navigation

**Current paper = the Lead-Lag Router (LLR), graph version.** Read these (top level):

- `GRAPH_ROUTER_RESULTS_AND_METHOD.md` — main table, method, ablation, **§4 application layer** (LLM-triage routing + event prediction)
- `THEORY.md` — problem (why LLM triage doesn't scale) → lead-lag origination potential → Result A/B → rigor; equations (1)–(9)
- `ABLATION_SUMMARY.md` — ablation (knock-out / design-choices / reality-checks) + prior-art controlled comparison
- `NOVELTY_AUDIT.md` — prior-art collision audit (CasMS / Yamada / Romero / Zhou …)
- `INTRO_RELATEDWORK_DRAFT.md` — intro + related work draft

Reference: `FINDATA_CATALOG.md`, `FINDATA_VERIFICATION.md`.

**Legacy (pre-graph writeups, superseded):** `legacy/` — original originator-structure discovery,
theoretical framing v1/v2, econ appendix, OOT retest plan, talk script, research log. Kept for
archive; not part of the current paper.

Code: `../experiments/socialenc/` (see its `MANIFEST.md` for graph-paper vs legacy script split —
kept flat because the graph experiments reuse shared base modules).
