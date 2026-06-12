---
title: "Pre-Registration: Dual Side-Specialized Ranking Transformers"
type: spec
status: dead
model: "Transformer"
verdict: DEAD
updated: 2026-06-12
tags: []
---
# 🧪 Pre-Registration: Dual Side-Specialized Ranking Transformers

> ⚠️ EXPLORATORY pre-registration. No Gauntlet verdict authority. Date: 2026-06-12.
> Build log: [[06 — Logs/Conversations/Conv-2026-06-12-Sophisticated-Transformer]]
> Supersedes the single-head P(up) design for the ranking experiment.

## Hypothesis
Decomposing the cross-sectional transformer into **two independent side-specialist rankers**
(`dualres_long` trained on forward return `y`, `dualres_short` trained on `−y`) with a
**listwise ranking loss** beats the single-head P(up) BCE model on each side's own
out-of-sample selection metric. Primary interest is the **short side** (the only side with
persistent raw signal in the XGBoost suite — see [[project_v8_1h_walkforward_demoted]]).

## Design (frozen before runs)
- **Architecture:** existing `DualResCSTransformer` unchanged (one scalar score per name).
- **Sign convention:** HIGH score = STRONG in the model's side. Long model: target `softmax(zscore(y))`.
  Short model: target `softmax(zscore(−y))`. Selection = Top-K by score; realized PnL = `+y` (long) / `−y` (short).
- **Loss:** ListNet top-1 (cross-sectional softmax CE on z-scored side target), masked to present & finite-label names.
- **Split:** chronological 70/15/15 with EMBARGO=30 decision timestamps (reused from `train.py`).
- **Early stop:** val cross-sectional rank-IC (mean per-timestamp Spearman of score vs side target).
- **Batch:** 16 timestamps (up from 8). One pre-registered run per side; no test-set tuning.

## Primary metrics (per side, OOS test)
- Raw cross-sectional Spearman ρ (score vs side target).
- Top-1/3/5 **net@10 bps** with t-stat; reported also @6 and @20.
- RAW vs NET win-rate + `median(net−gross) == −cost` cost-sign sanity (per [[feedback_validate_cost_accounting]]).

## Kill criteria (decided in advance)
- **SHORT (the live question):** PASS only if raw ρ ≥ 0.05 **AND** net@10 significantly > 0 (t > 2)
  over the **full** OOS span — not concentrated in one day (explicit guard against the
  2026-05-29 crash artifact noted in [[project_dualres_transformer_result]]).
- **LONG = control:** expected DEAD (longs net-negative across the suite). A fail is the expected
  outcome, not to be re-spun as success.
- If neither side clears: record the dead-end and stop. No config sweeps to chase it.

## Known limitations (stated up front — discipline rule #4)
- **Survivorship (audited 2026-06-12):** the panel universe is a **fixed 172-ticker set**
  (`data/raw_upstox_cache_15min_3y/*.csv`) — 170/172 span the full 2023-01→2026-06 window,
  only 2 late listings and ~1 genuine mid-panel delisting (retained). This is ≈today's
  constituents backfilled, **not** point-in-time index constituents → mild survivorship bias.
  Bounded for an intraday cross-sectional ranker, but **any Phase-4 / Gauntlet / deploy claim
  must carry this caveat.** NOT verified survivor-free.
- Information ceiling: the panel is information-limited, not capacity-limited
  ([[project_dualres_transformer_result]] — data-size ablation flat). A better loss/decomposition
  may sharpen ranking but is unlikely to break the 10 bps cost wall; the real lever is
  order-flow/microstructure data ([[project_cst_stage0_killed]]).

## Baselines for comparison (Phase 3)
- Single-head BCE transformer (`artifacts/dualres_transformer.pt`).
- netPnL veto short numbers (`n=493 / +1.4 bps net@10`).
- v10 XGBoost short ranker (production short signal).

---

## ✅ RESULTS & VERDICT (2026-06-12, ListNet arm) — ❌ KILLED

OOS test: 530 timestamps, **2025-11-28 → 2026-06-04** (chronological 15% tail, embargo 30).
Cost-sign sanity passed everywhere (`median(net−gross) == −cost` = −10.00). Artifacts:
`artifacts/dualres_short.pt`, `artifacts/dualres_long.pt`, `artifacts/dualres_sided_eval.json`.

| Model | rank-IC ρ (t) | K1 net@10 (t) | K3 net@10 (t) | K5 net@10 (t) |
|---|---|---|---|---|
| **dualres_short** | **+0.0058 (1.05)** | +0.78 (0.13) | −5.17 (−1.72) | −8.54 (−3.60) |
| **dualres_long** (control) | −0.0041 (−0.78) | −14.60 (−2.57) | −12.98 (−4.22) | −11.55 (−4.87) |

**SHORT fails the kill bar on BOTH conditions:** ρ = 0.006 is **~9× below** the 0.05 floor, and the
only positive cell (K1 net@10 +0.78 bps) has **t = 0.13 = noise**. The marginal K1@6 (+4.78, t=0.80)
is the single most-confident short/hour at low cost only — not significant, decays to negative by K3.
**LONG is DEAD** as pre-registered (negative ρ, all net significantly negative).

**Conclusion:** the side-specialized transformer ranker has **no post-cost edge**; the long/short
decomposition does **not** rescue it. This is fully consistent with [[project_dualres_transformer_result]]
(single-head was sub-cost) — the binding constraint is **information, not architecture or head
structure**. Per the pre-registered stop rule: **dead-end recorded, no config sweep.**

**Loss-family note:** ListNet was used here; switching to `rank:pairwise` (to match v10) is **not**
pursued — repo evidence shows listwise LambdaMART (v12–v15) never beat v10 pairwise, and v10 pairwise
itself is only ρ≈0.025 raw and net-negative after costs. A loss swap cannot turn ρ=0.006 into ≥0.05;
chasing it would be the "drift" the experiment was scoped to avoid. The real lever remains
order-flow/microstructure data ([[project_cst_stage0_killed]]).

---

## 🔁 OBJECTIVE CORRECTION + CUSTOM GATE LOSS (2026-06-12) — FAINT, SUB-SIGNIFICANCE

The listwise arm answered the wrong question. The real objective was a **per-side strength/conviction
estimator used as a VETO on v10's picks**, judged by **uplift to v10's net** — not standalone rank-IC.
Built a **custom cost-aware coverage-budgeted gate loss** for that: `g=sigmoid(score)`,
`n=side_sign·y−cost`, `L = −mean(g·n)·1e4 + λ·(mean(g)−ρ)²` (ρ=0.70 keep budget, λ=100). It optimizes
the keep/veto decision directly and is cost-aware. Two variants trained per side.

**Faithful eval = gate-as-veto on v10** (`scripts/transformer/gate_veto_v10.py`, OOS 2025-11-28→2026-06-04,
69,235 v10 rows, keep 70%). Short-side net Δ vs v10-alone:

| Variant | K1@6 | K3@6 | K5@6 | best cell sig |
|---|---|---|---|---|
| plain gate (full xsec) | +2.75 | +1.46 | +0.32 | K1@6 +6.54, **t=1.45** |
| v10-focused gate (`--v10_restrict`) | +2.04 | +1.99 | +1.62 | K1@6 +5.83, **t=1.21** |

**Findings:** the gate loss **works and is the right tool** — unlike the listwise arm (rank-IC≈0), it
produces a **consistent, correctly-signed short-side loss-reduction** at every K/cost, and restricting
to v10's picks made the uplift consistent across K (~+1.6 to +2.0 bps) instead of decaying. The single
best cell (short K1@6) edges past the prior single-head veto (+5.4). **But the kill bar is NOT met:**
no short cell reaches t>2; at the binding 10bps cost the gated short is +1.83 (t=0.38) / −6.38 / −6.14 —
not significantly positive. Long side dead throughout.

**FINAL VERDICT (whole arc): ❌ does not clear the bar.** Three arms — listwise ranker (dead), plain
cost-aware gate, v10-focused gate (best) — converge: the custom loss extracted the most that's there,
and it's a **faint, consistent, but sub-significance / sub-cost** short veto. The loss choice was
vindicated (it was the lever the user correctly flagged), but the **binding constraint is information,
not loss or architecture**. Per the pre-registered stop rule: recorded, **no keep_rate/λ sweep**. Way
through remains order-flow/microstructure data ([[project_cst_stage0_killed]], [[project_dualres_transformer_result]]).
Reusable: `train.py --objective gate [--v10_restrict]`, `make_v10_pickmask.py`, `gate_veto_v10.py`,
`artifacts/gate_veto_v10.json`.

### Net win-rate verification (hit-rate ≠ edge)
The v10-focused gate's short K1 picks (n=316) have raw directional WR **58%** (real on its own:
z≈2.85 vs 50%), but the **after-cost** win-rate collapses as the fee bites:

| K1 short, cost | raw WR | net WR | net bps (t) |
|---|---|---|---|
| @6 | 58% | 55% | +5.83 (1.21) |
| @10 | 58% | **51%** | +1.83 (**0.38**) |
| @20 | 58% | 44% | −8.17 (−1.69) |

At the binding 10bps cost the 58% becomes a **51% coin flip** and the P&L is noise (t=0.38). The
~7 points between raw and net are wins too small to clear the round-trip fee; the loss-side fat tail
(short squeezes) dominates the mean. Same lesson as [[feedback_validate_cost_accounting]] and the TBM
"56.5%" mirage — **judge net bps + t per side, never win-rate.**

### Gemini-veto "rescue" — REJECTED (do not re-propose without a forward test)
Proposed: stack the live Gemini S1/S2 veto on the gate to cut its losing shorts. Rejected:
1. **No measured uplift** — Gemini is already layered on v8/v10 live, which remain net-negative.
2. **It is the closed stacking line** — [[02 — Models/Meta-Veto/Meta-Veto Rectification Plan MV2|MV2]] permanently
   closed price/volume/macro/sentiment stacking (best meta-model net −5 bps).
3. **Unbacktestable** — the S2 layer uses Google Search grounding; querying *today* about a past date
   leaks the future → any historical backtest is self-deception.
Only honest path is a **forward/paper test**: run the gate live, log Gemini's decisions on the
gate-kept shorts in real time, measure net-with vs net-without to t>2. See
[[06 — Logs/Conversations/Conv-2026-06-12-Sided-Transformer-Gate|conversation log]].
