---
title: "Ray of Hope — Positive & Promising Edges"
type: reference
status: active
updated: 2026-06-15
tags: [register, positive, edges, alpha]
---
# 🌅 Ray of Hope — Positive & Promising Edges

> The **positive mirror of the [[00 — Start Here/Dead-Ends Register|Dead-Ends Register]]**. Everything
> here has shown a *real* positive signal — net-of-cost, certified, or robust research. Tiered by how
> trustworthy/tradeable it is. **Discipline:** every metric cites a Gauntlet `run_id` or is marked
> `⚠️ UNVERIFIED` (research, no Gauntlet verdict). Binding cost = **10 bps** unless noted.
>
> _Dead lines (transformers, directional classifiers, TBM 1h, co-sign, GCN, …) are NOT here — see the
> Dead-Ends Register._

---

## 🟢 Tier 1 — Net-positive TRADES (clears cost)

| Edge | What it is | Result | Status |
| --- | --- | --- | --- |
| **daily_macro_v2** | 10y daily macro/breadth gatekeeper, multi-day hold | **LONG `TRIGGER_GRADE`, SHORT `FILTER_GRADE`** — ≈ **+28 bps / 3-day-trade** (0.565%) | ✅ Certified `run 20260610T135608Z-5f7d069f` |
| **09:15 Overnight-Reversal Short** | SHORT yesterday's winners at the 09:15 open, hold to 15:15 close (k=10) | **net +10.6 bps/trade** @10bps, t≈4.5; market-neutral skill +12.4 bps gross; stable across halves, 2 datasets | ⚠️ UNVERIFIED (research, no Gauntlet). **Execution-constrained: entire edge is in 09:15→09:30** (needs open-auction fills; post-09:30 sub-cost). See [[project_intraday_overnight_reversal_edge]] |

> **Caution on hold period:** daily_macro_v2's edge lives in the **3-day** hold — shrinking to 1 day
> drops it to ~7 bps (loses significance). Deploy v2-alone on the 3-day hold.

---

## 🟡 Tier 2 — Certified FILTERS (positive ranking signal, not standalone trades)

These earned `FILTER_GRADE` in the Gauntlet (real cross-sectional ranking skill) but are **sub-cost as
standalone triggers** — use them to gate/filter, not to fire trades alone.

| Model | Verdict | Gauntlet run_id |
| --- | --- | --- |
| **v10_native_1h** (deployed 1h ranker) | L + S `FILTER_GRADE` | `20260610T184210Z-d795438c` |
| **v2_15min_3y** | L + S `FILTER_GRADE` | `20260610T173707Z-d795438c` |
| **v3_15min_clean** (serve-consistent 15m) | L + S `FILTER_GRADE` | `20260610T113721Z-5f7d069f` |
| **daily_macro_v3** | SHORT `FILTER_GRADE` (long DEAD) | `20260610T144343Z-5f7d069f` |
| **v20_rolling_1h** (rolling-1h ranker @ v10's :15 cadence) | L + S `FILTER_GRADE` | `20260615T175149Z-5f7d069f` — **certified peer of v10**: ≈ same net bps (both sub-cost), short ρ modestly higher (~0.029 vs 0.025), Top-1 short net less negative (−3.3 vs −6.3 bps). Certified on the NON-overlapping :15 cadence ONLY (the overlapping 18/day version can't be validly Gauntlet-graded). See [[04 — Research/V20 Rolling-1h Overlapping-Window Model]] |
| **v8_upstox_3y** | L + S `FILTER_GRADE` *static* | `20260610T172623Z-d795438c` — ⚠️ **DEMOTED**: purged WF shows decaying ρ + net-negative Top-1/3; dashboard-only. See [[project_v8_1h_walkforward_demoted]] |

---

## 🔵 Tier 3 — Promising but UNCERTIFIED (research; sub-cost filters / enhancers)

| Candidate | Signal | Status |
| --- | --- | --- |
| **Gate-1 graph features** (business-group + sector relation graph, 1-layer message-pass) | WF-robust rank-IC enhancer (BASE 0.017 → 0.027, 4/5 folds, survives neg-control). Gain mostly sector. | ⚠️ UNVERIFIED — research. Short net-negative; not a standalone trade. See [[project_gate1_graph_features]] |

---

## 📌 Reading guide

- **Want to actually make money** → Tier 1 only (daily_macro_v2 3-day; 09:15 reversal short with open-auction execution).
- **Want a ranking filter / gate** → Tier 2 (certified) or Tier 3 (promising, pending certification).
- **The recurring ceiling:** plain 1h/15m next-bar price/volume rankers are **sub-cost** — the lever for new
  *tradeable* alpha is new data (order-flow / microstructure), not more architecture. Tier 3 graduates to
  Tier 2 only via a Gauntlet run.
