---
title: Conviction Caps & Long-Side Filter
type: research
status: active
updated: 2026-07-10
model: v20_rolling_1h
verdict: "⚠️ UNVERIFIED (no Gauntlet). SHORT: signal is inverted-U in conviction — tradeable band [0.011,0.040], cap >0.040 fails; capping lifts the top-1 short stream break-even(−0.3)→+3.2 net@6. LONG: anti-selected (top-1 long −11.3, worse than a random long −8.7; the model's highest long_score/conv picks are its worst); NO feature filter makes longs robustly net-positive. BEST long lever = the NIFTY TRAILING-2h INDEX gate: keep longs only when the index is RISING (idx2h≥0) — on LIVE shadow (07-06→07-10) it lifts the long book −0.2→+7.9 net@6 (n=36, 61% kept), whole edge = staying out of non-rallying days (07-08). Threshold moved between reconstruction (≥0.3%) and live (≥0%) → robust finding is directional only. 2h de-dupe = hygiene not alpha (hurts the gated book). Gate longs, de-weight; let the working short book carry."
---

# Conviction Caps & the Long-Side Filter — OOS 2026-07

**Verdict (⚠️ UNVERIFIED — no Gauntlet, exploratory).** On the live model **v20_rolling_1h**, over
a clean OOS window (**2026-06-16 → 07-09**, 17 sessions the model never trained on — v20 `trained_at`
2026-06-15; 289 anchors, every 15-min 10:15→14:15, 1h hold), the short leg has a real conviction
structure we can exploit; the long leg is structurally sub-cost and its own confidence is inverted.
Conviction = the live centered long−short spread (`model_inference.py:218-222`;
`Long_Conviction = (long_score−mean) − (short_score−mean)`, `Short_Conviction = −Long_Conviction`).
Selection of the top-1 pick per side is by **raw** score (live `SIGNAL_RAW_SCORE_ONLY`); the
band/cap/floor thresholds are applied to the **combined conviction**.

Scripts: `scripts/backtests/short_conv_gt04_oos.py`, `conviction_caps_oos.py`,
`long_filter_search_oos.py`. Panels: `data/backtests/conviction_caps_oos_2026-07-10.csv`,
`long_filter_top1_2026-07-10.csv`, `long_filter_pool_2026-07-10.csv`.

## SHORT — inverted-U in conviction (a real lever)
Short edge **rises** with conviction to a peak at ~0.02–0.03, then **INVERTS** past 0.04 (over-extended
shorts mean-revert — same phenomenon as [[Loss-Cutting & Day-Regime Research|the mean-reverting universe]]).

| Short conviction | full pool net@6 | note |
| --- | --- | --- |
| [0.011, 0.020) | ~−1.5 to −4 | sub-cost (real gross edge, doesn't clear 6bps) |
| **[0.020, 0.025)** | **+6.9** (t/2 +0.9) | peak; top-1 stream +11.6 |
| [0.030, 0.040) | noise (n=2) | |
| **[0.040, 0.060)** | **−29.8** (WR 25%, t/2 −1.5) | the failure — CAP here |
| ≥ 0.060 | −8.5 | noisy |

- **conv>0.04 shorts fail in BOTH arms** (top-1 AND not-top-1): ARM A (is top-1) n=49 net@6 −16.0;
  ARM B (not top-1) n=12 net@6 −20.6 — failure is a property of the conviction level, not the pick.
- **Actionable:** cap short conviction at **0.040** → top-1 short stream **−0.3 → +3.2 net@6**
  (n 289→239, WR 57%→61%, drops 17%). On the traded stream the **floor is moot** (top-1 shorts almost
  never have conv<0.011); the cap is the whole lever. Registered in [[00 — Start Here/Ray of Hope]].
- **Floor sweep** (full pool, cumulative conv≥f, upper cap held): SHORT crosses net-negative below
  floor ≈0.017, LONG below ≈0.024. **Uncapped, neither side is net-positive at any floor** — the CAP
  does the work.

## LONG — anti-selected; no filter makes it net-positive
- **The top-1 long is anti-selected:** baseline top-1 long **−11.3 net@6** is *worse* than a random
  long candidate (pool avg −8.7). The model's **highest** long_score / conviction picks are its
  **worst** (long_score top tercile −21.9, conviction top tercile −25.3) — over-extension that reverts.
- **Robust "cut the bad longs" discriminators** (cross-validated top-1 tercile + full-pool direction):
  1. **Over-extension** — cap conviction ≤ 0.030; distrust the raw top long_score.
  2. **Small-caps** — `mcap_rank` is the single most robust discriminator (full-pool Spearman −0.019,
     **t/2 −2.1**; small-cap top-1 tercile −25.6 vs mid-cap −0.8). Cut `rank > 100`. Uses the weekly
     market-cap rank (`data/marketcap_ranks.json`, `scripts/fetch_marketcap_ranks.py`).
  3. **Below-VWAP in a down-market** — worst cell (pool grid: mkt-up×above-VWAP −7.5 … mkt-down×below-VWAP −10.6).
  4. **Afternoon** — longs die after ~12:15 (12:30 −24, 13:15 −29, 13:45 −25).
- **Recommended long gate:** take a long only if `VWAP_Dist ≥ 0` **AND** `Market_Mean_Return ≥ 0`
  → traded top-1 stream **−11.3 → +2.5 net@6** (WR 48→54%), keeps only **28%** of longs.
  **⚠️ Honest caveat:** on the full pool the same gate is still **−7.5** → the +2.5 is largely
  small-sample; read it as *"removes most of the long drag"*, not *"positive long alpha."*
- **Strategic take:** the long leg is information-limited and sub-cost under every robust filter
  (consistent with the whole [[00 — Start Here/Dead-Ends Register|dead-ends record]]). The efficient
  move is to **gate longs hard and de-weight them, letting the working short book carry the P&L.**

## LONG — the Nifty trailing-2h INDEX-MOMENTUM gate (best long lever found; 2026-07-10)
User hypothesis: *cut longs when Nifty soared >0.5% over the trailing 2h* (over-extension). **Tested and
INVERTED** — longs are best when the index has been **RISING** (trend, not reversion), worst when falling.
Monotonic on the full pool (net@6): idx2h<−0.5% **−46**, [−0.5,0) −24, [0,+0.5) −5, **[+0.5,+1.0) +33**.
So the rule is *keep longs only when the index is up*, not cut them. Index measured on **15-min ^NSEI**
over a 2h window; for early anchors (<11:15, window reaches pre-open) the overnight **gap is counted as
~1h** (prev-close reference). `scripts/backtests/nifty_2h_long_gate_oos.py`, `long_dedupe_2h_oos.py`.

- **Reconstruction OOS (06-16→07-09, 17d):** keep `idx2h≥+0.3%` → 47 longs, **+23.5 net@6, WR 77%,
  +₹10,973** (~11% on live notional); `idx2h≥0` → 165 longs, +0.4 (break-even). Mechanism = market-beta
  / intraday-trend timing (BOTH sides do better in the rising-index regime), NOT stock selection.
- **LIVE shadow validation (07-06→07-10, real fills, `vanguard_trades.db`):** ALL longs net@6 **−0.2 →
  `idx2h≥0` gate +7.9** (n=36, 61% kept); `idx2h≥+0.3%` **too aggressive on live (−2.4, n=11)**. The
  whole edge = **staying out of 07-08** (Nifty didn't rally; gate cut all 6 of that day's −77 net@6).
  ⚠️ **The best threshold MOVED between reconstruction (0.3%) and live (0%)** → the robust finding is
  only DIRECTIONAL: *longs need trailing-2h Nifty ≥ 0*; the exact cutoff is not nailed down (5-day / small n).
- **2-hour per-ticker DE-DUPE** (live buys a name once per ~2h): does NOT rescue longs — full book
  RAW −11.3 → SKIP −7.3 (only via fewer trades) / REPLACE −10.5 (negligible). It **HURTS the gated book**
  (≥0.3%: +₹10,973 → SKIP +₹7,094 / REPLACE +₹6,069) because on trend days it throws away *winning*
  repeated buys (06-24 cut 9→2). De-dupe = tail-risk/concentration hygiene, not alpha.
- **Why 06-16 & 06-22 flipped negative** (the gate's worst days): both were up-gap-fires-gate but **no
  intraday follow-through** (only early gap anchors qualified). NOT a market reversal (index flat during
  holds; corr(index-in-hold, longP&L)=−0.06) — the *stocks* fell while the index held = negative stock
  alpha. The ranker **fixated on falling knives**: 06-16 GRASIM ×5 (−159bps = 26% of day loss; GRASIM
  intraday −1.14%, RBLBANK −2.4%), 06-22 TATASTEEL ×5 / metals. Mean-reversion long premise failed
  because these were genuine intraday downtrends, not bounce-able dips (contrast a WINNING up-gap day
  07-03: POLICYBZR opened −3.5% capitulation → bounced +146). Index gate is blind to the picked stock.
- **Deploy as:** default-OFF reversible `LONG_INDEX_GATE` at the **≥0** threshold (shadow-log which longs
  it cuts), 2–4wk A/B before trusting a number. Registered in [[00 — Start Here/Ray of Hope]] Tier 3.

## Cross-check that anchored all of this (methodology note)
- **Active model is v20_rolling_1h** (`models/registry.json`); **v21 is NOT deployed** (research-only).
- **Candle-reconstruction backtests that enter at the :15 candle close understate the LIVE shadow
  short book by ~20–40 bps** (live enters at scan-time ~:17). The DB shadow log
  (`vanguard_trades.db`, `final_profit_pct` = clean 1h gross for VETOED_EXPIRED) was net-**positive**
  last week (+7 net@6 ALL, shorts +12) — but that hinges on the favorable scan-time fills, i.e. the
  same execution-sensitivity flagged for [[Open Window Trade Stacking|gap-fade]]. 2026-07-08's
  "−20% day" in reconstruction was mostly a live-outage day (only 13 signals logged) plus
  reconstruction losses on afternoon anchors the live book never traded.

Related memory: `project_short_conviction_inverted_u`. Conforms to the intraday mandate
(1h holds, 10:15–14:15).
