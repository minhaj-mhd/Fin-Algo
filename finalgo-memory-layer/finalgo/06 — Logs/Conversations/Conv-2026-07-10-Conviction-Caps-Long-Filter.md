---
title: "Conv 2026-07-10 — Conviction caps, long-side filter, market-cap rank"
type: log
status: concluded
updated: 2026-07-10
model: v20_rolling_1h
verdict: "⚠️ UNVERIFIED (exploratory, no Gauntlet). SHORT signal inverted-U in conviction: cap >0.040 (fails both top-1 & not-top-1); capping lifts top-1 short −0.3→+3.2 net@6. LONG anti-selected (top-1 −11.3 worse than random long −8.7; model's most-confident longs are worst); best gate VWAP≥0 & mkt-up removes drag (top-1 −11.3→+2.5) but pool still −7.5 → gate hard + de-weight, let shorts carry. Active model = v20 (v21 not live). :15-close reconstruction understates live shadow shorts ~20-40bps (scan-time fills)."
---
# 💬 Conversation Context: Conviction caps + long-side filter + market-cap rank

## 📌 Metadata
- **Start Date**: 2026-07-10
- **Status**: 🔴 Concluded
- **Focus Area**: Signal layer — conviction thresholds, long-leg gating, dashboard feature

## 🎯 Objectives
- [x] Backtest a conviction-based long/short entry-selection rule (band [0.011,0.040], take-both-if-similar) with per-day + time-of-day splits
- [x] Cross-check the backtest against the live shadow trades (they disagreed)
- [x] OOS-test the "shorts with conviction >0.04 fail" hypothesis (top-1 vs not-top-1)
- [x] Find conviction caps + floors for BOTH sides
- [x] Add weekly market-cap rank for the 172 tickers, shown in a bracket, wired into the dashboard
- [x] GOAL: find efficient ways to cut the long book down to only good signals

## 💻 Active Code Files
- `scripts/backtests/conviction_select_1h_backtest.py` (reconstruction), `conviction_on_shadow_1h.py` (shadow ground-truth)
- `scripts/backtests/short_conv_gt04_oos.py`, `conviction_caps_oos.py`, `long_filter_search_oos.py`
- `scripts/fetch_marketcap_ranks.py` (+ `data/marketcap_ranks.json`, `data/marketcap_history/`)
- `scripts/vanguard_dashboard.py`, `templates/vanguard_v2.html`, `templates/ticker_detail.html`

## 📝 Compacted Session Log
- **Reconstruction vs shadow — the fill gap.** A :15-candle-close reconstruction of the top-1 L/S stream
  showed the whole book net-negative (−12 to −25%/wk, dominated by 07-08). The **live shadow log**
  (`vanguard_trades.db`) told the opposite: **+7 net@6 ALL, shorts +12**, 07-08 actually +4.4. Cause:
  live enters at scan-time (~:17), reconstruction at the :15 close → **~20–40 bps favorable to live
  shorts**; plus 07-08 was a partial-outage (13 signals logged, afternoon anchors never traded). 07-09
  alignment: longs ~85% identical ticker-for-ticker → pipeline is faithful; the gap is fill-basis, not model.
- **Model check:** active model = **v20_rolling_1h** (`models/registry.json`); **v21 NOT deployed**.
  Conviction = centered long−short spread (`model_inference.py:218-222`); parity validated (my CANBK
  14:15 conviction 0.013464 vs live-logged 0.013286).
- **SHORT conviction is inverted-U (OOS 06-16..07-09):** edge peaks at conv 0.02–0.03, INVERTS >0.04
  ([0.04,0.06) net@6 −29.8, t/2 −1.5). conv>0.04 fails as top-1 (n=49, −16.0) AND not-top-1 (n=12, −20.6).
  **Cap at 0.040 → top-1 short stream −0.3 → +3.2 net@6** (drops 17%). Floor is moot on the traded stream.
- **LONG is anti-selected:** top-1 long −11.3 net@6 is WORSE than a random long (−8.7); the model's
  highest long_score/conviction picks are its worst (over-extension). No filter makes longs robustly
  net-positive. Robust cuts: small-caps (`mcap_rank` t/2 −2.1), conv≤0.030, below-VWAP+down-market,
  afternoon. Gate `VWAP≥0 & Market_Mean_Return≥0` → top-1 −11.3→+2.5 (keeps 28%) but **pool still −7.5**
  (small-sample). Conclusion: gate longs hard + de-weight, let the working short book carry.
- **Market-cap rank feature (shipped):** `scripts/fetch_marketcap_ranks.py` fetches name + market cap
  (yfinance) for all 172 tickers, ranks 1=largest, persists `data/marketcap_ranks.json` + weekly archive
  `data/marketcap_history/marketcap_ranks_<ISOWEEK>.json` (manual weekly run). Dashboard wired: new
  `/api/marketcap_ranks` endpoint + `withRank()` helper renders `TICKER (rank)` at all ticker-name spots
  in `vanguard_v2.html`; `ticker_detail.html` header shows `(MC #rank)`. Verified via Flask test client.

- **LONG index-momentum gate (the best long lever found).** User idea "cut longs when Nifty soared
  >0.5%/2h" tested and INVERTED — longs are best when the index has been RISING (trend), worst falling
  (full-pool monotonic net@6: idx2h<−0.5% −46 … [+0.5,1) +33). Gate = keep longs only when trailing-2h
  Nifty ≥0 (15-min ^NSEI; early anchors count the overnight gap as ~1h via prev-close). Reconstruction
  17d: ≥0.3% → 47 longs +23.5 net@6 / +₹10,973; ≥0 → break-even. **LIVE shadow 07-06→07-10 (real fills):
  ALL longs −0.2 → ≥0 gate +7.9 net@6 (n=36, 61% kept); ≥0.3% too aggressive on live (−2.4, n=11).
  Whole edge = staying OUT of 07-08 (Nifty flat; gate cut all 6 of that day's −77).** Best threshold
  MOVED recon(0.3%)→live(0%) ⇒ robust finding DIRECTIONAL only. Mechanism = market-beta/trend timing.
- **2-hour per-ticker de-dupe** does NOT rescue longs (full book −11.3→SKIP −7.3 via fewer trades) and
  HURTS the gated book (throws away winning repeats on trend days: ≥0.3% +₹10,973→SKIP +7,094). Hygiene, not alpha.
- **06-16 & 06-22 flip diagnosis:** up-gap fires the gate but no intraday follow-through; NOT a market
  reversal (index flat, corr(index-in-hold, longP&L) −0.06) — the ranker fixated on falling knives
  (GRASIM ×5 −159bps=26% of day; TATASTEEL ×5 / metals). Mean-reversion long premise fails on genuine
  downtrends; contrast 07-03 POLICYBZR opened −3.5% capitulation → bounced +146.

## 🔗 Core Memory Links & Backlinks
- [[04 — Research/Conviction Caps & Long-Side Filter (OOS 2026-07)]]
- [[00 — Start Here/Ray of Hope]] (short-conviction-cap lever registered)
- Related: [[04 — Research/V20 Rolling-1h Overlapping-Window Model]], memory `project_short_conviction_inverted_u`
