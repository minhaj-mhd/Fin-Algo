---
title: "Conv-2026-07-02-News-Shock-Event-Study"
type: log
status: concluded
updated: 2026-07-02
tags: [research, news, event-study, dead-end]
---
# 💬 Conversation Context: News-Shock Proxy Event Study

## 📌 Metadata
- **Conversation ID**: b405fa1f-ae08-4205-a37f-ea57788003fd
- **Start Date**: 2026-07-02
- **Status**: 🔴 Concluded
- **Focus Area**: Research — can we trade on news (frequency + drift/fade)?

## 🎯 Objectives
- [x] Estimate how many tradeable news events/day exist in our universe
- [x] Measure whether price DRIFTS or FADES after news arrival, net of cost

## 💻 Active Code Files Modified
- [news_shock_event_study.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/news_shock_event_study.py) (new, exploratory)
- Outputs: `data/research/news_shock/{report.txt, cell_summary.csv, events_per_day.csv}`

## 📝 Compacted Session Log
- **Question**: user asked "if we take trades by news, how many trades can we take daily?" and then "so there is no edge in that as well?"
- **Method**: no timestamped news feed exists, so used a **news-shock price proxy**: |idiosyncratic 5-min return| ≥ thr (stock ret minus cross-sectional median that bar → market moves excluded) AND rvol ≥ thr (vs trailing 20-day same-slot median). Data = `data/raw_upstox_cache_5min_v3` (147 tickers, 850 sessions 2023-01..2026-06, split-adjusted). Hygiene: intraday-only returns (no overnight/gap contamination), entries 09:30–14:30 (open window excluded — that's gap-fade territory), de-clustered to first event per ticker-day, |ret|>15% guard, instant vs +5-min-delayed entry, same-ticker/same-slot negative control.
- **Frequency answer**: material shocks (≥1.5% idio / 5x vol): **mean ~2.0/day, median 1, p90 4, ZERO events on 29% of days**. Milder ≥1%: ~6/day. Big ≥2%: 0.85/day; ≥3%: 0.3/day. News trading is structurally LOW-frequency on a 147-name universe.
- **Drift vs fade**: sign-aligned forward returns are **NEGATIVE at every threshold and horizon** → price **FADES** after the shock. News MOMENTUM (buy pos news / short neg news) is gross-negative even before costs (canonical cell: −6.8 bps @1h, −15.3 @2h instant entry). Both directions (pos-news longs AND neg-news shorts) negative.
- **Negative control clean**: non-event bars same ticker+slot ≈ 0 bps → the fade is event-specific overreaction, NOT generic mean reversion. Same phenomenon as the open gap-fade, at intraday timescale.
- **Can we FADE it instead?** Gross yes: +8.8 bps @15m (t 2.3), +15.3 @2h (t 2.4, day-clustered t 1.9) instant entry. But most of the reversal happens in the FIRST 5 MINUTES: with honest +5-min delayed entry the 2h fade grosses only +10.1 (t 1.7) → **net@10 ≈ 0, net@6 ≈ +4 sub-significance**; concentrated in 2025–26 (2023–24 ≈ 0). Same latency wall as gap-fade (where only the auction print pays) and the announcement game generally.
- **Verdict (⚠️ exploratory, no Gauntlet run)**: news-momentum DEAD (gross-negative); news-fade real-but-uncapturable at retail latency (~0 net after honest entry + costs). Unified picture: **this universe systematically OVERREACTS to information at every timescale; the only capturable expression found so far is the opening-auction gap-fade, because the auction mechanism hands you the extreme fill.**

## 🔗 Core Memory Links & Backlinks
- [[06 — Logs/Conversations/Conv-2026-07-02-Open-Window-Trade-Stacking|Gap-fade conversation]] — same overreaction phenomenon at the open
- Related dead-ends: stop-loss research, day-regime detection, horizon sweep (see Active Board 2026-06-30 entries)
