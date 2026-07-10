---
title: "Conv 2026-07-05 — Short/Long Exclusion Rule Engine feasibility"
type: log
status: concluded
updated: 2026-07-05
model: v20_rolling_1h
verdict: DEAD (alpha version); risk-gate version viable but sub-cost
---
# 💬 Conversation Context: Per-hour ticker EXCLUSION rule engine before ranking

## 📌 Metadata
- **Start Date**: 2026-07-05
- **Status**: 🔴 Concluded (research, NEGATIVE for the alpha framing)
- **Focus Area**: Ranking universe conditioning / rule-based pre-filter

## 🎯 Objective
Research a rule engine that, each hour, **removes tickers that shouldn't be shorted (or longed) for the next 1–2h**, feeding the cleaned universe to the existing v10/v20 ranker (short pool and long pool filtered independently).

## 📝 Compacted Session Log
- **Setup**: scored the LIVE `v20_rolling_1h` model on `data/research/v20_rolling_1h/panel.parquet`
  (features already cross-sectional z-scored per hour; scaler.pkl is an unfitted placeholder → scale-invariant XGB).
  Pseudo-OOS = last 30% of dates (2025-01→2026-06), non-overlapping :30 windows, top-5 books/side, 10bps/trade,
  next-hour return winsorized ±15%. Controls: random-exclusion of matched count; IS(H1)/OOS(H2) split.
- **Finding 1 — intuitive rules are WRONG-SIGNED.** Universe mean-reverts, so the extended movers a naive engine
  would drop are the model's BEST trades: short near-52w-high + RVOL>1 = **+0.62** net vs −3.40 rest; long
  down-streak + RVOL>1 = **+6.08** vs −7.67. Excluding them removes the fade/bounce edge.
- **Finding 2 — no rule is learnable.** All 86 features × both tails, threshold fit on H1, tested on H2:
  **corr(IS exclusion-gain, OOS exclusion-gain) ≈ 0** (+0.12 short / −0.02 long). Big IS gains collapse OOS.
  Nothing beats random-exclusion → pure deleveraging/overfit (the stop-loss-research trap).
- **Finding 3 — only real conditioning is time-of-day.** Mid-day (11–13:30) decays; short 14:30 +0.64, 12:30 −5.62.
  Excluding mid-day lifts short −3.20→−1.60, long −7.03→−5.74 — but it's a TIME gate, drops 60% of trades, and BOTH
  books stay negative. Not per-ticker; doesn't cross cost. (Matches the day-regime dead-end.)
- **Live tie-in**: deployed `FADE_QUALITY_GUARD` (config.py — within 0.5% of 52w-high + RVOL≥1.5 blocks the short)
  targets exactly the +0.62 breakout-shorts; population test says it likely removes the best shorts. It's a tail-RISK
  guard (RELAXO-class execution disasters), not a mean-edge improver — re-fit/backtest before trusting.
- **Only surviving "possible way"** = reframe from ALPHA filter → **risk/tradability gate** (F&O ban, earnings/results
  in-hold, ex-date, circuit/band lock, index add/drop, illiquid/wide-spread). Needs NEW data not in repo. Trims tail
  risk & bad fills; does NOT make a sub-cost book profitable. Binding constraint stays the 10bps cost; the open
  gap-fade remains the one place execution hands you edge.

## 💻 Artifacts (scratchpad, research-only)
- rule_engine_probe.py (conditional + substitution vs random), rule_engine_probe2b.py (sweep + IS/OOS stability),
  rule_engine_probe3.py (time-of-day / liquidity / reversion-concentration). No production state mutated.

## 🧭 Expanded scope — full "Candidate Gating Engine" research program (2nd goal)
- **Learned meta-gate (López de Prado meta-labeling) — DEAD.** XGBoost classifier predicting P(ranker's pick profitable) on top-30% candidate zone; feats = 86 ranker + causal market-context (breadth/dispersion/mkt_vol/mkt_ret/hod); WF train→2025-06. OOS hit-rate **0.529 long / 0.516 short**; gate−random −0.33/+0.56 bps, 95% CI spans 0. Predicting "when the ranker is right" ≈ predicting returns (same info). Importances tiny, time-dominated.
- **Robustness correction:** long reversion-selection beats random-same-size **4/5 years** (+1.3bps avg) = genuine info, but ≈ cancels the pool-shrinkage handicap → ≈0 vs full-universe baseline (earlier "OOS-stable edge" was 2025-weighted overstatement).
- **Regime conditioning (dispersion/vol) UNSTABLE:** high-disp short +5.9/+5.1bps (2022/24) → −12.1 (2026); sign reverses. Time-of-day is the only robust conditioning (still sub-cost).
- **Net verdict:** gating can't manufacture the missing edge — gate for RISK (events) & COST (execution/liquidity), not alpha; real lever is new INFORMATION (order-flow/OI). Full memo delivered as an **Artifact** (verdict ledger, 12-area taxonomy, validation protocol, phased P0–P3 roadmap, literature). Roadmap: P0 risk/event data+gate → P1 cost/liquidity+open-auction → P2 regime sizing (year-by-year) → P3 new-data axis (then re-test learned gate).
- Scratchpad probes: rule_engine_probe{,2b,3,4}.py, gate_learned.py, gate_robustness.py.

## 🔗 Core Memory Links
- [[00 — Start Here/Dead-Ends Register]] · confirms the 1h info-ceiling (CST, DualRes, sided-transformer, gate lines)
- Reversion evidence: open gap-fade, news-shock fade, stop-loss, day-regime research.
