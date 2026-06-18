---
title: "Conv 2026-06-12 Daily Transformer Veto Overlay on daily_macro_v2"
type: log
status: concluded
updated: 2026-06-12
model: daily_macro_v2
verdict: "⚠️ UNVERIFIED (exploratory, no Gauntlet run) — daily veto does NOT significantly improve v2 net; dead-end"
---

# 💬 Conversation Context: Daily Transformer Veto Overlay on `daily_macro_v2`

## 📌 Metadata
- **Start Date**: 2026-06-12
- **Status**: 🟢 Active
- **Focus Area**: Model Suite (daily 3-day ranker), cross-sectional transformer

## 🎯 Objective
Build a **daily-only** cross-sectional transformer that acts as a **cost-aware veto/filter overlay** on
`daily_macro_v2` (the only Gauntlet-certified post-cost edge: LONG TRIGGER / SHORT FILTER, run
`20260610T135608Z-5f7d069f`), to raise its net edge. User decisions (locked): veto overlay (not
standalone/long-specialist); daily-only resolution (no 1h/15m); build full → audit → one Gauntlet run.

## 🔬 Pre-registered hypothesis & stop rule
- **H1:** daily gate transformer, trained with the cost-aware gate loss, vetoes v2's losing Top-K picks
  to raise net.
- **Primary metric:** Δnet = net(v2+veto) − net(v2 alone) per side on the common purged-WF OOS window
  (478 days, 2024-07→2026-06), day-clustered bootstrap CI.
- **WIN** only if some side: Δnet CI > 0 **and** v2+veto net ≥ 2bps @ t ≥ 2 (TRIGGER analog).
- **Stop rule:** if flat/negative → dead-end; do NOT sweep keep_rate/λ/threshold to fish (threshold-
  deflation trap, cf. v8 / 20bps fragility probe). One hypothesis → at most one Gauntlet run.

## 💻 Active code (new)
- [build_daily_panel.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/build_daily_panel.py)
- [daily_model.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/daily_model.py)
- [make_v2_pickmask.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/make_v2_pickmask.py)
- [train_daily.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/train_daily.py)
- [daily_veto_walkforward.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/daily_veto_walkforward.py)

## 📝 Compacted Session Log
- **Data facts (verified):** `ranking_data_daily_macro_v2.csv` = 407,499 rows = 2,470 days × 172
  tickers, 2016-06-10→2026-06-04. 83 features already per-day cross-sectionally z-scored; **28 macro
  columns are ZEROED by that z-scoring** (constant within a day → dropped from the sequence; 55
  stock-level kept). Real macro FiLM sourced from `ranking_data_daily_macro_v3.csv`. `Label_3D` = RAW
  3-day fwd return (has a +8592% split outlier → winsorized for training only).
- **v2 OOS reproduction:** reran v2's exact 4-fold WF, dumped genuine-OOS Top-5 picks (478 OOS days).
  v2 OOS gross 3-day edge: long **+25.7bps** (net@10 +15.7), short **+11.8bps** (net@10 +1.8). Consistent
  with v2's certified ~56bps@K3 combined (the "0.565 bps/day" in memory is mis-stated units = 0.565% =
  56.5bps per 3-day trade).
- **Model:** `DailyCSTransformer` — daily TemporalEncoder + cross-sectional attention + sector emb +
  macro FiLM, ~86k params (small by design). Reuses model.py's encoder block + train.py's gate loss.
- **Training:** purged WF on v2's fold boundaries; train-only macro normalization + train-only label
  winsorization; gate loss, cost 10bps, keep_rate 0.70.
- **Gotchas fixed:** Windows OpenMP segfault from vectorized pandas-after-torch (→ numpy month labels);
  datetime64[us] vs [ns] mapping bug; dow embedding sized 7 (NSE Sat sessions).
- **Status:** CONCLUDED — dead-end.

## ❌ RESULT (exploratory, no Gauntlet — audit `artifacts/daily_veto_audit.log`)
Common OOS window 478 days (2024-07→2026-06), cost 10bps, 3-day returns. Cost-accounting clean
(`median(net-gross)==-10` per side; raw/net WR consistent — no cost-sign bug).
- **Pre-registered WIN bar:** Δnet bootstrap-CI>0 AND v2+veto net ≥2bps @ t≥2. **No cell clears it.**
- **LONG:** veto barely fires (cov 98-99% — the gate rarely disagrees with v2's already-TRIGGER longs).
  Best K=3 uplift +2.2bps but t=1.32, CI[-0.9,+5.6] straddles 0. Not significant.
- **SHORT:** uplifts consistently positive (+4.9/+6.2bps @K3/K5) and correctly signed (vetoed shorts are
  reliably bad, -11 to -23bps) but t=0.85/1.51 — sub-significant. The only CI>0 cell, SHORT K=1 (t=2.64),
  **FAILS the negative control** (shuffled-return uplift -7.9, not ~0) = n=478 single-pick small-sample
  mirage (exactly the stop-rule trap; harness correctly caught it).
- **Decision:** dead-end for the hypothesis. Per stop rule: NOT sweeping keep_rate/threshold to fish; NOT
  spending a Gauntlet run (null result would only deflate future t-thresholds). v2 stays the certified base.
- **Reusable assets:** `data/daily_transformer_panel/` (panel + v2 OOS picks + gate scores); the 5 scripts
  re-run the whole test in minutes. Same "right-direction, sub-significant, information-limited" pattern as
  [[project_sided_transformer_result]] / [[project_dualres_transformer_result]] — the ceiling is the
  information set, not the loss/architecture, at daily resolution too.

## 🔗 Core Memory Links
- [[02 — Models/Gauntlet Reports/Daily Macro v2 Report]]
- Related dead-ends/results: project_dualres_transformer_result, project_sided_transformer_result,
  project_validation_gauntlet (auto-memory).
