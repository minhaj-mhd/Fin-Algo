---
title: "Conv 2026-06-28 — v21 Cleanest Rolling-1h Ranker"
type: log
status: concluded
updated: 2026-06-28
model: v21_rolling_1h
tags: [log, v21, rolling-1h, data-cleaning, ranker]
---
# 💬 Conversation Context: v21 — Cleanest Rolling-1h Ranker

## 📌 Metadata
- **Conversation ID**: 7ee6cb35-caa9-47d5-b75f-e843b93343d4
- **Start Date**: 2026-06-28
- **Status**: 🔴 Concluded (lean v21 saved as research artifact; promotion/Gauntlet deferred — v21 ≈ wash vs v20)
- **Focus Area**: Model Suite — v20→v21 rolling-1h ranker, data cleanliness

## 🏁 Final outcome
**Lean v21 delivered** = v20 recipe + liquidity universe + bar hygiene + mask-not-fill + **wall-clock
lookback fix (the only ablation-positive lever)** + session-boundary candles/gap representation. Robust
MAD scoring and the sector-graph feature were **dropped** (ablation: neutral-to-negative). Corp-action
adjustment **unnecessary** (Upstox cache already split-adjusted — verified). No leak (shuffle ρ≈0).
**WF Long ρ 0.0312 / Short ρ 0.0291, still sub-cost** → a cleaner FILTER_GRADE peer of v20, not a
money-maker. Artifact: `models/research/v21_rolling_1h/`. **v20 stays LIVE; no Gauntlet spent.**

## 🎯 Objectives
- [x] Phase 0 — Scaffold: golden snapshot, versioned `clean_v21` feature path (v20 byte-identical VERIFIED), `1h_roll_v21` cfg, builder
- [x] Phase 1 — liquidity universe (top-110 ADV) + frozen/zero-vol bar hygiene
- [x] Phase 1.5 — Session-boundary candles (`close_stub`/`overnight`) + causal gap features + sidecar
- [x] Phase 2 — corporate-action adjustment **DROPPED** (cache already split-adjusted; audit clean)
- [x] Phase 3 — mask-not-fill (NaN instead of 0.0/0.5)
- [x] Phase 4 — tweaks evaluated: wall-clock lookback **KEPT** (only lever); robust scoring + graph **DROPPED**
- [x] Phase 5 — WF eval vs v20 + neg-control (no leak) + leave-one-out ablation
- [ ] Phase 6 (gated) — **deferred**: not pursued (v21 not clearly ≥ v20; a Gauntlet run isn't justified)

## 💻 Active Code Files (new / edited)
- [build_v21_rolling_1h_panel.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/build_v21_rolling_1h_panel.py) (new)
- [build_adjustment_factors.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/build_adjustment_factors.py) (new)
- [feature_utils.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/feature_utils.py) (add `clean_v21` branch — default bit-identical)
- [train_ranking_clean.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/training/train_ranking_clean.py) (add `1h_roll_v21` cfg)
- Reused: [build_rolling_1h_panel.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/build_rolling_1h_panel.py) (v20 reference), `scripts/structural/{build_relation_graph,gate1_walkforward}.py`, [build_v20_gauntlet_csv.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/build_v20_gauntlet_csv.py)

## 📝 Compacted Session Log
- **Initial analysis**: traced v20 data path (raw 15-min cache → rolling-1h windows → `compute_features(legacy=False)` → per-query CS z-score → next-1h label). Confirmed v20_rolling_1h is the **live** active model (`config.ROLLING_1H_CANDLES=True`, registry `active_model`, consumed at orchestrator.py:859/753/1847).
- **Contamination found**: silent `fillna(0.0/0.5)` → z-scored fake signal (feature_utils 365-400); unadjusted prices (no split/bonus handling in collector); no liquidity filter (`MIN_PER_Q≥5` only); mean/std z-score outlier-sensitive; overnight transition silently dropped (no session-boundary awareness).
- **Honest framing**: 1h is **information-limited** per prior research ([[04 — Research/V20 Rolling-1h Overlapping-Window Model]] and CST/PA-SMC dead-ends) → v21 will most likely stay FILTER_GRADE. Win = artifact-free + tradeable-honest data + possible modest graph rank-IC lift. Reported truthfully.
- **Plan approved** (user added the session-boundary candle requirement: `close_stub` + `overnight` as separate marked non-tradable candles, data-layer = feature now + masked-token ready).
- **Phase 0a done**: golden v20 feature snapshot captured for regression gate — RELIANCE 21647 rows / 90 cols / hash `f18250660d6ecd0e` (also TCS `f73946e0a33fc92d`, HDFCBANK `f59e8cf875f44f9b`). Snapshot parquet in session scratchpad.
- **Phase 0c/3/4-T2 done**: added `clean_v21` branch to `compute_features` (mask-not-fill + ×4 wall-clock lookback scale `S()`). **Regression PASS** — all 3 golden hashes identical with `clean_v21=False`, so v20 live path byte-identical. `clean_v21=True` confirmed active (RSI 14→56, 52w warmup 49→199 ≈4×).
- **Phase 2 DROPPED (halt-and-report finding)**: Upstox 15-min cache is **already split/bonus-adjusted** — pre/post close ratio ≈1.0 at all 3 known ex-dates (RELIANCE 2:1 0.995, TATASTEEL 10:1 0.957, HDFCBANK 2:1 1.013), no discontinuity. Back-adjusting would **double-adjust/corrupt**. Repurposed as an audit: 172-ticker overnight-gap scan → 15 names >25% are real demergers (NMDC, VEDL)/Adani circuit-halts (Mar-2023)/crashes, NOT split artifacts; contiguity guard already keeps them out of labels. **yfinance is fully out of v21** (it only ever appears in daily-index macro collection, not the 1h ranker).

- **Phase 0d/1/1.5/4 done — v21 builder** [build_v21_rolling_1h_panel.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/build_v21_rolling_1h_panel.py): liquidity universe (top-110 by median daily ₹-volume), bar hygiene (drop High==Low + zero-vol), session-boundary candles (`close_stub`+`overnight`→sidecar), causal `Overnight_Gap_Prior`/`Time_To_Close`/`Is_Last_Tradable_Hr`, robust median/MAD ±5 winsorized CS scoring, dynamic sector+group neighbor features (`nb_grp_*`/`nb_sec_*`, 5 momentum feats × 2, static topology excluded, reuses `data/research/graph/edges.csv`). Train cfg `1h_roll_v21` added to [train_ranking_clean.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/training/train_ranking_clean.py).
- **Phase 5a done — full panel**: `data/research/v21_rolling_1h/panel.parquet` = **1,907,529 rows, 17,441 queries, 88 z-feats, 109.4 tickers/query, 2022-01→2026-06**. Sidecar `boundary_candles.parquet` = 215,092 (107,931 close_stub + 107,161 overnight). Universe `universe.json` (HDFCBANK 1971.7cr → CONCOR 83.3cr cutoff).
- **Phase 5b/c DONE** — controlled purged WF [eval_v21_vs_v20.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/eval_v21_vs_v20.py) (same universe, 8 folds, rank:pairwise; `eval_summary.json`):

  | variant | L_rho | S_rho | L_top3 net | S_top3 net |
  |---|---|---|---|---|
  | v20@110 | 0.0232 | **0.0293** | −7.90bps | **−7.74bps** |
  | v21 (graph on) | 0.0284 | 0.0262 | −6.34bps | −9.15bps |
  | v21_nograph | **0.0293** | 0.0265 | −6.72bps | −9.18bps |
  | v21_shuffle (NEG) | 0.0011 | 0.0006 | −9.91 | −9.64 |

  - **NEG-control PASS** (shuffle rho ≈0) → v21 panel has **no leak**.
  - **Cleaning effect** (v21−v20@110): LONG rho **+0.0052 (+22%)**, SHORT rho **−0.0031 (−11%)**, LONG top3 net +1.5bps, SHORT −1.4bps → a **lateral rebalance** (long↑ short↓), not a clear win.
  - **Graph tweak (T3) is DEAD at 1h**: v21 − v21_nograph ≈ −0.0009/−0.0003 → adds **nothing** (nograph marginally better). Daily-panel graph lift did NOT transfer to intraday. **Drop the nb_ features.**
  - **Still sub-cost both sides** (all top-3 net negative @10bps) → v21 = a cleaner, leak-verified **FILTER_GRADE peer** of v20, NOT a money-maker. Reconfirms the 1h info-ceiling: cleanliness ≠ edge.
  - **No Gauntlet spent** — v21 is not clearly ≥ v20, so a cert run is not justified yet (awaiting user decision).
- **Phase 5d DONE — leave-one-out ablation** [ablate_v21_cleaning.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/ablate_v21_cleaning.py) (`ablation_summary.json`), SAME production-faithful feature set for all variants (CORRECTS the headline eval, where v20@110 was shortchanged its raw cols):

  | variant | L_rho | S_rho |
  |---|---|---|
  | BASE v20@110 (corrected) | 0.0280 | **0.0316** |
  | FULL clean | **0.0312** | 0.0288 |

  Leave-one-out element contributions (FULL − element_off):
  | element | ΔL_rho | ΔS_rho |
  |---|---|---|
  | **clean_feats (mask + wall-clock lookback)** | **+0.0048** | **+0.0016** |
  | hygiene | +0.0003 | +0.0012 |
  | gap features | +0.0003 | −0.0006 |
  | robust MAD scoring | **−0.0009** | −0.0001 |

  - **The headline "+22% long" was mostly a feature-set artifact.** Corrected baseline raised v20@110 (L 0.0232→0.0280, S 0.0293→0.0316); true FULL−BASE = **L +0.0031 / S −0.0028** (near-wash).
  - **The ONLY robustly positive element is `clean_feats` = the wall-clock lookback fix (T2) + mask** — +0.0048 long (~+17%), +0.0016 short. Fixing v20's 4×-too-short bar-count lookback is the real lever.
  - **DROP robust scoring (−0.0009 long) and the graph feature (dead).** hygiene ~neutral-positive (keep; good practice). gap features ~neutral for ranking (keep for the boundary-awareness intent, not performance).
  - Short-side configs all within ~0.004 (overlap-inflated noise floor). Still **sub-cost both sides**. Net: v21 ≈ a clean wash vs v20; the defensible win is narrow (the lookback fix, long side).

## 🔗 Core Memory Links & Backlinks
- Host/predecessor: [[04 — Research/V20 Rolling-1h Overlapping-Window Model|v20_rolling_1h]]
- Info-ceiling line: [[04 — Research/_MOC]] (CST / PA-SMC / dual-TF dead-ends)
- Graph tweak basis: Gate-1 graph features (dynamic neighbor-agg WF-robust; static memorizes)
- Overnight signal basis: intraday overnight-reversal edge (RAY OF HOPE)
