# 💬 Conversation Context: 30M Model Edge Analysis

## 📌 Metadata
- **Conversation ID**: 5e6cbf1f-25d0-46c0-9be1-c966efbfa4cc
- **Start Date**: 2026-06-04
- **Status**: 🔴 Concluded
- **Focus Area**: Model Suite & Trading Strategies

## 🎯 Objectives
- [x] Read 1-hour model edge methodology from memory layer
- [x] Replicate FULL 1-hour research depth for 30-minute models
- [x] Signal inversion testing (both directions)
- [x] Single-model threshold sweeps with WR/PnL/trade count tables
- [x] Dead-end tests (Agreement, Spread, Ratio)
- [x] Dual-Lock grid sweeps with heatmaps
- [x] Time-of-Day conviction heatmaps
- [x] Weekly consistency & regime analysis
- [x] Write all 5 memory layer documents + 7 visual assets

## 💻 Active Code Files Modified
- [30m_complete_edge_research.ipynb](file:///c:/Users/loq/Desktop/Trading/finalgo/30m_complete_edge_research.ipynb) (Jupyter notebook with all analysis)

## 📝 Compacted Session Log
- **Initial Analysis**: User requested replicating the 1-hour model edge research for 30-minute models. First attempt by Gemini was superficial — missing inversions, threshold tables, dual-lock sweeps, heatmaps, and regime analysis.
- **Step 1**: Read all 6 documents from `08. Model Analysis/1-Hour Vanguard Model/` to understand the exact methodology: OOS split verification, signal inversion, threshold sweeps, dead-end tests, dual-lock grid search, time-of-day heatmaps, and quarterly consistency.
- **Step 2**: Booted Jupyter on port 8896 via `uv run --with jupyter`, installed packages, set up `30m_complete_edge_research.ipynb`.
- **Step 3 (Phase 1)**: Loaded data (541K rows), verified OOS split (Train: 501K rows 2025-05→2026-04, Test: 40K rows 2026-05), confirmed zero leakage.
- **Step 4 (Phase 2)**: Tested signal inversions — both FAILED (max 51% WR for Short→Long, 47.3% WR for Long→Short). Critical structural difference from 1-hour model.
- **Step 5 (Phase 3)**: Single-model sweeps revealed Long Model crosses 50% WR at >0.070 (peaks 62.2% at >0.090), but Short Model NEVER crosses 50% WR globally. Time-filtered analysis discovered Long's edge concentrates at 15:15 IST (60.2% WR at >0.080), Short has narrow edge at 14:15 only (56.3% WR at >0.050).
- **Step 6 (Phase 4)**: Dead-end tests confirmed: Agreement Long/Short (ZERO valid configs), Score Spread (dead), Score Ratio (dead). Dual-Lock Long adds <2% WR (negligible vs 1-hour's +4-8%). Dual-Lock Short barely above fee hurdle.
- **Step 7 (Phase 5)**: Generated 7 publication-quality PNGs: time_of_day_conviction_long/short, oos_calibration, dual_long/short_heatmap, complete_edge_map, weekly_consistency.
- **Step 8 (Phase 6)**: Weekly consistency revealed volatile performance — May 11th was catastrophic for Longs (28% WR) but the Short Model's best week (65.5% WR), proving inverse regime dependency.
- **Step 9 (Phase 7)**: Wrote 5 comprehensive markdown reports, updated Welcome.md with all new links.

## 🔗 Core Memory Links & Backlinks
- Linked Core Specs: [[03. Trading Strategies/Strategy March 2026 Revision]]
- Linked Core Specs: [[02. Model Suite/Multi-Timeframe Models]]
- Generated: [[08. Model Analysis/30-Minute Vanguard Model/Complete Edge Catalog]]
- Generated: [[08. Model Analysis/30-Minute Vanguard Model/OOS Calibration & Thresholds]]
- Generated: [[08. Model Analysis/30-Minute Vanguard Model/Time of Day Conviction]]
- Generated: [[08. Model Analysis/30-Minute Vanguard Model/Dual Confirmation Architecture]]
- Generated: [[08. Model Analysis/30-Minute Vanguard Model/Weekly Consistency & Regimes]]
