---
title: "Conversation Context: 15m Model Audit & Logic Fix Discovery"
type: log
status: active
updated: 2026-06-12
tags: []
---
# 💬 Conversation Context: 15m Model Audit & Logic Fix Discovery

## 📌 Metadata
- **Conversation ID**: 35c04023-538b-4c5e-8303-b820d4731353
- **Start Date**: 2026-06-11
- **Status**: 🟢 Active
- **Focus Area**: Model Suite & Trading Strategies

## 🎯 Objectives
- [x] Investigate why the S2 model approved a SHORT on RBLBANK at 14:16 when the 14:15 candle was heavily green.
- [x] Uncover the blind spot: The orchestrator checks the strictly completed (previous) candle and ignores the currently forming candle.
- [x] Identify critical bug: `orchestrator.py` was feeding 60-minute data features to the `v3_15min_clean` model, completely corrupting the 15m scores during live execution.
- [ ] Run a comprehensive audit of all trades to extract true 15m conviction scores (with pricing data) and log to the memory layer.
- [ ] Implement the look-back candle fix in `orchestrator.py` to prevent entering trades against massive currently-forming momentum.

## 💻 Active Code Files Modified
- [generate_15m_audit_detailed.py](file:///C:/Users/loq/.gemini/antigravity/brain/35c04023-538b-4c5e-8303-b820d4731353/scratch/generate_15m_audit_detailed.py)
- [15m_Conviction_Audit_Report.md](file:///C:/Users/loq/Desktop/Trading/finalgo/finalgo-memory-layer/finalgo/02.%20Model%20Suite/15m_Conviction_Audit_Report.md)

## 📝 Compacted Session Log
- **Initial Analysis**: The user flagged a discrepancy in an RBLBANK S2 SHORT trade on June 11, where the engine ignored a massive green candle.
- **Step 1**: Confirmed the orchestrator uses `resample('15Min', origin='start_day')` and strictly checks the *last completed* candle, causing a 15-minute blind spot to massive reversals.
- **Step 2**: The user requested historical conviction scores for the 15-minute model. Discovered a critical bug: `orchestrator.py` evaluates the `v3_15min_clean` model using `60minute` data features, completely scrambling the model's inputs.
- **Step 3**: Drafted a script (`generate_15m_audit_detailed.py`) to properly resample 1-minute data into 15-minute candles, correctly compute the 15-minute features, and evaluate the XGBoost models over the full `T-4` to `T+4` tracking window for all 938 historical trades in the ledger.
- **Step 4**: Writing the detailed output (including entry/exit prices and candle returns) directly to the memory layer.

## 🔗 Core Memory Links & Backlinks
- Linked Core Specs: [[02 — Models/_Shared/Multi-Timeframe Models]]
- Report Artifact: [[02 — Models/15m/15m Conviction Audit Report]]
