# 💬 Conversation Context: Shadow Tracking Peak Negative PnL

## 📌 Metadata
- **Conversation ID**: 82b1d5ff-b39d-4f60-8227-ad8734536b66
- **Start Date**: 2026-06-17
- **Status**: 🔴 Concluded
- **Focus Area**: Execution & Runtime

## 🎯 Objectives
- [x] Research current shadow tracking logic and database schema for tracking peak PnL
- [x] Design the schema additions and tracking logic for peak negative PnL alongside peak PnL
- [x] Implement database upgrades and model changes
- [x] Update execution tracking code in orchestrator, trade_state, and risk_manager
- [x] Verify functionality

## 💻 Active Code Files Modified
- [database_manager.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/database_manager.py)
- [orchestrator.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/orchestrator.py)
- [risk_manager.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/risk_manager.py)
- [vanguard_v2.html](file:///c:/Users/loq/Desktop/Trading/finalgo/templates/vanguard_v2.html)
- [strategy_detail.html](file:///c:/Users/loq/Desktop/Trading/finalgo/templates/strategy_detail.html)
- [ticker_detail.html](file:///c:/Users/loq/Desktop/Trading/finalgo/templates/ticker_detail.html)
- [test_trade_state.py](file:///c:/Users/loq/Desktop/Trading/finalgo/tests/test_trade_state.py)

## 📝 Compacted Session Log
- **Initial Analysis**: The user wants to add peak negative PnL (drawdown tracking) alongside peak PnL in shadow tracking.
- **Database Schema**: Added `peak_adverse_pct` column to `trades` SQLite schema and automated migrations.
- **Execution Tracking**: Updated `orchestrator.py` to track the lowest adverse PnL (`min(0, pnl)`) reached during shadow trade lifecycles.
- **Adapter Mapping**: Exposed `peak_adverse_pct` to the active open trades mapped to the UI.
- **Frontend Display**: Added `Peak Neg` table columns and Javascript/Jinja cell formatting in all trades views (`vanguard_v2.html`, `strategy_detail.html`, `ticker_detail.html`).
- **Validation**: Added unit test `test_peak_adverse_pct_tracking` and verified that the entire test suite passes successfully.

## 🔗 Core Memory Links & Backlinks
- Linked Core Specs: [[01 — Architecture/Execution & Runtime/Shadow Tracker & Execution Loop]]
