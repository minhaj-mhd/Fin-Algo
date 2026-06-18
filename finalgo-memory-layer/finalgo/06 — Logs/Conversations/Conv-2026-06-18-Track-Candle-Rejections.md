# 💬 Conversation Context: Track Candle Rejections

## 📌 Metadata
- **Conversation ID**: 8e23b21e-f8e3-427a-92ce-28a0ba4015b3
- **Start Date**: 2026-06-18
- **Status**: 🟢 Active
- **Focus Area**: Trading Strategies

## 🎯 Objectives
- [ ] Phase 1: Append candle VETOED/CANCELLED trades to active shadow tracking with qty=0, margin=0.
- [ ] Phase 1: Introduce structured fields to trade dict (reject_stage, reject_reason, decision-time features, px bounds).
- [ ] Phase 2: Emit JSONL row per terminal candle outcome to data/research/candle_rejections.jsonl.
- [ ] Phase 2: Assert median(net-gross) == -cost per side.
- [ ] Phase 3: Create scripts/research/analyze_candle_rejections.py for comprehensive analysis and output report.
- [ ] Phase 4: Extend veto_stats dashboard output with candle tallies and running guard-value.
- [ ] Phase 5: Add unit test in tests/ verify transition to VETOED_EXPIRED/CANCELLED_EXPIRED.

## 💻 Active Code Files Modified
- [orchestrator.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/orchestrator.py)
- [persistence.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/persistence.py)
- [analyze_candle_rejections.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/analyze_candle_rejections.py)
- [test_candle_tracking.py](file:///c:/Users/loq/Desktop/Trading/finalgo/tests/test_candle_tracking.py)

## 📝 Compacted Session Log
- **Initial Analysis**: The current veto-counterfactual mechanism only tracks AI-vetoed trades and ignores candle-stage rejections/vetoes/cancellations. This prevents performance analysis of the fade-entry and candle veto guards.

## 🔗 Core Memory Links & Backlinks
- Linked Core Specs: [[01 — Architecture/Data & Code/Database Architecture]]
