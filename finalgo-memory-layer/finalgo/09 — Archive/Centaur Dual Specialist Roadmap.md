---
title: "Centaur Dual-Specialist & ADX Meta-Routing Roadmap (May 27, 2026)"
type: archive
status: archived
updated: 2026-06-12
tags: []
---
# 🏗️ Centaur Dual-Specialist & ADX Meta-Routing Roadmap (May 27, 2026)

### The Proposal
- **Concept**: Split Vanguard's prediction engine into two specialized XGBoost binary classification sub-specialists:
  - **Model A: Reversal Specialist (`MR_Model_v2`)**: Trained strictly to classify if a price overextension (measured as a stretch above EMA/ATR boundaries) would revert cleanly.
  - **Model B: Breakout Specialist (`BO_Model_v1`)**: Trained to detect volatility compression (Bollinger Squeeze) followed by volume expansion.
- **Meta Routing Layer**: A heuristic routing system that evaluated market regime metrics (ADX/VIX) before triggering signals:
  - If `ADX < 20` (Range-Bound): Route capital to the **Reversal Specialist**; ignore breakouts.
  - If `ADX > 25` (Strong Trend): Route capital to the **Breakout Specialist**; ignore reversals.

### Why It Was Retired
- **May 2026 Empirical Sim**: A full 50-strategy backtesting sweep across May 2026 NSE data proved that heuristic routing underperformed compared to our **Gated/Ungated Strategy Regime Framework** (Strategy 8 ORB and Strategy 10 Quad-TF), which natively captures structural breakouts with superior risk-adjusted drawdowns without the overhead of dual active XGBoost classifiers.
