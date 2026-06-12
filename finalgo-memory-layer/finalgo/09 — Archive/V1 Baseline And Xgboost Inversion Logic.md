---
title: "Vanguard V1 Baseline & XGBoost Inversion Logic (Dec 2024)"
type: archive
status: archived
updated: 2026-06-12
tags: []
---
# 📈 Vanguard V1 Baseline & XGBoost Inversion Logic (Dec 2024)

### The System Concept
- **Objective**: Develop an hourly relative-ranking trading model using pairwise XGBoost to pick the top NSE equities.
- **Problem**: The trained model learned the inverse of the expected return. When running predictions, the top-ranked stock was highly likely to *fall*, while the worst-ranked stock was highly likely to *rise*.
- **Metrics**: 
  - Non-Inverted Top-1 Win Rate: **~10.0%** (Catastrophic)
  - Correlation (Spearman's Rho): **-0.63** (Strong negative correlation)
  - Return Drag: **-0.38%** hourly return on non-inverted predictions.

### The "Inversion Workaround"
- **The Rationale**: Rather than retraining a noisy dataset, we exploited the strong negative correlation. We added a single minus sign to the model outputs: `predictions = -model.predict(X)`.
- **Outcome**: By inverting the model's rankings, the system selected the lowest-ranked volume/momentum stocks. 
  - Inverted Top-1 Win Rate: **~58.0%**
  - Compounded Hourly Return: **+0.29%**
  - **The Pattern Learned**: The model had successfully discovered **Volume-Based Mean Reversion** (High retail FOMO volume is distributed by institutions and drops over the next hour; quiet, low-volume accumulation leads to breakouts).

### Why It Was Retired
- **Vanguard V2 Evolution**: Retaining inverted logic is dangerous and unintuitive. In **Vanguard V2.3**, the codebase migrated to a **directionally correct** (+0.0426 Spearman) dual-model ensemble utilizing separate, dedicated `xgb_long_model` and `xgb_short_model`, eliminating raw prediction inversion.
