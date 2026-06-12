---
title: "50 Strategies Regime Backtest Plan"
type: spec
status: active
updated: 2026-06-12
tags: []
---
	# 50 Strategies Regime-Aware Backtest Plan (Gated vs Ungated)

## 1. Goal Description
The objective is to evaluate all 50 trading strategies in the catalog using a Regime-Aware Comparative Backtest. For each strategy, we will run the simulation twice:
1. **GATED Pass**: The Daily Macro Gatekeeper is strictly enforced. (Trades opposing the daily trend are blocked).
2. **UNGATED Pass**: The Daily Gatekeeper is entirely bypassed. (Pure tactical/intraday execution).

This dual-pass simulation will mathematically prove whether each of the 50 strategies acts better as a `TREND` strategy (needs gating) or a `REVERSAL/SCALP` strategy (should remain ungated).

## 2. Current Architecture & Gaps
* The existing `scripts/strategy_regime_backtest.py` is hardcoded for Strategies 1 through 11.
* Strategies 12-35 are implemented in `strategy_35x_backtest.py`.
* Strategies 36-50 logic is found in `build_s36_s50.py`.
* We currently lack a unified engine that can iterate over all 50 strategies while toggling the gatekeeper.

## 3. Implementation Steps

### Phase 1: Data Consolidation & Model Loading
* Identify and load all required dataframes for the 50 strategies (e.g., Daily, 1H, 30M, 15M, and any 5M/Microstructure data introduced in S12-S50).
* Load the corresponding XGBoost/Transformer models for all timeframes.

### Phase 2: Engine Unification (`strategy_50x_regime_backtest.py`)
* Create a new master backtest script: `scripts/strategy_50x_regime_backtest.py`.
* Port the entry/exit logic for all 50 strategies into this script's `simulate_strategy()` function.
* For each strategy, wrap the Daily Gatekeeper logic in an `if apply_gatekeeper:` block, ensuring it can be dynamically toggled.

### Phase 3: The Comparative Loop
* Construct a loop iterating from Strategy ID 1 to 50.
* Inside the loop, execute:
  * `simulate_strategy(s_id, name, apply_gatekeeper=True)`
  * `simulate_strategy(s_id, name, apply_gatekeeper=False)`
* Record the Win Rate, Net Return, and Max Drawdown for both passes.

### Phase 4: Reporting and Verdicts
* Generate a unified JSON report: `data/strategy_50_regime_results.json`.
* Output a Markdown matrix summarizing the results (Gated WR, Ungated WR, Gated Net Return, Ungated Net Return).
* Assign an empirical "Verdict" for each strategy (e.g., "TREND: Requires Gatekeeper", "TACTICAL: Run Ungated").

## 4. Open Questions for the User
* Do any of the newer strategies (12-50) have internal rules that *inherently* rely on the Daily Gatekeeper being active, meaning they cannot logically be run "Ungated"?
* Should this 50-strategy backtest run on the standard intraday holdout month (`2026-05`), or do you want a wider testing period?
* Do you want this backtest to run synchronously in the terminal, or should it be converted to a background task due to potential long execution times?
