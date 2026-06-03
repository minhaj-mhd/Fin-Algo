# 💬 Conversation Context: 10:30 Momentum Strategy

## 📌 Metadata
- **Conversation ID**: 30c6b574-57dc-4f6f-ba94-63225606d5c3
- **Start Date**: 2026-06-03
- **Status**: 🔴 Concluded
- **Focus Area**: Model Suite / Strategy Research

## 🎯 Objectives
- [x] Review archived transformer/iTransformer research and `research_1030_strategy` folder
- [x] Reality-check the architecture (dismissed Mamba/Autoformer, confirmed XGBoost-only)
- [x] Design two-layer system (Layer A market filter + Layer B stock selector)
- [x] Create V1 implementation plan
- [x] User review → identified critical flaws (overfitting, tautological features, data size)
- [x] Create V2 implementation plan addressing all criticisms
- [x] Write config.py, data_collection.py, feature_engineering.py, train.py
- [x] Write backtest.py
- [x] Run Phase 1 (data collection)
- [x] Run Phase 2 (training)
- [x] Run Phase 3 (backtest)

## 💻 Active Code Files Modified
- [config.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/strategy_1030/config.py) — V2: 10 Layer A features, 20 Layer B features, 30-min cache as primary
- [data_collection.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/strategy_1030/data_collection.py) — V2: uses 30-min cache (~1100 days)
- [feature_engineering.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/strategy_1030/feature_engineering.py) — V2: removed slow-moving indicators
- [train.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/strategy_1030/train.py) — V2: 6-2-2 walk-forward (~15 folds), reg_alpha=1, reg_lambda=5, switched Layer B to XGBRegressor
- [backtest.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/strategy_1030/backtest.py) — V2: OOS fold-level simulation, Z-scoring, NaN cleaning, threshold sweep

## 📝 Compacted Session Log
- **Initial Analysis**: Bootstrapped and reviewed archived iTransformer scripts and `research_1030_strategy` folder.
- **Step 1**: Reviewed V1/V2 transformer architectures. Confirmed retirement was correct.
- **Step 2**: Reviewed `architecture_research_report.md` — dismissed Mamba/Autoformer.
- **Step 3**: Validated data sources: 15min cache (250d), 30min cache (~1100d), daily cache (~5y), yfinance (7 indices confirmed).
- **Step 4**: Created V1 implementation plan. User approved direction (both layers + XGBoost).
- **Step 5**: Wrote V1 code (config, data_collection, feature_engineering, train, backtest).
- **Step 6 (Critical Review)**: User identified 4 major flaws (overfitting, tautological features, data size, slow indicators).
- **Step 7**: Rewrote all modules for V2. Updated implementation plan in memory layer.
- **Step 8**: Completed data collection, model training (modified Layer B from XGBRanker to XGBRegressor to handle continuous float targets), and backtested the dynamic system.
- **Backtest Results**: 
  - Layer A Only: Return = -50.35%, Sharpe = -2.10
  - Layer B Only: Return = +2.80%, Sharpe = +0.13
  - Two-Layer (0.060% threshold): Return = **+17.32%**, Sharpe = **0.77**, Max Drawdown = **-4.6%**, Trades = 498.

## 🔗 Core Memory Links & Backlinks
- Implementation Plan: [[research_1030_strategy/implementation_plan]]
- Linked Core Specs: [[02. Model Suite/Model Registry & File Structures]]
- Original Research: [[research_1030_strategy/architecture_research_report]]
