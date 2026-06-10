# Welcome to the Vanguard Brain (Obsidian Memory Layer)

Welcome to the **Vanguard V2.0 Memory Layer**. This vault serves as our shared core memory, capturing the architectural blueprint, strategic intelligence, model configurations, and execution loops of the **Vanguard High-Precision Intraday Trading Engine**.

This index is designed to link every major component of the system together, enabling us to navigate, query, and develop without losing context or missing critical dependencies.

---

## Navigation Map

**[[agent|AI OPERATING PROTOCOL & MEMORY GUIDE]]**: **MUST-READ FOR ALL AI AGENTS** (Claude, Gemini, etc.). Outlines bootstrapping, context segregation, and memory update rules.

### 1. Core Architecture

* **[[01. Core Architecture/Global System Architecture|Global System Architecture]]**: The high-level blueprint of the Vanguard Engine, linking technical ranks (XGBoost) and sentiment RAG filters (PrimoGPT).
* **[[01. Core Architecture/Vanguard System Features|Vanguard System Features]]**: Core features, release specifications, and settings for Vanguard v2.5 Hardened.
* **[[01. Core Architecture/Validation Gauntlet Architecture|Validation Gauntlet Architecture]]**: 📐 The canonical self-testing evaluation harness spec — the single source of truth for all model metrics (purged WF, leakage probes, cost invariants, 3-tier verdicts, phased build plan P0–P7).
* **[[01. Core Architecture/Validation Gauntlet Remediation Plan|Validation Gauntlet Remediation Plan]]**: 🔧 Post-audit fix spec (R1–R8) closing the 6 critical + 9 moderate gaps found in the P0–P7 build: prefix-invariance wiring, label-integrity rebuild, verdict-engine fixes, true pre-registration, T8 + preds.npz, hygiene, and the final re-baseline campaign.

### 2. Model Suite

* **[[02. Model Suite/Multi-Timeframe Models|Multi-Timeframe Models]]**: Breakdown of the 4 core models (Daily, 1H, 30M, 15M) and their respective features.
* **[[02. Model Suite/Training Data & Regime Requirements|Training Data Requirements]]**: The "row count fallacy", regime overfitting, and the 3-year standard mandate for all models.
* **[[02. Model Suite/Model Registry & File Structures|Model Registry & File Structures]]**: Code locations, metadata, scaler requirements, and registry paths.
* **[[02. Model Suite/Model Performance & Statistics|Model Performance & Statistics]]**: Hyperparameter logs, walk-forward validation folds, and return metrics under market friction.
* **[[02. Model Suite/Empirical Regime Simulation Results|Empirical Regime Simulation Results]]**: The dual-pass backtest matrix deciding the Trend vs Reversal routing of all strategies.
* **[[02. Model Suite/Feature Engineering & Normalization|Feature Engineering & Normalization]]**: Details on absolute/relative feature calculations, normalization strategies, and cross-sectional Z-scoring.
* **[[02. Model Suite/Model Inference Data Structure|Model Inference Data Structure]]**: Master specification for the feature vector used by `xgb_long_model` and `xgb_short_model` at inference time.
* **[[02. Model Suite/V8 Microstructure Feature Comparison|V8 Microstructure Feature Comparison]]**: Performance analysis of microstructure features (IBS, Buy Pressure) and the removal of the leaky `Gap_Pct`.
* **[[02. Model Suite/Advanced Tree Models Roadmap|Advanced Tree Models Roadmap]]**: The 5 advanced techniques (CatBoost, LightGBM, Custom Objectives, Monotonic Constraints, DART) to explore for combating structural fee erosion.

### 3. Trading Strategies & Friction

* **[[03. Trading Strategies/Strategy Catalog|Strategy Catalog]]**: The comprehensive catalog of backtested trading strategies, highlighting Strategy 8 (ORB) and Strategy 10 (Quad-TF).
* **[[03. Trading Strategies/Market Friction & Slippage|Market Friction & Slippage]]**: A critical analysis of the 0.06% round-trip drag and its impact on high-frequency scalpers vs. low-frequency trend riders.
* **[[03. Trading Strategies/Upstox Fees & Statutory Taxes|Upstox Fees & Statutory Taxes]]**: Structured breakdown of Indian statutory fees, taxes (STT, GST, Stamp Duty), and brokerage friction for NSE Equity Intraday.
* **[[03. Trading Strategies/Strategy March 2026 Revision|Strategy March 2026 Revision]]**: Analysis of the directionally correct Spearman correlation shift, Golden Hour trading, and volatility pause filters.

### 4. Data & Code Map

* **[[04. Data & Code Map/Database Architecture|Database Architecture]]**: SQLite schema for trade logging, performance tables, and data structure migrations.
* **[[04. Data & Code Map/Shadow Tracker & Execution Loop|Shadow Tracker & Execution Loop]]**: Runtime execution rules: dynamic ATR-based take profit, 1h hard-close, and 15:15 IST EOD flush.
* **[[04. Data & Code Map/AI Veto & Gemini Audit|AI Veto & Gemini Audit]]**: Details of live sentiment verification, support/resistance calculation, and trade auditing.
* **[[04. Data & Code Map/Codebase File Directory|Codebase File Directory]]**: Master catalog mapping python files in `scripts/`, logs, and configurations.
* **[[04. Data & Code Map/Upstox Brokerage API Plan|Upstox Brokerage API Plan]]**: Dynamic integration roadmap for using Upstox's live charges API to audit transaction friction.

### 5. Archives

* **`05. Archives/`**: A folder containing individual, modular markdown files for all retired and obsolete systems. We use individual files here instead of a monolithic ledger for better searchability in Obsidian, while still keeping them physically separated from active codebase documentation.

### 6. Context & Logs

* **[[06. Context & Logs/Current Context|Current Context]]**: Tracks the ongoing development topics, current focus, and immediate next steps.
* **[[06. Context & Logs/Codebase Cleanup Strategy|Codebase Cleanup Strategy]]**: Audit and phased strategy to isolate and prune stale Python and obsolete Markdown files.
* **[[06. Context & Logs/Vanguard Engine Refactor Roadmap|Vanguard Engine Refactor Roadmap]]**: Why/how plan for splitting `vanguard_signal_engine.py` into safer production modules, with task lists for artifact hygiene, tests, validation discipline, docs, and reproducible environments.
* **[[06. Context & Logs/Conversations/Conv-2026-06-01-Docs-Reorganization|Conversations Log]]**: Compacted, segregated logs of active work sessions.
* **[[06. Context & Logs/Daily Logs/2026-05-31|Daily Logs (Latest)]]**: Daily updates and work logs.

### 7. Research & Backtests

* **[[07. Research & Backtests/V18-Hybrid-Veto-Scalability|V18 Hybrid Veto Scalability]]**: Portfolio simulation, leverage stress testing, and the efficient frontier (A1 vs A3 vs A5) of the Hybrid Random Forest system.

### 8. Model Analysis

* **[[08. Model Analysis/1-Hour Vanguard Model/Advanced Alpha Visualizations|1-Hour Alpha Visualizations]]**: SHAP analysis, prediction bucket evaluation, and cumulative returns for the 1-Hour Core Model (v8_upstox_3y).
* **[[08. Model Analysis/30-Minute Vanguard Model/Complete Edge Catalog|30-Min Edge Catalog]]**: Exhaustive dual-model parameter exploration, dead-end tests, and tiered execution strategy for the 30-Minute models.
* **[[08. Model Analysis/30-Minute Vanguard Model/OOS Calibration & Thresholds|30-Min OOS Calibration]]**: Signal inversion discovery (both failed), threshold sweeps, and fee-adjusted win rate analysis.
* **[[08. Model Analysis/30-Minute Vanguard Model/Time of Day Conviction|30-Min Time of Day Conviction]]**: Heatmaps revealing 15:15 IST as the primary alpha window for Longs, 14:15 IST for Shorts.
* **[[08. Model Analysis/30-Minute Vanguard Model/Dual Confirmation Architecture|30-Min Dual-Lock Architecture]]**: Why Dual-Lock adds minimal value for 30-min models (unlike 1-hour).
* **[[08. Model Analysis/30-Minute Vanguard Model/Weekly Consistency & Regimes|30-Min Weekly Consistency]]**: Week-by-week stability analysis and inverse regime dependency discovery.

### 10. MCP Integrations

* **[[10. MCP Integrations/MCP Registry|MCP Registry]]**: Tracks the research, configuration, and status of external Model Context Protocol (MCP) tools connected to the Vanguard workspace.

---

## Core Performance Baseline (May 2026 Simulation)

| Strategy | Trades | Win Rate (WR) | Long WR | Short WR | Net Return | Profit Factor | Max Drawdown | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Strategy 8: ORB** | 14 | **64.3%** | 50.0% | 75.0% | **+2.31%** | **3.92** | **-0.51%** | Best Strategy |
| **Strategy 10: Quad-TF** | 24 | **54.2%** | **85.7%** | 41.2% | **+1.86%** | **1.60** | **-1.32%** | Most Robust |
| **Strategy 1: Daily Macro** | 108 | 50.9% | 50.0% | 51.7% | **+2.20%** | 1.17 | -4.04% | Profitable |
| **Strategy 2: Short-Side** | 17 | 58.8% | N/A | 58.8% | **+0.48%** | 1.17 | -1.01% | Profitable |

> [!IMPORTANT]
> **Performance figures are calculated under a strict 0.06% round-trip slippage and transaction cost model.** This cost is the primary differentiator between robust, structural strategies and high-frequency failures.

---

## Live Connection Guides

* **To run the live signal engine:**
    ```powershell
    .\run_vanguard_system.bat
    ```
* **To run the 10x strategies backtester:**
    ```bash
    python scripts/backtests/strategy_10x_backtest.py
    ```

This memory layer is automatically maintained and expanded as the Vanguard project evolves. Use it to check structural designs, research hyperparameters, or verify execution rules before making changes.
