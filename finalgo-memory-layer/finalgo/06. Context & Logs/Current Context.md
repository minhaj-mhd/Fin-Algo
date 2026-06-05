# Current Context

This document tracks the active development topics, current focus, and immediate next steps for the Vanguard V2.3 project.

## Active Focus

* **Fix Upstox Rate Limits & Pandas Fragmentation**: Resolved Upstox Historical Data Cloudflare 429 Rate Limiting by throttling concurrent fetches from 15 to 5 workers and adding a 0.3s delay in `orchestrator.py`. Simultaneously resolved widespread Pandas "DataFrame is highly fragmented" warnings in `model_inference.py`, `orchestrator.py`, and `persistence.py` by refactoring repetitive `.assign()` chains using `pd.concat()` and appending `.copy()` correctly.
* **Fix Live Engine Pulse Scan Hang**: Identified and resolved a critical bug where the live Vanguard Engine would hang indefinitely at the "Pulse Scan initiated" step. The issue was caused by `yf.download()` calls for `^NSEI` and `^INDIAVIX` lacking a `timeout` parameter, causing the main thread to block forever when the Yahoo Finance API became unresponsive. Added `timeout=15` to enforce failover.
* **Fix Upstox Portfolio UI Bugs**: Successfully resolved an issue where closed trades and open/in-progress trades were not reflecting correctly in the Upstox portfolio tab and intraday tracking cards. Diagnosed and fixed a hardcoded SQL prefix (`T-%` instead of `TRADE-%`) bug in `database_manager.py` that caused the frontend summary APIs to return empty trade sets. Enabled tracking of unrealized P&L on the top dashboard Intraday cards by expanding the SQL queries to include active `"OPEN"` positions.
* **Vanguard Engine Modular Refactoring**: Successfully completed the **[[06. Context & Logs/Vanguard Engine Refactor Roadmap|Vanguard Engine Refactor Roadmap]]**, breaking down the 3,300+ line monolith into a clean, 9-module cohesive package under `scripts/vanguard/`. All imports, schedule scans, WebSocket integrations, and trade state machines are successfully isolated, and backward-compatible execution is verified.
* **Scripts Directory Restructuring**: Successfully organized and grouped 70+ scripts inside `scripts/` into modular subdirectories (`backtests/`, `training/`, `research/`, `collectors/`), keeping the `scripts/` root clean and aligned. Updated import references and verified zero system compilation or execution issues.
* **Codebase Streamlining & Phase 1/2 Cleanup**: Executed Phase 1 and 2 of the Codebase Cleanup Strategy. Successfully isolated 35 stale Python and Markdown files to `legacy_archive/`, updated the shared codebase index directory, and verified system compilation and dashboard snapshot APIs run with zero PyTorch or deprecated file dependencies.
* **Daily Gatekeeper Fix & PyTorch Cleanup**: Successfully pruned all retired daily temporal transformer PyTorch code from `vanguard_signal_engine.py`. Aligned the daily macro scan to the correct 165-feature `daily_xgb` schema, resolving a critical feature-mismatch bug and restoring true gatekeeper filters in live production.
* **Regime-Aware Routing**: Completed implementation and execution of `strategy_50_regime_backtest.py` across all 50 strategies, evaluating both gated and ungated passes for May 2026. Restored strategy-specific high-precision exits for top performers.
* **Market Psychology Indicators**: Architecting an intelligent layer that goes beyond technical analysis to quantify market psychology. Integrating order flow, volume profiles, and news sentiment into a predictive model to anticipate shifts in retail and institutional trader behavior ahead of momentum inflection points.
* **System Refinement**: Completed transition from Vanguard V1 static thresholds to V2.3 dynamic ATR-based logic, conviction scoring, and live UI strategy badges.
* **Codebase and Memory Layer Synchronization**: Successfully untracked cache directories to resolve repository bloat, aligned documentation files (`Codebase File Directory.md`, `Global System Architecture.md`, `Model Inference Data Structure.md`, etc.) to the new modular architecture, conducted a Gemini Pro audit, stripped dead tracker code from the AI veto logic, and pruned the `scratch/` directory of disposable one-off scripts.
* **30-Minute Model Complete Edge Research**: Conducted exhaustive 6-phase empirical analysis of the 30-minute XGBoost models via Jupyter MCP. Discovered the Long Model's edge concentrates at **15:15 IST** (60.2% WR, +32 bps at Score > 0.080), while the Short Model is structurally broken globally but has a narrow edge at **14:15 IST only** (56.3% WR). Signal inversions and Dual-Lock configurations provide negligible value — fundamentally different from the 1-hour model. Wrote 5 comprehensive reports with 7 visual assets to `08. Model Analysis/30-Minute Vanguard Model/`. See [[08. Model Analysis/30-Minute Vanguard Model/Complete Edge Catalog|30-Min Edge Catalog]].
* **Training Data & Regime Overfitting Discovery**: Uncovered the structural reason why the 1-hour model succeeds and the 30-min/15-min models fail: the **Row Count Fallacy**. Despite having equivalent raw row counts (~540K), the 1-hour model was trained on 3 years of calendar data (multiple regimes), while the intraday models were trained on just 1 year. Established the mandate that **all Vanguard models must be trained on a minimum of 3 years of historical data** to prevent regime overfitting. Documented in [[02. Model Suite/Training Data & Regime Requirements]].
* **Memory Integration**: Established this Obsidian vault as a shared memory layer for project continuity, updated with the latest 50-strategy backtest outcomes.
* **10:30 AM Momentum Strategy (V2 & V3)**: Completed implementation and testing of the V3 two-layer XGBoost strategy following the V2 forensic audit. V3 utilized a 4.5-year dataset (~1100 days) and vol-normalized features. Results conclusively confirm the model is noise: Layer A achieved 51.4% OOS accuracy, Layer B achieved 0.017 Spearman rho, and the best backtest threshold yielded flat returns with a -33.4% max drawdown. Standard tabular XGBoost models on intraday OHLCV cannot robustly predict 10:30 momentum returns. The project for Strategy 1030 has been shelved. Details in [[06. Context & Logs/Conversations/Conv-2026-06-03-Strategy-1030-V3|Conversation Log: 10:30 Momentum Strategy V3 Run]].

## Next Steps

* [x] Build the first real `tests/` suite for trade lifecycle, risk math, feature-schema validation, and broker adapter mocks.
* [x] Audit tracked generated artifacts under `data/`, `models/`, and raw cache folders, then move non-canonical outputs out of git tracking.
* [ ] Deprecate or archive random-split training scripts and document walk-forward/temporal validation as the production standard.
* [x] Rewrite `README.md` into a clean operator guide with fixed encoding, current run modes, caveats, token requirements, and safety limits.
* [x] Pin dependencies or add a lock file so future installs reproduce the current working environment.
* [ ] Complete Phase 3 of the **[[06. Context & Logs/Codebase Cleanup Strategy|Codebase Cleanup Strategy]]** (permanent pruning after 5 consecutive days of sandbox observation).
* [ ] Review and deploy the top-performing gated structural strategies (S3, S4, S13, S19, S23, S24) in the production live engine environment.
* [ ] Build predictive models for Market Psychology using volume profiles and order flow data.

Linked to: [[Welcome|Main Navigation Index]]

Concluded Session: [[06. Context & Logs/Conversations/Conv-2026-06-04-Fix-Upstox-Portfolio-UI-Bugs|Conversation Log: Fix Upstox Portfolio UI Bugs]]
Concluded Session: [[06. Context & Logs/Daily Logs/2026-06-04|Conversation Log: Model Comparison Query]]
Concluded Session: [[06. Context & Logs/Daily Logs/2026-06-04|Conversation Log: MCP Tools Suggestions]]
Concluded Session: [[06. Context & Logs/Daily Logs/2026-06-04|Conversation Log: MCP Tools Integration]]
Concluded Session: [[06. Context & Logs/Conversations/Conv-2026-06-04-Database-Health-Analysis|Conversation Log: Database Health Analysis]]
Concluded Session: [[06. Context & Logs/Conversations/Conv-2026-06-04-Jupyter-XGBoost-Visualizations|Conversation Log: Jupyter XGBoost Visualizations]]
Concluded Session: [[06. Context & Logs/Conversations/Conv-2026-06-04-30M-Model-Edge-Analysis|Conversation Log: 30M Model Edge Analysis]]
Concluded Session: [[06. Context & Logs/Conversations/Conv-2026-06-05-Debug-Pulse-Scan-Stuck|Conversation Log: Debug Pulse Scan Stuck Issue]]
Concluded Session: [[06. Context & Logs/Conversations/Conv-2026-06-05-Fix-Upstox-Rate-Limit-and-Pandas-Warnings|Conversation Log: Fix Upstox Rate Limit & Pandas Warnings]]
