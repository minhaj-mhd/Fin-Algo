# Current Context

This document tracks the active development topics, current focus, and immediate next steps for the Vanguard V2.3 project.

## Active Focus

* **Vanguard Engine Modular Refactoring**: Successfully completed the **[[06. Context & Logs/Vanguard Engine Refactor Roadmap|Vanguard Engine Refactor Roadmap]]**, breaking down the 3,300+ line monolith into a clean, 9-module cohesive package under `scripts/vanguard/`. All imports, schedule scans, WebSocket integrations, and trade state machines are successfully isolated, and backward-compatible execution is verified.
* **Scripts Directory Restructuring**: Successfully organized and grouped 70+ scripts inside `scripts/` into modular subdirectories (`backtests/`, `training/`, `research/`, `collectors/`), keeping the `scripts/` root clean and aligned. Updated import references and verified zero system compilation or execution issues.
* **Codebase Streamlining & Phase 1/2 Cleanup**: Executed Phase 1 and 2 of the Codebase Cleanup Strategy. Successfully isolated 35 stale Python and Markdown files to `legacy_archive/`, updated the shared codebase index directory, and verified system compilation and dashboard snapshot APIs run with zero PyTorch or deprecated file dependencies.
* **Daily Gatekeeper Fix & PyTorch Cleanup**: Successfully pruned all retired daily temporal transformer PyTorch code from `vanguard_signal_engine.py`. Aligned the daily macro scan to the correct 165-feature `daily_xgb` schema, resolving a critical feature-mismatch bug and restoring true gatekeeper filters in live production.
* **Regime-Aware Routing**: Completed implementation and execution of `strategy_50_regime_backtest.py` across all 50 strategies, evaluating both gated and ungated passes for May 2026. Restored strategy-specific high-precision exits for top performers.
* **Market Psychology Indicators**: Architecting an intelligent layer that goes beyond technical analysis to quantify market psychology. Integrating order flow, volume profiles, and news sentiment into a predictive model to anticipate shifts in retail and institutional trader behavior ahead of momentum inflection points.
* **System Refinement**: Completed transition from Vanguard V1 static thresholds to V2.3 dynamic ATR-based logic, conviction scoring, and live UI strategy badges.
* **Memory Integration**: Established this Obsidian vault as a shared memory layer for project continuity, updated with the latest 50-strategy backtest outcomes.

## Next Steps

* [ ] Build the first real `tests/` suite for trade lifecycle, risk math, feature-schema validation, and broker adapter mocks.
* [ ] Audit tracked generated artifacts under `data/`, `models/`, and raw cache folders, then move non-canonical outputs out of git tracking.
* [ ] Deprecate or archive random-split training scripts and document walk-forward/temporal validation as the production standard.
* [ ] Rewrite `README.md` into a clean operator guide with fixed encoding, current run modes, caveats, token requirements, and safety limits.
* [ ] Pin dependencies or add a lock file so future installs reproduce the current working environment.
* [ ] Complete Phase 3 of the **[[06. Context & Logs/Codebase Cleanup Strategy|Codebase Cleanup Strategy]]** (permanent pruning after 5 consecutive days of sandbox observation).
* [ ] Review and deploy the top-performing gated structural strategies (S3, S4, S13, S19, S23, S24) in the production live engine environment.
* [ ] Build predictive models for Market Psychology using volume profiles and order flow data.

Linked to: [[Welcome|Main Navigation Index]]

Concluded Session: [[06. Context & Logs/Conversations/Conv-2026-06-03-Scripts-Restructuring|Conversation Log: Scripts Restructuring]]

