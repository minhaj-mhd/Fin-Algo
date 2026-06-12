---
title: "Vanguard Engine Refactor Roadmap"
type: spec
status: active
updated: 2026-06-12
tags: []
---
# Vanguard Engine Refactor Roadmap
## Why and How to Split `vanguard_signal_engine.py`

This document converts the latest project analysis into an executable refactor plan. The goal is not to rewrite Vanguard from scratch. The goal is to separate live trading responsibilities so the system becomes safer to test, easier to reason about, and harder to accidentally break.

---

## Why the Engine Must Be Split

`scripts/vanguard_signal_engine.py` has grown into the runtime center of the whole system. It currently carries too many responsibilities in one file:

- Model registry loading and XGBoost inference.
- Daily macro gatekeeping.
- Live Upstox price and order interactions.
- Gemini/AI sentiment veto logic.
- Strategy filtering and signal selection.
- Trade state transitions: pending, open, vetoed, closed, stop loss, take profit.
- Portfolio capital, margin, charges, and P&L accounting.
- SQLite logging and dashboard state updates.
- Main loop scheduling, market-hour behavior, and error handling.

This is risky because a change to one concern can accidentally change another. For example, improving Gemini retry behavior should not be able to affect margin accounting. Adjusting signal ranking should not require editing order placement code. The current shape makes those boundaries blurry.

For a trading system, this is more than style. Separation is a safety feature. Each layer should have a small contract, focused tests, and clear failure behavior.

---

## Target Module Split

The refactor should preserve behavior first. Do not redesign the strategy while moving code. Extract one boundary at a time and keep the live engine running after each step.

### 1. `scripts/vanguard/config.py`

Owns runtime settings.

Responsibilities:
- Market hours and timezone constants.
- Capital defaults and allocation settings.
- Brokerage/slippage assumptions.
- Model paths and registry defaults.
- Feature flags such as Gemini enabled, sandbox/live mode, and WebSocket enabled.

Why:
- Removes scattered magic numbers.
- Makes paper/live safety switches visible.
- Gives future tests a clean way to override config.

### 2. `scripts/vanguard/model_inference.py`

Owns model loading and scoring.

Responsibilities:
- Load active model from `ModelRegistry`.
- Load long/short XGBoost boosters.
- Load daily macro gatekeeper models.
- Validate feature schema before inference.
- Score ticker feature frames and return structured scores.

Why:
- Model loading failure should be isolated from broker execution.
- Feature mismatch bugs become easier to test.
- It becomes possible to backtest the exact production inference path.

### 3. `scripts/vanguard/signal_generation.py`

Owns signal selection before external vetoes.

Responsibilities:
- Combine raw long/short model scores.
- Apply daily gatekeeper output.
- Apply `StrategyFilters`.
- Deduplicate ticker/side signals.
- Return candidate signals with strategy id, scores, and reason fields.

Why:
- Strategy routing becomes inspectable without placing orders.
- The dashboard and backtests can reuse the same signal path.
- It gives a clean checkpoint for "what would Vanguard trade before AI/broker constraints?"

### 4. `scripts/vanguard/ai_veto.py`

Owns Gemini and market-intelligence veto behavior.

Responsibilities:
- Gemini key rotation and usage tracking.
- Prompt construction.
- AI sentiment/veto decision parsing.
- TradingView sentiment checks if still used as a veto input.
- Explicit fallback behavior when AI services fail.

Why:
- AI failures should degrade predictably.
- Prompt changes should not touch trade execution.
- Veto statistics can be tested and audited separately.

### 5. `scripts/vanguard/broker_adapter.py`

Owns broker-facing execution calls.

Responsibilities:
- Wrap `UpstoxSandboxBroker`.
- Normalize order responses.
- Provide paper/sandbox/live mode abstraction.
- Enforce "no live order unless explicitly enabled" guardrails.
- Return typed order results instead of raw SDK objects.

Why:
- Broker SDK quirks stay contained.
- Sandbox and live safety boundaries become explicit.
- Order placement can be mocked in tests.

### 6. `scripts/vanguard/trade_state.py`

Owns trade lifecycle and state transitions.

Responsibilities:
- Pending entry confirmation.
- Open trade tracking.
- Stop loss, take profit, trailing stop, breakeven, hard-close, and EOD close transitions.
- Duplicate ticker prevention.
- Trade event records.

Why:
- This is the highest-risk logic in the system.
- It should be testable without Upstox, Gemini, Flask, or model files.
- A state machine makes unintended transitions easier to catch.

### 7. `scripts/vanguard/risk_manager.py`

Owns sizing and portfolio accounting.

Responsibilities:
- Position sizing.
- Capital availability.
- Used margin.
- Brokerage/slippage/charges accounting.
- Per-trade and per-day risk limits.
- Max open positions and concentration checks.

Why:
- Risk logic should be boring, deterministic, and heavily tested.
- It should be impossible for signal code to silently bypass sizing constraints.

### 8. `scripts/vanguard/persistence.py`

Owns writing and reading runtime state.

Responsibilities:
- SQLite trade writes.
- Stats JSON writes.
- Latest scores JSON writes.
- Ledger file updates if still required.
- Atomic writes for dashboard-consumed files.

Why:
- Disk I/O and schema migrations should not be mixed into decision logic.
- Dashboard state becomes easier to validate.

### 9. `scripts/vanguard/orchestrator.py`

Owns the live loop only.

Responsibilities:
- Initialize components.
- Schedule scan cycles.
- Handle market-hour gates.
- Call model inference, signal generation, AI veto, risk manager, broker adapter, trade state, and persistence in order.
- Log high-level lifecycle events.

Why:
- The orchestrator should be thin enough to read in one sitting.
- This file becomes the operational entrypoint.
- Most logic becomes testable outside the live loop.

---

## Safe Migration Strategy

Do this in small extraction phases. After each phase, run import/compile checks and a smoke run in sandbox mode.

### Phase 1: No-Behavior-Change Scaffolding

- Create `scripts/vanguard/` package with empty modules.
- Move constants into `config.py`.
- Keep `vanguard_signal_engine.py` as the entrypoint.
- Add imports from the new package, but preserve current behavior.

Exit criteria:
- Existing batch file still launches the engine.
- Dashboard APIs still read the same state files.
- No live trading behavior changes.

### Phase 2: Extract Model Inference

- Move model registry loading into `model_inference.py`.
- Move daily gatekeeper model loading and schema validation into the same module.
- Keep all returned score column names identical.

Exit criteria:
- Active model and daily gatekeeper load exactly as before.
- A small smoke script can score a saved feature frame without starting the live engine.

### Phase 3: Extract Signal Generation

- Move score combination, rank selection, and strategy filter application into `signal_generation.py`.
- Preserve the current `StrategyFilters` behavior first, even where it is a live proxy.
- Add a clear TODO for replacing proxy strategy logic with exact production definitions.

Exit criteria:
- Given a saved `latest_scores`-style input, the new module emits the same candidate signals as the old path.

### Phase 4: Extract Trade State and Risk

- Move lifecycle transitions into `trade_state.py`.
- Move capital, sizing, margin, brokerage, and P&L math into `risk_manager.py`.
- Introduce small dataclasses or dictionaries with stable fields before going deeper.

Exit criteria:
- Unit tests cover pending-to-open, open-to-stop-loss, open-to-take-profit, hard-close, EOD close, and duplicate prevention.
- Sizing and charge math is tested without live broker calls.

### Phase 5: Extract Broker Adapter and Persistence

- Wrap Upstox order placement behind `broker_adapter.py`.
- Normalize order response shapes.
- Move SQLite/JSON/ledger writes into `persistence.py`.

Exit criteria:
- Order placement can be fully mocked.
- Persistence can be tested with a temporary DB/path.
- The live loop no longer directly writes raw DB/JSON state.

### Phase 6: Thin Orchestrator

- Move the main loop into `orchestrator.py`.
- Reduce `vanguard_signal_engine.py` to a compatibility entrypoint that calls the orchestrator.
- Update docs and run scripts once the new entrypoint is stable.

Exit criteria:
- `vanguard_signal_engine.py` is small and boring.
- Most code paths have isolated tests.
- Live/sandbox mode is explicit at startup.

---

## Project-Wide Next-Move Tasks

### A. Engine Split and Runtime Safety

- [ ] Create `scripts/vanguard/` package.
- [ ] Extract config constants and safety flags.
- [ ] Extract model loading and feature-schema validation.
- [ ] Extract signal generation and strategy routing.
- [ ] Extract AI veto/key-rotation logic.
- [ ] Extract trade lifecycle state transitions.
- [ ] Extract risk and portfolio accounting.
- [ ] Extract broker adapter with sandbox/live guardrails.
- [ ] Extract persistence for SQLite, JSON state, and ledgers.
- [ ] Reduce `scripts/vanguard_signal_engine.py` to a compatibility entrypoint.

### B. Data and Artifact Hygiene

- [ ] Audit tracked files under `data/`, `models/`, and cache folders.
- [ ] Decide which model artifacts are canonical and which are reproducible/generated.
- [ ] Stop tracking large ranking CSVs and raw cache outputs.
- [ ] Keep generated datasets outside git, or manage them through a dedicated artifact workflow.
- [ ] Add a small `data/README.md` explaining which files are source data, generated data, runtime state, and ignored cache.
- [ ] Ensure `.gitignore` covers `data/*.csv`, DB files, logs, raw caches, latest scores, and scratch outputs.

### C. Test Coverage and Verification

- [ ] Add a real `tests/` directory outside `scratch/`.
- [ ] Add tests for feature generation no-lookahead behavior.
- [ ] Add tests for model feature schema validation.
- [ ] Add tests for strategy filter deduplication and rank handling.
- [ ] Add tests for trade lifecycle transitions.
- [ ] Add tests for risk sizing, charges, margin use, and P&L math.
- [ ] Add tests for broker adapter response normalization using mocks.
- [ ] Add a smoke command that imports the engine without starting live execution.

### D. Training and Validation Discipline

- [ ] Mark random-split training scripts as deprecated or archive-only.
- [ ] Document the active validation standard: temporal split or walk-forward only.
- [ ] Record which model versions were trained with which validation style.
- [ ] Add leakage checks for target columns, future returns, and feature windows.
- [ ] Reconcile model registry claims against actual metadata files.
- [ ] Separate "research result" metrics from "production acceptance" metrics.

### E. Documentation and Operator Clarity

- [ ] Rewrite `README.md` as an operator-focused guide.
- [ ] Fix README encoding/mojibake.
- [ ] Replace promotional accuracy claims with current validation caveats.
- [ ] Document active run modes: backtest, sandbox shadow, sandbox order, live order if ever enabled.
- [ ] Document required environment variables and token scopes.
- [ ] Document safety limits and failure behavior.
- [ ] Link this roadmap from `Welcome.md`, `Current Context.md`, and the codebase directory map.

### F. Dependency and Environment Reproducibility

- [ ] Pin dependency versions or add a lock file.
- [ ] Add a clean environment bootstrap path that does not depend on a checked-in `env/`.
- [ ] Verify Python version expectations.
- [ ] Add a smoke validation command after install.
- [ ] Keep virtual environments out of source control.

---

## Refactor Rule

Every extraction should preserve behavior before improving behavior. First move code behind a boundary, then test the boundary, then improve the logic. This keeps the live system from changing shape and changing strategy at the same time.

---

Linked to: [[06 — Logs/Active Board]] - [[Codebase Cleanup Strategy]] - [[01 — Architecture/Data & Code/Codebase File Directory|Codebase File Directory]]
