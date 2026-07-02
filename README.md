# Fin-Algo — NSE Intraday ML Trading Research Platform & Vanguard Engine

A research-first algorithmic trading system for the **Indian equity market (NSE)**. It combines a
cross-sectional machine-learning ranking stack, a statistically disciplined validation framework
(the **Gauntlet**), and **Vanguard** — a live 15-minute paper-trading engine with layered risk
vetoes (candle microstructure guards + a dual-stage Gemini LLM audit) executing against the
Upstox sandbox.

> [!IMPORTANT]
> **🤖 AI agents**: if you are an AI coding agent pair-programming in this repository, you **must**
> read and follow [`AGENTS.md`](AGENTS.md) as your very first action. It defines the cross-session
> memory protocol (Obsidian vault under [`finalgo-memory-layer/`](finalgo-memory-layer/)), the
> model-metric discipline, and hard-learned engineering rules.

---

## Table of Contents

1. [Philosophy](#philosophy)
2. [System Architecture](#system-architecture)
3. [The Vanguard Live Engine](#the-vanguard-live-engine)
4. [The Validation Gauntlet](#the-validation-gauntlet)
5. [Model Suite & Verdicts](#model-suite--verdicts)
6. [Research Library & Key Findings](#research-library--key-findings)
7. [Repository Layout](#repository-layout)
8. [Getting Started](#getting-started)
9. [Costs, Conventions & Market Rules](#costs-conventions--market-rules)
10. [The Memory Layer](#the-memory-layer)
11. [Large-File Policy](#large-file-policy)
12. [Disclaimer](#disclaimer)

---

## Philosophy

Three principles separate this repo from a typical "backtest looks great" project:

1. **Net-of-cost or it didn't happen.** Every edge is evaluated after brokerage, STT, and
   slippage. Many models here show *real, statistically significant* cross-sectional skill that
   is nonetheless **not tradeable** because it sits below the ~10 bps round-trip cost line — and
   the repo says so explicitly rather than hiding it.
2. **One source of verdict authority.** Only the [Validation Gauntlet](#the-validation-gauntlet)
   can grade a model. Ad-hoc backtest scripts are exploratory and carry no authority. Every
   performance claim must cite a Gauntlet `run_id` or be labeled `⚠️ UNVERIFIED`.
3. **Negative results are first-class.** Dead ends (directional classifiers, transformer
   architectures, stop-loss schemes, horizon sweeps…) are documented in the memory vault so no
   future session re-burns the same compute. The honest summary of 20+ model generations: the
   1-hour price/volume information ceiling is ρ ≈ 0.03 — real, but sub-cost. The levers that
   remain are **new data** and **execution**, not architecture.

---

## System Architecture

```
                        ┌────────────────────────────────────────────┐
                        │        DATA LAYER  (scripts/collectors/)    │
                        │  Upstox candles: 5m/15m/30m/1h/daily (10y)  │
                        │  yfinance global daily · NIFTY500 index     │
                        │  live option-chain OI snapshots · websocket │
                        └───────────────┬────────────────────────────┘
                                        │ cached to data/ (parquet/csv)
                 ┌──────────────────────┼───────────────────────┐
                 ▼                      ▼                       ▼
   ┌───────────────────────┐ ┌────────────────────┐ ┌─────────────────────────┐
   │ FEATURE / PANEL BUILD │ │  MODEL TRAINING    │ │  RESEARCH (exploratory) │
   │ scripts/features/     │ │  scripts/training/ │ │  scripts/research/      │
   │ scripts/labeling/     │ │  scripts/          │ │  scripts/analysis/      │
   │ ~86-feature vectors   │ │    transformer/    │ │  scripts/structural/    │
   └───────────┬───────────┘ └─────────┬──────────┘ │  no verdict authority   │
               │                       │            └─────────────────────────┘
               │                       ▼
               │        ┌───────────────────────────────┐
               │        │  VALIDATION GAUNTLET          │
               │        │  scripts/gauntlet/            │
               │        │  purged WF splits · leakage   │
               │        │  audit · cost model · ledger  │
               │        │  (multiple-testing deflation) │
               │        │  → verdicts + registry stamps │
               │        └───────────────┬───────────────┘
               │                        │ models/registry.json (active_model)
               ▼                        ▼
   ┌────────────────────────────────────────────────────────────────┐
   │                 VANGUARD LIVE ENGINE (scripts/vanguard/)       │
   │                                                                │
   │  15-min scan → cross-sectional rank (172 NSE names)            │
   │    → Top-K signal (K=3) → Candle Confirmation Layer            │
   │    → Dual-Stage Gemini AI Veto → Risk Manager (ATR brackets)   │
   │    → Upstox sandbox execution → SQLite ledger                  │
   │                                                                │
   │  network_monitor.py halts the loop during connectivity loss    │
   └───────────────────────────┬────────────────────────────────────┘
                               │
                               ▼
                 ┌──────────────────────────────┐
                 │  FLASK DASHBOARD (port 5001) │
                 │  scripts/vanguard_dashboard  │
                 └──────────────────────────────┘
```

---

## The Vanguard Live Engine

`scripts/vanguard/` is the modular production engine; `scripts/vanguard_signal_engine.py` is its
entry point. Every 15 minutes during market hours it runs one full pipeline pass:

### 1. Signal generation (`signal_generation.py`, `model_inference.py`)
- Builds the feature vector for each of the **172 liquid NSE names** (`scripts/tickers.py`).
- The active ranker is resolved from `models/registry.json` → currently **`v20_rolling_1h`**,
  which scores **overlapping trailing-1h candles stepped every 15 minutes**
  (`ROLLING_1H_CANDLES = True` in `config.py`), so a fresh 1-hour-horizon signal exists at every
  scan instead of only on the hour.
- Long and short boosters produce a cross-sectional conviction ranking; the **top K = 3**
  candidates per side become trade candidates (`ENTRY_TOP_K`).

### 2. Candle Confirmation Layer (`trade_state.py`, master switch `CANDLE_LAYER_ENABLED`)
Microstructure guards derived from live post-mortems, applied to the last completed 15-m bar:
- **Direction confirmation** — the look-back bar must agree with the trade direction; if it
  fails, the engine places a patient *pending-limit* fade toward the bar extreme instead of a
  market order.
- **Violent-thrust veto** — cancels fades into bars with >2.5% range closing in the extreme
  quartile against the trade (prevents shorting into a breakout rip).
- **Fade-Entry Quality Guard** — volume-gated knife-catch protection: blocks fading
  heavy-volume breakouts at fresh 52-week highs (rvol ≥ 1.5 within 0.5% of the high) and
  adverse-extreme closes with real participation. *Thresholds fitted to one session —
  `⚠️ UNVERIFIED`, flagged as such in `config.py`.*
- **Live-reversal veto** — checks the in-progress 1-m candle before commitment.

### 3. Dual-Stage Gemini AI Veto (`ai_veto.py`, `scripts/gemini_client_manager.py`)
- **Stage 1 — technical triage** (`gemini-3.5-flash` tier, rotated): structural-wall and
  momentum-trap checks against dynamic S/R, RSI, and band context. Bypassable at runtime with
  `GEMINI_S1_VETO_ENABLED=0` for A/B testing.
- **Stage 2 — news & governance audit** (`gemini-2.5-flash` tier): Google-grounded search for
  block deals, earnings shocks, and SEBI/regulatory actions; fundamental reality overrides
  technicals.
- API keys auto-rotate across a comma-separated pool with per-day budget tracking and fast
  503/timeout escalation.

### 4. Risk management (`risk_manager.py`)
- **ATR-based brackets** from 15-m resampled ATR (no static percentage targets).
- Breakeven locks and trailing stops engage once a profitability buffer is reached.
- Entry window **10:15–14:15 IST** (`FIRST_ENTRY_TIME`/`LAST_ENTRY_TIME`) so every 1-h hold can
  complete before the **15:15 IST hard EOD flush**.
- Capital model: 5 trade slots, 5× intraday margin (`config.py`).

### 5. Execution & persistence (`broker_adapter.py`, `persistence.py`)
- `SANDBOX_MODE = True` by default — all fills are paper trades against the Upstox sandbox,
  recorded in the SQLite ledger `data/vanguard_trades.db`.
- `network_monitor.py` probes connectivity and **halts the engine** (rather than trading blind)
  during internet/broker outages, resuming with linear backoff.

### Dashboard
`python scripts/vanguard_dashboard.py` → Flask UI at **http://127.0.0.1:5001** with live shadow
trades, portfolio P&L, per-ticker and per-strategy detail pages, and veto diagnostics
(`templates/`).

---

## The Validation Gauntlet

`scripts/gauntlet/` is the **only** component allowed to issue model grades. It exists because
this repo repeatedly caught "great" results that were artifacts (test-set leakage, overnight
returns bleeding into intraday labels, cost-sign bugs, zero-filled features).

```powershell
python -m scripts.gauntlet run --model <name> ...   # pre-registered evaluation run
python -m scripts.gauntlet ledger                   # inspect the run ledger
python -m scripts.gauntlet leakage-audit            # standalone leakage checks
python -m scripts.gauntlet selftest                 # harness self-test
```

Key properties:

- **Purged walk-forward splits** (`splits.py`) — no static test sets; embargoes against
  look-ahead.
- **Leakage audit** (`leakage.py`) and **data audit** (`data_audit.py`) run before scoring.
- **Full Indian cost model** (`costs.py`) — brokerage, STT, slippage per side.
- **Multiple-testing correction** (`ledger.jsonl`) — every run on a dataset family *deflates the
  t-thresholds for all future runs* on that family. Runs are therefore rationed: one
  pre-registered run per hypothesis, no batch sweeps, no "rerun until it passes".
- **Registry stamping** (`registry.py`) — verdicts are written to model metadata with a SHA-256
  stamp checksum; manual stamp edits are prohibited.
- **Verdict scale** — `DEAD` (no exploitable signal) · `FILTER_GRADE` (real cross-sectional
  skill, sub-cost standalone; usable as a filter/ranker inside a richer pipeline) ·
  `TRIGGER_GRADE` (net-positive as an actual trade trigger).
- `GAUNTLET_ENFORCEMENT` is in `"warn"` mode during rollout (see `scripts/vanguard/config.py`).

---

## Model Suite & Verdicts

`models/registry.json` is the source of truth; `active_model` is currently **`v20_rolling_1h`**.
Latest Gauntlet verdicts per model (each row cites its `run_id`, per repo metric discipline):

| Model | Long | Short | Gauntlet run |
|---|---|---|---|
| **`v20_rolling_1h`** (active) | FILTER_GRADE | FILTER_GRADE | `20260615T175149Z-5f7d069f` |
| `v10_native_1h` (prev. active) | FILTER_GRADE | FILTER_GRADE | `20260610T184210Z-d795438c` |
| `v10_depth4_1h` | FILTER_GRADE | FILTER_GRADE | `20260610T184210Z-d795438c` |
| `daily_macro_v2` (3-day horizon) | **TRIGGER_GRADE** | FILTER_GRADE | `20260610T135608Z-5f7d069f` |
| `daily_macro_v3` | DEAD | FILTER_GRADE | `20260610T144343Z-5f7d069f` |
| `v8_upstox_3y` | FILTER_GRADE | FILTER_GRADE | `20260610T172623Z-d795438c` |
| `v2_15min_3y` | FILTER_GRADE | FILTER_GRADE | `20260610T173707Z-d795438c` |
| `v2_30min_v3_3y` | FILTER_GRADE | FILTER_GRADE | `20260611T155824Z-5f7d069f` |
| `v9_clean_1h` | FILTER_GRADE | FILTER_GRADE | `20260610T110022Z-5f7d069f` |
| `v11_utility_1h` | FILTER_GRADE | FILTER_GRADE | `20260610T184210Z-d795438c` |
| `v12`–`v16` LambdaMART/NDCG/breakout variants | mixed | mostly FILTER_GRADE | `20260610T18*Z-d795438c` |
| `v17_random_forest_1h` | DEAD | DEAD | `20260610T121944Z-5f7d069f` |
| `v18_random_forest_1h` | DEAD | FILTER_GRADE | `20260610T124108Z-5f7d069f` |
| `v19_catboost_1h` | FILTER_GRADE | FILTER_GRADE | `20260610T191237Z-d795438c` |
| `daily_xgb` | DEAD | DEAD | `20260610T102743Z-5f7d069f` |

The headline: **many models carry real, certified cross-sectional skill (FILTER_GRADE), but only
`daily_macro_v2`'s long side has graded as an actual net-positive trigger.** The intraday stack
trades in sandbox/paper mode precisely because certified skill ≠ certified post-cost profit.

Notable non-registry work (transformers, TBM ensembles, graph features) lives under `models/` and
`scripts/transformer/` with its post-mortems in the memory vault.

---

## Research Library & Key Findings

`scripts/research/`, `scripts/analysis/`, `scripts/transformer/`, and `scripts/structural/` hold
the exploratory codebase. **Nothing in this section carries verdict authority** — findings are
exploratory research, documented so they aren't re-run blindly. Highlights (full write-ups in the
memory vault):

- **The 1-h information ceiling.** Across XGBoost generations, CatBoost, random forests,
  LambdaMART objectives, hyper-parameter sweeps (Optuna), and five transformer architectures
  (dual-resolution, sided/veto-gate, co-sign confirmation, daily veto, cross-sectional with
  lead-lag or SMC/price-action features, level-graph GCN): cross-sectional rank-IC converges to
  **ρ ≈ 0.03 and will not cross the ~10 bps cost line**. Architecture is not the lever;
  price/volume at 1 h is information-limited.
- **Horizon is not a lever either.** 1h/2h/3h holds show flat skill; longer "gross" edges are
  market drift exposed by negative controls.
- **Directional (non-cross-sectional) classification is dead** — v18/v19-style ">0 bps"
  classifiers are coin-flips after costs.
- **Stops can't rescue a sub-cost book.** Win/loss distributions are near-symmetric and the
  universe mean-reverts, so every stop width tested clips recoverable dips as often as it saves
  real losses.
- **Overnight gap-fade is the strongest exploratory lead** (`⚠️ UNVERIFIED`, no Gauntlet run):
  fading the open-auction gap (short top gap-ups / long bottom gap-downs at 09:15, |gap| ≤ 3%)
  survived a seven-test kill battery — but ~90% of the edge is the open print itself, so it is an
  **execution problem** (NSE pre-open auction fills) before it is a modeling problem.
- **Beware artifact edges.** Multiple past "breakthroughs" dissolved under audit: overnight
  returns leaking into intraday labels, a cost-sign bug that *added* costs to the benchmark
  side, silently zero-filled features, and resolution artifacts from 15-min-close simulation.
  The audit checklist lives in `AGENTS.md` and the vault.

---

## Repository Layout

```
finalgo/
├── AGENTS.md                  # AI-agent operating protocol (read first)
├── CLAUDE.md                  # imports AGENTS.md for Claude Code
├── bootstrap.py / .bat        # environment bootstrap
├── run_vanguard_system.bat    # one-click: signal engine + dashboard
├── requirements.txt           # pinned deps (Python 3.12, torch cu128, xgboost 3.2)
│
├── scripts/
│   ├── vanguard/              # live engine: orchestrator, signal_generation,
│   │                          #   model_inference, ai_veto, risk_manager, trade_state,
│   │                          #   broker_adapter, persistence, network_monitor, config
│   ├── vanguard_signal_engine.py   # engine entry point
│   ├── vanguard_dashboard.py       # Flask dashboard (port 5001)
│   ├── gauntlet/              # validation framework (run via python -m scripts.gauntlet)
│   ├── collectors/            # Upstox/yfinance data collection + upstox_login.py
│   ├── features/  labeling/   # feature engineering & label construction
│   ├── training/              # XGBoost walk-forward training pipelines
│   ├── transformer/           # PyTorch research stack (panels, training, evals)
│   ├── structural/            # relation-graph features (sector/business-group)
│   ├── research/  analysis/   # exploratory studies (no verdict authority)
│   ├── memory/build_index.py  # regenerates the vault indexes/MOCs
│   ├── upstox_broker.py  upstox_websocket.py  gemini_client_manager.py
│   └── tickers.py             # the 172-name NSE universe
│
├── models/                    # model zoo v1→v20 + daily_macro + registry.json
├── data/                      # candle caches, panels, ledgers (large files ignored)
├── templates/                 # dashboard HTML
├── tests/                     # pytest suite (run against sandbox paths only)
└── finalgo-memory-layer/      # Obsidian memory vault (00–09 taxonomy)
```

---

## Getting Started

### 1. Environment

Python **3.12** (the checked-in tooling assumes the venv lives at `env/`):

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
```

A CUDA GPU is optional; it accelerates the transformer research stack (`torch 2.11 + cu128`)
and XGBoost training. Inference for the live engine runs fine on CPU.

### 2. Configuration

Create `.env` in the repo root:

```env
# Upstox API (data + sandbox execution)
UPSTOX_API_KEY="..."
UPSTOX_API_SECRET="..."

# Gemini keys, comma-separated for auto-rotation
GEMINI_API_KEYS="key1,key2,key3"
BACKUP_GEMINI_API_KEY="emergency_key"

# Optional runtime switches (defaults shown)
GEMINI_S1_VETO_ENABLED=1        # Stage-1 flash veto on/off
CANDLE_LAYER_ENABLED=1          # candle confirmation layer master switch
LIMIT_EXPIRY_MARKET_FILL_ENABLED=0
CONVICTION_FLIP_EXIT_ENABLED=0
```

Authenticate the Upstox session (daily token) via `python scripts/collectors/upstox_login.py`.

### 3. Data

Candle history is collected into `data/` by the scripts in `scripts/collectors/`
(e.g. `collect_upstox_1h_v3.py`, `collect_upstox_15min_3y.py`, `collect_upstox_daily_10y.py`).
Note: **historical F&O open interest is paywalled** behind Upstox Plus; only the live option-chain
snapshot is free.

### 4. Run

```powershell
# One-click (dashboard + engine in separate windows)
.\run_vanguard_system.bat

# Or manually
python scripts\vanguard_signal_engine.py
python scripts\vanguard_dashboard.py     # → http://127.0.0.1:5001
```

`SANDBOX_MODE = True` in `scripts/vanguard/config.py` keeps everything in paper mode. Do not flip
it until you have independently validated the pipeline against your own account and risk limits.

### 5. Tests

```powershell
pytest
```

The suite covers the broker adapter, candle tracking, model inference, orchestrator concurrency,
risk manager, and trade state. **Tests must never touch production state** — they run against
sandboxed paths (see `AGENTS.md`, Engineering Discipline §1).

---

## Costs, Conventions & Market Rules

- **Timezone**: everything is IST (`Asia/Kolkata`); market 09:15–15:30, engine hard-close 15:15.
  NSE holidays for 2026 are tabled in `scripts/vanguard/config.py`.
- **Cost model** (Upstox intraday): ₹10 brokerage per order (₹20 round-trip), STT 0.025% on the
  sell side, 0.03% slippage per leg. Research evaluations conventionally use a **10 bps
  round-trip cost line**.
- **Timestamp labeling**: 1-h candles are labeled by *left* edge, 15-m candles by *right* edge —
  misaligning these (or letting overnight returns leak into "intraday" labels) has manufactured
  fake edges before. See the vault's ranking-data conventions note before any cross-timeframe
  work.
- **Rolling vs anchored candles**: `v20_rolling_1h` requires trailing-1h bars stepped every
  15 min; rollback to `v10_native_1h` requires flipping `ROLLING_1H_CANDLES = False` *and*
  repointing `registry.json`.

---

## The Memory Layer

[`finalgo-memory-layer/finalgo/`](finalgo-memory-layer/finalgo/) is an Obsidian vault providing
cross-session memory for AI-assisted development: architecture specs, per-model documentation,
Gauntlet reports, research post-mortems, dead-end registers, and daily logs under a fixed `00–09`
taxonomy. Rules that keep it trustworthy:

- Indexes (`Welcome.md`, `_MOC.md`, `INDEX.json`) are **generated** — never hand-edit; run
  `python scripts/memory/build_index.py` after adding or moving docs.
- Every doc carries YAML front-matter (`title`, `type`, `status`, `updated`).
- Any performance number in the vault must cite a Gauntlet `run_id` or carry `⚠️ UNVERIFIED`.

The full protocol lives in [`AGENTS.md`](AGENTS.md).

---

## Large-File Policy

GitHub rejects files >100 MB, so on 2026-07-02 all blobs >50 MB were stripped from git history
(`git filter-repo`). Regenerable data artifacts — `data/structural_panel_15m.parquet`,
`data/tbm_feature_views/`, `data/strategy_1030/dataset_stocks.csv`, plus the candle caches and
research panels — are `.gitignore`d and rebuilt via the collector/panel scripts. Check
`.gitignore` **before** running artifact-producing jobs; generated files >1 MB need an explicit
reason to be tracked.

---

## Disclaimer

This repository is a personal research project. Nothing here is financial advice. The certified
results above mostly say the *opposite* of "this prints money": the intraday models carry real
but **sub-cost** skill and therefore run in paper mode. Trading Indian equities intraday involves
substantial risk of loss; statutory costs alone (brokerage + STT + slippage) consume edges that
look large in gross backtests. Do your own validation before risking capital.
