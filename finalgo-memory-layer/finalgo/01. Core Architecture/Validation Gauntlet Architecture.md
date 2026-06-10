# 🛡️ Validation Gauntlet — Architecture & Implementation Plan

> **Status**: 📐 SPEC APPROVED FOR BUILD — not yet implemented
> **Architect**: Claude (Fable 5), 2026-06-10
> **Purpose**: One canonical, self-testing evaluation harness that becomes the **only source of truth** for any model metric in this repo. A model either passes the Gauntlet or it does not exist for deployment purposes.

---

## 1. Why This Exists (Design Motivation)

Every major model metric in this project has later been invalidated by a harness bug, never a model bug, and every correction moved the number **down**:

| Claimed result | Bug class that produced it | True result |
|---|---|---|
| v8 Spearman 0.046/0.049 | Early stopping evaluated on the test set | ρ≈0.025, decaying, net-negative |
| TBM Short +5.18 bps, PF 1.25 | Cost **added** instead of subtracted (one side only) | Net-negative |
| Sniper 68–76% WR | Same-bar IBS lookahead at evaluation time | 55–60% WR |
| 15m dual-TF "edge" | Label silently crossed the overnight boundary | Artifact, no intraday edge |
| Hybrid depth-4 +7.82 bps | Single favorable validation window | −2.5 bps under full WF |

**Root cause**: every evaluation was a bespoke script (29 files in `scripts/analysis/` alone), each a fresh opportunity for a leak or sign error. The Gauntlet replaces all of them as the *verdict authority*. Exploration scripts may still exist, but **no metric may be quoted in the vault, the registry, or the live engine without a Gauntlet run ID attached**.

The five bug classes above are converted into **automated assertions** (§5) so they can structurally never recur.

---

## 2. Top-Level Design

### 2.1 Repo placement

```
scripts/gauntlet/                  # the package (importable: from scripts.gauntlet import ...)
    __init__.py
    cli.py            # python -m scripts.gauntlet run|selftest|ledger
    contracts.py      # DatasetSpec, ModelSpec, GauntletConfig, Verdict dataclasses
    data_audit.py     # Stage 0: dataset integrity & label-construction audit
    leakage.py        # Stage 1: feature-timing & label-leak probes
    splits.py         # Stage 2: purged + embargoed walk-forward fold generator
    harness.py        # Stage 3: model-agnostic train/predict loop + adapters
    costs.py          # cost model + per-side sign invariants
    metrics.py        # Stage 4: Spearman, top-K bps/WR, decay, bootstrap CIs
    verdict.py        # Stage 5: pre-registered pass/fail engine (3-tier verdict)
    report.py         # JSON + Markdown reports, stamping, vault note emission
    registry.py       # writes gauntlet block into models/<name>/metadata.json
    synth.py          # synthetic panel generator for self-tests
tests/gauntlet/                    # meta-tests: the Gauntlet validating itself
    test_selftest_bug_classes.py
    test_splits.py
    test_costs.py
    test_synth_power.py
data/gauntlet/                     # run outputs (gitignored except ledger + reports)
    ledger.jsonl                   # append-only run ledger (tracked in git)
    <run_id>/report.json, report.md, preds.npz, config.lock.json
```

### 2.2 The pipeline (six stages, all mandatory)

```
DatasetSpec ─► [0] Data Audit ─► [1] Leakage Probes ─► [2] Fold Generation
                                                            │
ModelSpec ──────────────────────────────────────────► [3] WF Harness
                                                            │
                                                      [4] Metrics + Costs
                                                            │
GauntletConfig (pre-registered, hashed) ────────────► [5] Verdict + Report
```

A run **aborts hard** on any Stage 0–2 assertion failure. Stages 3–5 produce the stamped report. There is no code path that emits a metric without passing through all stages.

### 2.3 Determinism & provenance (non-negotiable)

Every run records, in `report.json` and the ledger:
- `run_id` = `{UTC timestamp}-{8-char config hash}`
- `dataset_sha256` (of the source file), `dataset_path`, row count
- `config_hash` = SHA256 of the **canonical-JSON GauntletConfig, computed and written to `config.lock.json` BEFORE any training starts** (pre-registration — see §7)
- `git_commit` of the repo, `seed` (default 42, single global), package versions (xgboost, numpy, pandas)
- `n_prior_runs` on the same dataset family from the ledger (multiple-testing context, §7)

Two runs with identical inputs must produce byte-identical metrics (`tree_method='hist'`, fixed seed, single-GPU determinism caveat documented; tolerance 1e-9 on CPU).

---

## 3. Contracts (`contracts.py`)

### 3.1 `DatasetSpec` — declares what the data IS, so the audit can verify it

```python
@dataclass(frozen=True)
class DatasetSpec:
    path: str                       # e.g. data/ranking_data_upstox_1h_v3_3y.csv
    label_col: str                  # e.g. "Next_Hour_Return"
    bar_minutes: int                # 60, 30, 15
    bar_label_side: str             # "left" (1h convention) | "right" (15m convention)  ← see vault note "Ranking data conventions"
    label_horizon_bars: int         # 1 for Next_Hour_Return
    label_may_cross_session: bool   # False for all intraday models (overnight guard)
    qid_col: str = "Query_ID"       # cross-sectional group = one timestamp
    ticker_col: str = "Ticker"
    datetime_col: str = "DateTime"
    session_close: str = "15:30"    # NSE
    raw_close_col: str | None = "Close"   # needed for label recomputation audit; None disables (audit then FAILS unless explicitly waived in config with a written reason)
```

### 3.2 `ModelSpec` — the only way a model enters the harness

```python
@dataclass(frozen=True)
class ModelSpec:
    name: str                       # e.g. "v8_upstox_3y_rerun"
    adapter: str                    # "xgb_ranker" | "xgb_binary" | "catboost" | "sklearn"
    params: dict                    # full hyperparameters
    features: list[str]             # exact feature list (order matters, hashed)
    sides: tuple[str, ...] = ("long", "short")   # ranker trains one model per side via label inversion
    num_boost_round: int = 500
    early_stopping_rounds: int = 50
```

Adapters implement two methods only: `fit(Xtr, ytr, qtr, Xva, yva, qva) -> model` and `predict(model, X) -> np.ndarray`. The harness owns everything else (NaN fill from train-only stats, ranking-label construction via `int_ranks`, group sizes, device detection — all lifted from the proven `v8_walkforward.py`).

### 3.3 `GauntletConfig` — pre-registered evaluation terms

```python
@dataclass(frozen=True)
class GauntletConfig:
    min_train_months: int = 18
    test_horizon_months: int = 2
    step_months: int = 4
    embargo_bars: int | None = None      # default: = label_horizon_bars
    costs_bps: tuple = (6.0, 10.0)       # 10 bps is the BINDING cost (Indian statutory reality); 6 is informational
    binding_cost_bps: float = 10.0
    top_k: tuple = (1, 3)
    recent_window_months: int = 12
    # verdict thresholds — see §6; changing any of these is a new pre-registration
    trigger_min_net_bps: float = 2.0
    trigger_min_t: float = 2.0
    filter_min_rho_p: float = 0.01
    filter_min_recent_z: float = 2.0
    seed: int = 42
    tod_diagnostic_only: bool = True     # time-of-day tables NEVER feed the verdict unless a ToD strategy was pre-registered
```

---

## 4. Stage Specifications

### Stage 0 — Data Audit (`data_audit.py`)

Hard assertions (abort on failure), each mapped to a historical bug:

| # | Check | Catches |
|---|---|---|
| A0.1 | Schema: all `ModelSpec.features` + label + qid + datetime + ticker present; no duplicate (ticker, datetime) rows | silent feature drift |
| A0.2 | Timestamps strictly increasing per ticker; `Query_ID` ↔ unique timestamp bijection; ≥5 tickers per query (matches `build_training_data.py` rule) | corrupted joins |
| A0.3 | **Bar-label-side verification**: infer convention from last-bar-of-day timestamps and assert it equals `spec.bar_label_side` | the 1h-left vs 15m-right misalignment class |
| A0.4 | **Label recomputation**: for a stratified sample (≥50 tickers × full span), recompute `label = Close.shift(-horizon)/Close − 1` per ticker and assert ≈ `label_col` (atol 1e-10) | mislabeled / stale label columns |
| A0.5 | **Overnight guard**: if `label_may_cross_session=False`, assert for every row the label's terminal bar falls in the same session date; rows violating it must not exist (not "be rare" — exist) | the 15m overnight-return artifact |
| A0.6 | NaN/Inf census: report per-feature NaN%; assert label has zero NaNs; assert no feature is >50% NaN | dead features |
| A0.7 | Dataset SHA256 computed & stamped; CSV optionally cached to Parquet in `data/gauntlet/_cache/` keyed by hash (the 1.5 GB CSVs make this near-mandatory for iteration speed) | provenance |

### Stage 1 — Leakage Probes (`leakage.py`)

| # | Probe | Mechanics | Catches |
|---|---|---|---|
| A1.1 | **Prefix invariance** | For ≥10 sampled tickers and ≥5 cut timestamps each: run the *actual feature pipeline* (`scripts/feature_utils.compute_features` + cross-sectional steps) on `data[:t]` vs `full_data`, assert all feature values at and before `t` are identical (atol 1e-9). EMA/rolling(past) pass; any centered window, future shift, or full-sample normalization fails. | same-bar / lookahead features (sniper IBS class) |
| A1.2 | **Within-query label shuffle** | Shuffle labels within each query on train+val, run 2 quick folds, assert mean OOS ρ ∈ [−0.005, +0.005] | features that encode the label (target leakage) |
| A1.3 | **Early-stopping disjointness** | Structural: harness API only accepts `(train, val, test)` index triples from `splits.py`, which asserts pairwise disjointness + embargo. There is no parameter to pass test data into `evals`. | the v8 bug class — made *impossible by construction*, then double-checked by an assertion |
| A1.4 | **Same-bar correlation screen** | Spearman of each feature vs the *current* bar's realized return; features above 0.95 flagged (warn, listed in report) | accidentally-included outcome columns |

A1.1 runs against the *feature pipeline*, so it must be runnable standalone (`python -m scripts.gauntlet leakage-audit --pipeline ranking_1h`) and re-run whenever `feature_utils.py` changes — wire a pytest in `tests/gauntlet/` so CI catches it.

### Stage 2 — Folds (`splits.py`)

Generalizes the proven scheme in [v8_walkforward.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/analysis/v8_walkforward.py#L100-L105):

- Calendar-month rolling folds: `train = months[:i]`, `val = months[i]` (early stopping only), `test = months[i+1 : i+1+horizon]`, stepping by `step_months`. Expanding window default; sliding window available as a config flag.
- **Purge**: drop the last `label_horizon_bars` bars of every train *and* val segment whose label window overlaps the next segment.
- **Embargo**: skip `embargo_bars` after each boundary (default = label horizon; cheap insurance).
- Emits `FoldPlan` objects; asserts: pairwise disjointness, chronological ordering, every test month covered at most once, ≥ `min_train_months` of training data.
- Unit-tested exhaustively in `tests/gauntlet/test_splits.py` (synthetic month grids, off-by-one boundary cases).

### Stage 3 — Harness (`harness.py`)

- Loads cached Parquet, materializes `X` (float64), `y`, `qids`, per the specs.
- Per fold: train-only NaN-fill means (exactly as `v8_walkforward.py:121-125`), build adapter, fit with early stopping on val, predict test, accumulate `(idx, pred_long, pred_short)`.
- GPU autodetection (lift `detect_device()`); fold loop optionally parallel over CPU workers when on CPU.
- Saves concatenated OOS predictions to `preds.npz` (same format as today's `walkforward_preds.npz` so existing analysis tooling keeps working).

### Stage 4 — Metrics & Costs (`metrics.py`, `costs.py`)

All computed per fold AND pooled, per side, per cost level, per K:

- Query-wise Spearman ρ (mean across queries) — long on `y`, short on `−y`.
- Top-K trade arrays → `raw_bps, net_bps, raw_win, net_win, t_stat, n` (lift `trade_stats`).
- **Cost invariants (the TBM guard), asserted on every trade array, both sides:**
  - `assert net_bps == raw_bps − cost_bps` (atol 1e-9)
  - `assert median(net − raw) == −cost` elementwise per trade
  - `assert (net < raw).all()` — cost can never help
- **Decay diagnostics**: OLS slope of fold-ρ and fold-net-bps vs fold index, with p-value; recent-window (last 12 mo) metrics computed separately.
- **Uncertainty**: stationary block bootstrap (per-query resampling, 1000 reps) → 95% CI on net bps; paired t-test of fold ρ vs 0.
- Time-of-day table — **diagnostic only** (§7), clearly watermarked in the report.

### Stage 5 — Verdict (`verdict.py`) — three tiers, not two

The pipeline doesn't need every model to be a standalone trader — the 1h layer's job is ranking/filtering. Encode that distinction:

| Verdict | Criteria (ALL must hold, at `binding_cost_bps`, per side independently) | Meaning |
|---|---|---|
| **`TRIGGER_GRADE`** | pooled OOS net ≥ `+2.0 bps/trade`, t ≥ 2.0; recent-12mo net > 0; decay slope not significantly negative (p<0.05 one-sided) | may *initiate* live trades |
| **`FILTER_GRADE`** | fails trigger, but fold-ρ paired-t p < 0.01 AND recent-12mo top-K **raw** WR z ≥ 2.0 | may rank/shortlist/veto only — never initiate |
| **`DEAD`** | neither | archive; metrics quarantined |

Each side (long/short) gets its own verdict — e.g. current evidence says v8-class models would land `short: FILTER_GRADE, long: DEAD`.

---

## 5. Self-Test Suite — the Gauntlet validating itself (`synth.py`, `tests/gauntlet/`)

**This is the heart of "self-testing": before the Gauntlet is trusted to judge models, pytest must prove it catches every historical bug class on synthetic data where ground truth is known.**

`synth.py` generates a panel (default 50 tickers × 3 years of 1h bars, GBM prices, realistic vol) with a **plantable signal**: `signal_feature = α · future_return + noise`, with α calibrated to target a chosen ρ.

| Test | Construction | Required outcome |
|---|---|---|
| T1 power | planted ρ = 0.05 | Gauntlet measures ρ within ±0.01 and net bps within bootstrap CI |
| T2 false-positive | pure noise features | verdict `DEAD`; pooled |net| bps within CI of −cost |
| T3 leak detection | inject feature = next-bar return (the leak) | Stage 1 A1.2/A1.4 must FAIL the run |
| T4 lookahead detection | feature built with centered rolling window | A1.1 prefix invariance must FAIL the run |
| T5 cost-sign tamper | monkeypatch cost application to `+cost` on one side | Stage 4 invariants must raise |
| T6 split tamper | force val month == test month | `splits.py` assertion must raise |
| T7 overnight tamper | relabel a sample with cross-session returns | A0.5 must FAIL the run |
| T8 regression baseline | run real v8 spec end-to-end | reproduces the known WF result (ρ≈0.026/0.024 avg, net-negative top-K) within tolerance — calibrates the Gauntlet against the one audit we trust |

T8 doubles as the acceptance test for the whole build: **if the Gauntlet reproduces the v8 reanalysis numbers from `data/model_analysis/v8_walkforward/walkforward.json`, it is correctly assembled.**

---

## 6. Registry & Live-Engine Integration (`registry.py`)

1. On completion, the Gauntlet writes into `models/<name>/metadata.json`:
   ```json
   "gauntlet": {
     "run_id": "20260611T093000Z-a1b2c3d4",
     "verdict": {"long": "DEAD", "short": "FILTER_GRADE"},
     "binding_cost_bps": 10.0,
     "dataset_sha256": "...", "config_hash": "...", "git_commit": "...",
     "evaluated_at": "2026-06-11"
   }
   ```
   Only `registry.py` may write this block (enforced by convention + a checksum over the block keyed to the run ledger).
2. **Live-engine guard**: `scripts/vanguard/model_inference.py` checks the block at model-load time:
   - trade-trigger role requires `TRIGGER_GRADE` on the used side;
   - ranker/filter role requires ≥ `FILTER_GRADE`;
   - missing/`DEAD` → **warn loudly in Phase A (warn-only rollout), hard-refuse in Phase B** (flip via `config.py` flag `GAUNTLET_ENFORCEMENT = "warn" | "enforce"`).
3. **Vault rule (add to `agent.md` protocol)**: any performance number written into the memory layer MUST cite its `run_id`. Numbers without one are marked `⚠️ UNVERIFIED`.

---

## 7. Multiple-Testing & Pre-Registration Discipline

This project has trained ~20 model generations on the same 3-year window — the trial count itself inflates false positives.

- `data/gauntlet/ledger.jsonl` is append-only; every run (including failures) is recorded with dataset family + config hash.
- The report prints `n_prior_runs` for the dataset family and a **deflated t-threshold** `t* = Φ⁻¹(1 − 0.025/N)` alongside the raw t-stat, so the verdict page always shows whether the result survives trial-count correction.
- `config.lock.json` is written before training; the verdict engine refuses to run if the live config hash ≠ locked hash (no moving goalposts after seeing results).
- Time-of-day, regime, and threshold sub-slices are **diagnostic watermarked** — using one as grounds for deployment requires a NEW pre-registered run where that slice is declared in the config up front (e.g. `pre_registered_slice: {"time": "14:30"}`) and evaluated on folds not previously used to discover it.

---

## 8. Phased Build Plan (for task dispersal to implementing agents)

Each phase is self-contained, has its own acceptance criteria, and can be assigned to a separate agent. Dependencies are strictly downward.

| Phase | Deliverables | Key source to lift from | Acceptance criteria | Est. effort |
|---|---|---|---|---|
| **P0 — Skeleton & contracts** | `contracts.py`, `cli.py` stub, run-folder creation, config canonical-JSON hashing, ledger append, dataset SHA256 + Parquet cache | — | `python -m scripts.gauntlet run --dry-run` produces run folder + `config.lock.json`; hash stable across runs; pytest for hash canonicalization | 0.5 day |
| **P1 — Data audit** | `data_audit.py` (A0.1–A0.7) | `build_training_data.py` (label & query construction) | All assertions pass on `ranking_data_upstox_1h_v3_3y.csv`; A0.5 verified to FAIL on a deliberately overnight-labeled sample | 1 day |
| **P2 — Splits + harness + adapters** | `splits.py`, `harness.py`, adapters (`xgb_ranker` first; `xgb_binary`, `catboost` after) | `v8_walkforward.py` lines 54–175 (int_ranks, group_sizes, NaN fill, fold loop, device detect) | `tests/gauntlet/test_splits.py` green; end-to-end run on v8 spec completes and writes `preds.npz` | 1.5 days |
| **P3 — Metrics, costs, verdict, report** | `costs.py`, `metrics.py`, `verdict.py`, `report.py` (JSON+MD) | `v8_walkforward.py` `trade_stats` + top-K logic | **T8 regression**: reproduces v8 WF numbers (fold ρ table, top-K net bps) within ±0.002 ρ / ±0.3 bps; cost invariants demonstrably trip under T5 tamper | 1 day |
| **P4 — Leakage probes** | `leakage.py` (A1.1–A1.4) standalone CLI + harness wiring | `feature_utils.py` (the pipeline under test) | A1.1 passes on current `compute_features`; T3/T4 synthetic tests trip correctly | 1 day |
| **P5 — Self-test suite** | `synth.py`, `tests/gauntlet/test_selftest_bug_classes.py` (T1–T7) | — | Full pytest matrix green; T1 power & T2 false-positive within stated tolerances | 1 day |
| **P6 — Registry + live guard + docs** | `registry.py`, `model_inference.py` warn-only guard, `agent.md` vault-rule amendment, README section | `scripts/vanguard/model_inference.py` | Stamped block appears in a test model's metadata; live engine logs warning for unstamped models; protocol updated | 0.5 day |
| **P7 — Re-baseline campaign** | Gauntlet runs for `v8_upstox_3y`, `v10_native_1h`/depth4, `v2_15min_3y`, `daily_xgb`; vault notes per run | — | Every active model has a verdict block; `v2_15min_3y` (last unaudited) gets its first honest WF verdict; `Model Performance & Statistics.md` updated with run IDs; failures archived to `05. Archives/` | 1–2 days |

**Build order is strict: P0 → P1 → P2 → P3 → (P4 ∥ P5) → P6 → P7.** Total ≈ 7–8 agent-days.

### Hard constraints for all implementing agents
1. **Do not invent new evaluation math** — lift the proven logic from `v8_walkforward.py` and generalize it. Novelty is a bug source; this project's history proves it.
2. Python 3.12, existing pinned deps only (xgboost, numpy, pandas, scipy; catboost already present). No new heavy dependencies.
3. Every assertion must have a pytest that proves it can fire (a guard that never fires in tests is assumed broken).
4. 15m dataset is 5.3 GB CSV — all loaders must go through the Parquet cache; never `pd.read_csv` the raw file twice in one run.
5. Windows host: paths via `os.path`/`pathlib`, no POSIX-isms; long runs must print fold-level progress (the user watches consoles).
6. The Gauntlet never mutates source datasets, model artifacts (other than the metadata stamp), or anything in `scripts/vanguard/` beyond the P6 guard.

---

## 9. Out of Scope (explicitly)

- New features / new data sources (options OI, depth, sentiment-as-feature) — that is the *next* project, and it must arrive **after** the Gauntlet exists so the first new-data model is judged honestly from day one.
- Backtest of multi-bar holding strategies / portfolio sizing — the Gauntlet judges per-bar signal quality; strategy-level simulation stays in `scripts/backtests/` (those scripts should later cite Gauntlet-passed models only).
- Deleting the 29 `scripts/analysis/` scripts — they remain as exploration tools but lose verdict authority the moment P6 lands.

---

## 🔗 Backlinks
- [[02. Model Suite/Model Performance & Statistics]] — the metrics this replaces as authority
- [[06. Context & Logs/Conversations/Conv-2026-06-10-V8-Walkforward-Reanalysis]] — the audit that motivated this
- [[06. Context & Logs/Conversations/Conv-2026-06-10-Signal-Layer-Rectification-Plan]] — the parent rectification strategy
- [[02. Model Suite/Training Data & Regime Requirements]] — 3-year mandate enforced via `min_train_months`
