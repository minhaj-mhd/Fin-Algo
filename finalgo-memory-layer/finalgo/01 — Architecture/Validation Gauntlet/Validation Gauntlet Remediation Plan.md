# 🔧 Validation Gauntlet — Remediation Plan (R1–R8)

> **Status**: 📐 SPEC APPROVED FOR BUILD — disperse to implementing agents
> **Architect**: Claude (Fable 5), 2026-06-10
> **Parent spec**: [[01 — Architecture/Validation Gauntlet/Validation Gauntlet Architecture]] — read it FIRST; all conventions there still bind.
> **Trigger**: Architecture-conformance audit of the P0–P7 build (see [[06. Context & Logs/Conversations/Conv-2026-06-10-Signal-Layer-Rectification-Plan|audit log]]). Core harness verified correct (reproduces trusted v8 WF: ρ 0.0253/0.0243 vs 0.0261/0.0245; short Top-3 net within 0.02 bps). Six critical + nine moderate gaps must be closed before any verdict is trusted or `GAUNTLET_ENFORCEMENT` is flipped to `"enforce"`.

---

## 0. Ground Rules for All Implementing Agents

1. **Never weaken an existing assertion** to make a fix pass. If a fix makes a real dataset fail an audit, that is a *finding* — stop and report it; do not patch around it.
2. Any change to `GauntletConfig` fields changes the canonical config hash. That is intended: post-remediation runs are NEW pre-registrations. Do not try to keep the old hash.
3. Every new guard ships with a pytest that proves it **fires** (negative test) and a pytest that proves it **passes on clean input** (positive test). A guard without a firing test is treated as broken.
4. Keep `preds.npz`/report formats compatible with `scripts/analysis/` tooling (same field names as `data/model_analysis/v8_walkforward/walkforward_preds.npz`: `idx, ym, q, y, time, rl, rs`).
5. Python 3.12, existing pinned deps only. Windows paths via `os.path`/`pathlib`.
6. The current four registry stamps (v8, v10, v2_15min, daily_xgb) are **provisional** — do not build anything on top of them; R8 reissues them.
7. Work package order: **R1–R7 may run in parallel** (disjoint files where possible; R3 and R4 both touch `cli.py` — coordinate or serialize). **R8 strictly last.**

---

## R1 — Wire the Prefix-Invariance Probe (A1.1) Into Every Run
**Severity**: CRITICAL — the sniper-IBS lookahead bug class is currently uncaught.
**Files**: `scripts/gauntlet/leakage.py`, `scripts/gauntlet/cli.py`, `scripts/gauntlet/contracts.py`, `tests/gauntlet/`

### Problem
`check_prefix_invariance` exists but is never called in `run_gauntlet` ([cli.py L121–124](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/gauntlet/cli.py) wires only A1.4 + A1.2). Its sole invocation is test T4, which **mocks** `compute_features` — the real 902-line `scripts/feature_utils.py` pipeline has never been prefix-tested. Additionally, the probe swallows exceptions (`except: continue`) so a pipeline that crashes on sliced input silently "passes", and `compute_features` errors skip the whole ticker.

### Implementation
1. **Add to `DatasetSpec`**: `feature_pipeline: Optional[str] = "ranking_v3"` — names a registered pipeline callable. Maintain a registry in `leakage.py`: `{"ranking_v3": lambda df: compute_features(df, legacy=False)}`. If `feature_pipeline=None`, A1.1 is skipped **only with** a new explicit field `prefix_invariance_waiver_reason: Optional[str]` (must be non-empty; waiver and reason printed in the report and stamped in `report.json`).
2. **Wire into `run_gauntlet`** immediately after Stage 0, before A1.4: select ≥10 tickers (deterministic: seeded RNG from `config.seed`, stratified by data span) from the loaded dataset; the dataset must contain the raw OHLCV columns the pipeline needs (`Open, High, Low, Close, Volume`) — if missing, **hard-fail** unless waived per (1).
3. **Remove all exception swallowing**: any exception raised by the pipeline on full OR sliced data is an audit FAILURE (`raise` wrapped with context), not a skip. A ticker skipped for insufficient length (<50 bars) is fine but must be *counted*; assert ≥5 tickers were actually checked.
4. **Refactor for testability**: `check_prefix_invariance(raw_df, tickers, pipeline_fn, n_cuts=5)` — take the pipeline as a **parameter** (no module-level import binding), so tests inject leaky pipelines without `unittest.mock.patch`.
5. **Standalone CLI**: `python -m scripts.gauntlet leakage-audit --dataset 1h_v3_3y` runs A1.1 only (for re-running whenever `feature_utils.py` changes).
6. **Comparison detail**: compare ALL numeric feature columns at every index ≤ cut, `atol=1e-9, equal_nan=True` (as now), but also assert the sliced output's index tail actually reaches `cut_dt` (guards against a pipeline that silently truncates).

### Tests (rewrite T4 + add)
- `test_t4_lookahead_detection`: inject `pipeline_fn` returning `Close.shift(-1)` → must raise `AssertionError` with "Prefix invariance". **No mock.patch.**
- `test_t4b_centered_window`: pipeline with `Close.rolling(5, center=True).mean()` → must raise.
- `test_a11_real_pipeline_clean`: run A1.1 against the **actual** `scripts.feature_utils.compute_features` on a synthetic OHLCV panel from `synth.py` (extend synth to emit Open/High/Low/Volume — see R2 item 4) → must pass. *This is the first-ever lookahead certification of the production feature pipeline; if it FAILS, stop everything and report (that would mean live features are leaky).*
- `test_a11_exception_is_failure`: pipeline that raises on sliced input → audit must fail, not skip.

### Acceptance
A1.1 executes inside `run` for the 1h dataset (or aborts with a documented missing-OHLCV waiver decision); all four tests green; `leakage-audit` subcommand works.

---

## R2 — Rebuild the Overnight/Label-Integrity Guard (A0.4 + A0.5)
**Severity**: CRITICAL — the guard certifies what it never checked; ~1/6 of 1h rows have in-file-unverifiable labels.
**Files**: `scripts/gauntlet/data_audit.py`, `scripts/gauntlet/contracts.py`, `scripts/gauntlet/synth.py`, `tests/gauntlet/`

### Problem
A0.5 checks `dt + bar_minutes` calendar arithmetic — it never looks at the **actual next bar** the label was computed from. A0.4 recomputes labels only `if target_dt in close_map`, silently skipping every row whose target bar is absent. Empirical audit finding on `ranking_data_upstox_1h_v3_3y`: bar times are 09:15–13:15 only; **all 1,083 last-bar-of-day (13:15) rows per ticker have labels whose 14:15 target bar is not in the file** — unverifiable, yet certified. (They are *not* naive overnight returns — only 2/1083 match next-day close — so they were likely computed from a pre-drop 14:15 close. But "probably fine" is exactly what this Gauntlet exists to eliminate.)

### Implementation
1. **Row classification** (vectorized, per ticker, no `iterrows`): compute `next_row_dt = DateTime.shift(-1)`; classify every labeled row as:
   - `INTRA`: target bar (`dt + horizon*bar_minutes`) exists in-file → recompute label, assert match (`atol=1e-9`).
   - `UNVERIFIABLE`: target bar absent, next row is same session → recompute impossible in-file.
   - `BOUNDARY`: next row is a different session date.
2. **Coverage accounting**: report `pct_verified`, `pct_unverifiable`, `pct_boundary` per dataset; stamp all three into `report.json` and the Markdown report.
3. **Anti-overnight statistical check** (the direct kill-shot for the 15m bug class): for every `BOUNDARY` and `UNVERIFIABLE`-at-day-end row, compute the **overnight return** (next session's first in-file Close ÷ current Close − 1) and the match rate `|label − overnight| < 1e-9`. **Assert match rate < 1%** (chance-level collisions only). If a dataset's labels ARE overnight returns, this fires on ~100% of boundary rows.
4. **Raw-source verification (primary path for UNVERIFIABLE rows)**: add `raw_source_glob: Optional[str]` to `DatasetSpec` (e.g., `"data/historical/*.parquet"`). When provided, for a seeded sample of ≥500 UNVERIFIABLE rows, locate the true next-hour close in the raw source and assert the label matches. When not provided, require `unverified_label_waiver_reason: Optional[str]` (non-empty) — waiver + `pct_unverifiable` go into the report and the registry stamp.
5. **Hard gate**: run aborts if `pct_verified < 0.80` AND no raw-source verification AND no waiver. (The 1h dataset sits at ~83% verified — passes the floor but must carry either raw verification or a waiver.)
6. **Replace the old A0.5 time-arithmetic check** with: every `INTRA` row's target bar must be in the same session (this is now exact, not arithmetic), keep the `terminal time ≤ session_close` assert for declared conventions.
7. **Extend `synth.py`**: add `plant_overnight_labels: bool = False` — when True, last-bar-of-day labels are set to the true next-day return (the bug). Also emit `Open/High/Low/Volume` columns (needed by R1's real-pipeline test): OHLC derived from the GBM path with small intra-bar noise, Volume lognormal.
8. **Performance**: all checks vectorized with groupby/shift; full-dataset A0.4 (not 50-ticker sample) is now feasible — do it. The 5.3 GB 15m dataset must stay within memory via column-subset loading from the Parquet cache.

### Tests
- `test_t7_overnight_real` (replaces T7): synth panel with `plant_overnight_labels=True` → audit must FAIL on the anti-overnight check. Clean panel → pass.
- `test_a04_coverage_reported`: synth with deliberately dropped target bars → `pct_unverifiable` correct, gate behavior correct (abort without waiver, pass with waiver, pass with raw-source glob).
- `test_a04_full_recompute`: corrupt one INTRA label by 1e-4 → audit must FAIL.

### Acceptance
Re-running the 1h dataset audit prints coverage stats, executes the anti-overnight check on all boundary rows, and either verifies UNVERIFIABLE labels against `data/historical/*.parquet` or carries an explicit waiver. All three tests green.

---

## R3 — Fix the Verdict Engine (FILTER_GRADE Over-Leniency)
**Severity**: CRITICAL — v8 LONG was stamped FILTER_GRADE, contradicting the trusted raw-signal audit (longs weak/decayed; only shorts real).
**Files**: `scripts/gauntlet/verdict.py`, `scripts/gauntlet/cli.py`, `scripts/gauntlet/contracts.py`, `tests/gauntlet/`

### Problem
Three compounding leniencies in `verdict.py` / `cli.py`:
(a) the win-rate z-test baseline is a fixed **0.5** — in a drifting market the *universe* long WR exceeds 0.5, so top-K longs clear z≥2 without any selection skill;
(b) `get_best_verdicts` takes the **best grade across K∈{1,3}** — cherry-picking;
(c) the "recent 12mo" mask `ym >= f"{year-1}-{month:02d}"` spans **13 months**.

### Implementation
1. **Universe-baseline z-test**: baseline per side per period = raw WR of **all** OOS candidate rows in that period (long: `(y > 0).mean()` over recent OOS rows; short: `(−y > 0).mean()`). Then
   `z = (topK_wr − baseline_wr) / sqrt(baseline_wr·(1−baseline_wr)/n_topK)`.
   This matches the methodology of the trusted Raw-Signal-Only Audit (see [[02 — Models/_Shared/Model Performance & Statistics]]). Compute baseline inside `cli.py` from `harness_res` and pass it through; `evaluate_verdict` takes `baseline_wr` as an argument — no hardcoded 0.5 anywhere.
2. **Kill best-of-K**: add `primary_k: int = 3` to `GauntletConfig` (pre-registered). The verdict is computed **only** at `primary_k` and `binding_cost_bps`. All other K/cost cells remain in the report as diagnostics, watermarked like ToD. Delete `get_best_verdicts`'s max-grade logic; rename to `compute_verdict(side, ...)`.
3. **True 12-month window**: `recent_months = sorted(unique OOS months)[-config.recent_window_months:]`; mask = `isin(recent_months)`. No string arithmetic.
4. **Decay input**: decay slope/p currently computed on fold ρ only — additionally compute on per-fold Top-`primary_k` net bps and require *neither* to be significantly negative for TRIGGER_GRADE (spec intent: "decay slope not significantly negative" on performance, not just rank correlation).
5. **Report**: print baseline WR next to every WR cell so leniency is visible at a glance.

### Tests
- `test_verdict_universe_baseline`: synthetic returns with universe drift (all-y mean > 0, no selection skill: top-K WR == baseline) → z≈0 → side must be DEAD even though WR > 0.5.
- `test_verdict_no_best_of_k`: stats where K=1 would pass and K=3 (primary) fails → verdict reflects primary only.
- `test_recent_window_exact`: 30 months of fold output → recent mask covers exactly 12 distinct months.
- `test_verdict_filter_real_skill`: top-K WR = baseline + 5pp, n large → FILTER_GRADE.

### Acceptance
All four tests green. **Expected outcome on re-run (R8): v8 long flips to DEAD, v8 short likely retains FILTER_GRADE** — consistent with the trusted audit (short z=5.4 vs universe baseline). If v8 long *still* grades FILTER under the universe baseline, halt and report the evidence rather than accepting silently.

---

## R4 — True Pre-Registration (Lock Config BEFORE Training)
**Severity**: CRITICAL — "no moving goalposts" is currently theater; the lock file is written after results exist.
**Files**: `scripts/gauntlet/cli.py`, `scripts/gauntlet/report.py`, `tests/gauntlet/`

### Implementation
1. **At the very top of `run_gauntlet`** (before Stage 0 data loading): mint `run_id`, create the run directory, write `config.lock.json` (canonical dict + its hash + dataset path + model name + UTC timestamp), and append a `{"event": "started", ...}` record to the ledger.
2. **At verdict time (Stage 5 entry)**: recompute the canonical hash of the live in-memory `GauntletConfig` and re-read `config.lock.json`; **abort with a hard error if either hash mismatches** the locked one. This makes mid-run config mutation (human or agent) structurally impossible to slip through.
3. `save_report` no longer writes the lock file (it already exists); it appends the `{"event": "completed", ...}` ledger record. `verdicts` only ever appear in `completed` records.
4. **Ledger schema**: add `"event": "started" | "completed"` to every record; `n_prior_runs` counts **started** events for the dataset family (a run you started and abandoned because results looked bad still counts as a trial — that's the whole point of trial-count deflation).
5. Keep `--dry-run` behavior (lock-only) but route it through the same code path as step 1.

### Tests
- `test_r4_lock_before_train`: run on a tiny synth dataset with the harness monkeypatched to assert `config.lock.json` already exists when `run_harness` is entered.
- `test_r4_tamper_aborts`: mutate the config object (e.g., `object.__setattr__` around frozen dataclass, or patch the hash function input) after lock, before verdict → run must abort.
- `test_r4_ledger_started_counted`: a started-but-crashed run increments `n_prior_runs` for the next run.

### Acceptance
All three tests green; a normal run produces `started` + `completed` ledger records with matching `run_id`.

---

## R5 — Save `preds.npz` + Encode the T8 Regression Test
**Severity**: CRITICAL (T8 was the build's named acceptance criterion; npz is the reproducibility artifact).
**Files**: `scripts/gauntlet/cli.py` or `report.py`, `tests/gauntlet/test_t8_regression.py`

### Implementation
1. After `run_harness`, save `<run_dir>/preds.npz` (compressed) with fields exactly: `idx, ym, q, y, time` plus one array per side named `rl` (long) / `rs` (short) for compatibility with existing `scripts/analysis/` consumers (ground rule 4). For single-side models, save only the present side.
2. **T8 test** (`@pytest.mark.t8` + registered in `cli.py selftest --full`): runs the v8 `ModelSpec` (from `models/v8_upstox_3y/metadata.json`) through the full Gauntlet on the registered `1h_v3_3y` dataset and asserts against **hardcoded reference values** (do not read the trusted JSON at test time; freeze the numbers into the test with a comment citing `data/model_analysis/v8_walkforward/walkforward.json`):
   - mean fold ρ long ∈ 0.0261 ± 0.004, short ∈ 0.0245 ± 0.004
   - full-OOS Top-3 @6bps net: long −3.6 ± 1.0 bps... **use the gauntlet's own audited first-run values as the frozen baseline instead** (long −1.71, short −3.74; Top-1 long −1.0, short −2.95) with ±0.5 bps tolerance — rationale: the gauntlet's embargo/purge legitimately differs from the legacy script by ~2 bps on the long side; what T8 must catch is *future drift of the gauntlet itself*, so the baseline is the verified-correct gauntlet run `20260610T074638Z-c7de73f9`, which was manually validated against the trusted audit during the conformance review.
   - fold count == 9.
3. **Runtime control**: ~4 min on CUDA per the ledger timestamps; mark `t8` excluded from the default pytest run, included in `python -m scripts.gauntlet selftest --full` and required before any registry re-stamp (document in AGENTS.md rule, R6).
4. T8 must use a **throwaway ledger/run root** (depends on R6's path indirection) so regression runs don't pollute trial counts.

### Tests / Acceptance
`pytest -m t8` green on this machine; `preds.npz` present in new run dirs and loadable by `numpy.load`; field names verified by a unit test.

---

## R6 — Ledger Hygiene, Git Tracking, Protocol Rule
**Severity**: MODERATE but cheap and foundational (R5's T8 depends on path indirection).
**Files**: `scripts/gauntlet/` (new `paths.py`), `tests/gauntlet/conftest.py`, `.gitignore`, `AGENTS.md`, ledger cleanup

### Implementation
1. **Path indirection**: new `scripts/gauntlet/paths.py` exposing `gauntlet_root()` → `os.environ.get("GAUNTLET_ROOT", "data/gauntlet")`; every module (cli, report, registry, data_audit cache) resolves paths through it. `tests/gauntlet/conftest.py` gets an autouse fixture setting `GAUNTLET_ROOT` to `tmp_path` — **tests can never touch the production ledger again**.
2. **Clean existing pollution**: rewrite `data/gauntlet/ledger.jsonl` keeping only the four real model runs (drop the `dummy_dataset_sha256` skeleton record and all six `run_t1`/`run_t2` records); delete `data/gauntlet/run_t1/`, `run_t2/` directories. Record the cleanup in the daily log.
3. **`.gitignore`**: add
   ```
   data/gauntlet/_cache/
   data/gauntlet/*/preds.npz
   ```
   then `git add data/gauntlet/ledger.jsonl data/gauntlet/*/report.json data/gauntlet/*/report.md data/gauntlet/*/config.lock.json` so verdict provenance is version-controlled. Verify `git status` shows no parquet.
4. **AGENTS.md amendment** (P6 debt): add a short section "🛡️ Model Metric Discipline": (a) any performance metric written to the vault/registry/README MUST cite a Gauntlet `run_id`; metrics without one are labeled `⚠️ UNVERIFIED`; (b) `scripts/analysis/` outputs are exploratory only — no verdict authority; (c) registry stamps may only be written by `scripts/gauntlet/registry.py`; (d) `GAUNTLET_ENFORCEMENT` lives in `scripts/vanguard/config.py` and may only be flipped to `"enforce"` after the R8 re-baseline. Mirror the same text into the vault copy of the protocol if one exists.
5. **Registry checksum**: switch `compute_stamp_checksum` from MD5 to SHA-256 (`registry.py`); re-stamping in R8 regenerates all checksums, so no migration needed — but `verify_model_stamp` must reject old-format stamps cleanly ("stale pre-remediation stamp" reason), which also automatically invalidates the four provisional stamps until R8.

### Tests / Acceptance
`pytest tests/gauntlet -q` leaves `data/gauntlet/` byte-identical (assert via before/after hash in a meta-test or manual check); ledger contains exactly 4 model records (until R8 adds more); `git status` clean of cache files; AGENTS.md section present.

---

## R7 — Robustness Bundle (Moderate/Minor Fixes)
**Severity**: MODERATE. Independent small fixes; one agent can take the whole bundle.
**Files**: `scripts/gauntlet/data_audit.py`, `splits.py`, `harness.py`, `cli.py`, `tests/gauntlet/`

1. **A0.3 bar-side inference**: replace the global `max(time)` heuristic with a per-day vote: for each session date, take the last bar's time; require ≥99% of days to agree on the inferred convention; ambiguity = audit failure listing offending dates. (One anomalous timestamp can no longer flip the verdict.)
2. **T6 made real**: extract `validate_fold_plan(train_idx, val_idx, test_idx, qids)` as a public function in `splits.py` containing the disjointness/chronology asserts; `generate_folds` calls it. Rewrite `test_t6_split_tamper` to call `validate_fold_plan` with genuinely overlapping index arrays — delete the `mock.patch("numpy.intersect1d")` hack.
3. **CatBoost seed**: adapter must take seed from `config.seed` (thread it through `run_harness` → `adapter.fit(..., seed=config.seed)`), not hardcoded 42.
4. **`xgb_binary` threshold**: replace hardcoded `0.0020` with `ModelSpec.params.get("binary_threshold", 0.0020)`; document in the adapter docstring; threshold value echoed in `report.json` model section.
5. **CLI custom-dataset fallback removed**: unknown `--dataset` values no longer guess a `DatasetSpec` from the filename. Accept either a registered name or `--dataset-spec path/to/spec.json` (full DatasetSpec as JSON, which then participates in config-lock hashing). Guessing defeats pre-registration.
6. **A0.4 tolerance**: tighten label recompute to `atol=1e-9` (from 1e-6) — document as a deliberate deviation from the spec's 1e-10 (CSV float roundtrip).
7. **Ledger reader hardening**: `print_ledger` / `n_prior_runs` must not silently `except: pass` on corrupt lines — print a warning with line number (a corrupt ledger silently undercounting trials would quietly weaken deflation).
8. **Device probe caching**: `detect_device()` is called per adapter `fit` (twice per fold) — cache the result at module level (`functools.lru_cache`). Pure waste otherwise.

### Tests / Acceptance
Each item with behavior change gets a focused unit test (1, 2, 4, 5, 7 at minimum). Full `pytest tests/gauntlet` green.

---

## R8 — Re-Baseline Campaign (STRICTLY LAST)
**Severity**: the payoff step. **Blocked on R1–R7 all merged and `pytest tests/gauntlet` + `selftest --full` (incl. T8) green.**
**Files**: none new — runs + vault/doc updates

1. Re-run the Gauntlet for `v8_upstox_3y`, `v10_native_1h`, `v2_15min_3y` (on `15min_3y_clean`), `daily_xgb` (on `daily_5y`). New run IDs, new SHA-256 stamps (old MD5 stamps auto-invalidated per R6.5).
2. **Expected-vs-actual check** (do not skip): v8 long → expected DEAD; v8/v10 short → expected FILTER_GRADE; any TRIGGER_GRADE anywhere is a red flag demanding manual review before celebrating. Compare every verdict against [[02 — Models/_Shared/Model Performance & Statistics]] expectations; discrepancies are findings to report, not noise.
3. Record per-run: coverage stats from R2 (verified/unverifiable/boundary %), waivers used, baseline WRs from R3.
4. **Vault updates**: `Model Performance & Statistics.md` gets a "Gauntlet Verdicts (post-remediation)" table with run IDs; old provisional verdicts marked superseded; Current Context Active Focus updated; conversation note concluded into the daily log per protocol.
5. Only after the user reviews R8 results may `GAUNTLET_ENFORCEMENT` be discussed for `"enforce"` — flipping it is a **user decision**, not an agent action.

---

## Dispersal Summary

| Pkg | Touches | Depends on | Est. effort | Parallel-safe |
|---|---|---|---|---|
| R1 prefix-invariance wiring | leakage, cli, contracts, synth(OHLCV via R2.7 — coordinate), tests | — | 1 day | ⚠️ shares synth with R2 |
| R2 label-integrity rebuild | data_audit, contracts, synth, tests | — | 1 day | ⚠️ shares synth with R1 |
| R3 verdict engine | verdict, cli, contracts, tests | — | 0.5 day | ⚠️ shares cli with R4 |
| R4 pre-registration | cli, report, tests | — | 0.5 day | ⚠️ shares cli with R3 |
| R5 preds.npz + T8 | cli/report, tests | R6 (path indirection) | 0.5 day | ✅ |
| R6 hygiene + git + protocol | paths(new), conftest, .gitignore, AGENTS.md, registry | — | 0.5 day | ✅ |
| R7 robustness bundle | data_audit, splits, harness, cli, tests | — | 0.5–1 day | ⚠️ wide but shallow |
| R8 re-baseline | runs + vault | **R1–R7 complete** | 0.5 day | ❌ last |

Practical sequencing for agents: **(R6 → R5)** ∥ **(R2 → R1)** ∥ **(R3 + R4 as one cli-owning task)** ∥ **R7**, then **R8**. Total ≈ 4–5 agent-days.

### Definition of Done (whole plan)
1. `pytest tests/gauntlet` green; `python -m scripts.gauntlet selftest --full` (incl. T8) green.
2. Production ledger contains only real model runs; tests provably cannot write to it.
3. The 1h dataset run reports label coverage and passes the anti-overnight check (or carries an explicit, visible waiver).
4. All four models re-stamped with SHA-256 checksums and post-remediation verdicts; expected-vs-actual table reviewed.
5. AGENTS.md carries the metric-discipline rule; `.gitignore` keeps the 2.7 GB cache out of git while ledger + reports are tracked.
6. `GAUNTLET_ENFORCEMENT` still `"warn"` — flip is a user decision after reviewing R8.

---

## 🔗 Backlinks
- Parent spec: [[01 — Architecture/Validation Gauntlet/Validation Gauntlet Architecture]]
- Audit findings: [[06. Context & Logs/Conversations/Conv-2026-06-10-Signal-Layer-Rectification-Plan]]
- Verdict expectations source: [[02 — Models/_Shared/Model Performance & Statistics]]
