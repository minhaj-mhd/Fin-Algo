# 🏛️ Daily Gatekeeper V2 — Rebuild & Certification Plan (D0–D6)

> **Status**: 📐 SPEC APPROVED FOR BUILD — disperse to implementing agents
> **Architect**: Claude (Fable 5), 2026-06-10
> **Prereqs**: [[01. Core Architecture/Validation Gauntlet Architecture]] and [[01. Core Architecture/Validation Gauntlet Remediation Plan]] — all Gauntlet conventions bind. Read both before starting.
> **Decision (user, 2026-06-10)**: Build a better daily model FIRST (macro/cross-asset features, longer label), then test it as a gatekeeper. The uplift test on the current edge-less model was rejected as noise-measurement.

---

## 1. Why V2 Exists — the Evidence

`daily_xgb` (165 single-name price/volume features, 1-day label) was stamped **DEAD/DEAD** by Gauntlet run `20260610T102743Z-5f7d069f`, but the autopsy shows a *fixable* failure, not a hopeless one:

- **Rank signal is real and NOT decaying**: fold-ρ ≈ +0.021/+0.020 over 11 folds (p < 0.02 both sides); recent folds among the strongest (fold 10: L +0.054, S +0.042).
- **It failed only the recent-12mo WR z-test** (long z=0.40, short z=1.00, need ≥2.0), for two structural reasons:
  1. **Sample starvation**: 12 months of daily = n=720 top-3 trades → needs a +3.7pp WR edge for z=2 (intraday models were tested at n=3,702–15,759).
  2. **Signal-shape mismatch**: the daily edge is *magnitude*-shaped, not hit-rate-shaped — recent short was **+13.2 bps raw / +3.2 bps net (post-cost POSITIVE)** at only 51.2% WR. A WR test is structurally blind to tail-driven edges.
- **All 165 features are single-name price/volume.** The daily horizon is exactly where cross-asset/macro information (VIX, overnight global moves, breadth, currency) is documented to carry signal — and none of it is in the model.

**V2 thesis**: add the macro/cross-asset information block + a multi-day label (regime questions are multi-day), and fix the two criteria mismatches via a universal, pre-registered Criteria-v2 — then certify once.

---

## 2. Work Packages

### D0 — Decision-Time Contract & Feature-Availability Table
**Severity**: foundational — the #1 leakage risk in cross-asset daily models is the *join*, not the series (A1.1 cannot catch a same-day-close join).
**Deliverable**: `finalgo-memory-layer/finalgo/02. Model Suite/Gatekeeper V2 Feature Availability.md` + enforced in the builder.

1. **Decision timestamp**: pre-open **09:00 IST on trade day T**. Every feature value in row T must be knowable strictly before then.
2. Per-feature availability table (feature | source | source timestamp | available-at IST | lag applied). Anchor cases:
   - India equities/indices/VIX: data through **T−1 15:30 close** → lag 0 vs T−1 close. ✅
   - US close (S&P/Nasdaq/DXY/10Y): the "T−1" US session closes ~01:30–02:30 IST *on calendar day T* → available at 09:00 IST T. ✅ Label it by IST availability date in the join, not by US calendar date.
   - Asia morning (Nikkei/HSI): mid-session at 09:00 IST — **use T−1 closes only** in V2 (no partial-session values).
   - FII/DII flows (if included): published evening T−1 → join with 1-day lag minimum. Optional in V2; skip if scraping is friction.
3. **Row indexing convention**: row for trade day T carries `DateTime = T−1` (the last India close used). This keeps the dataset Gauntlet-native (see D2 label).

### D1 — Collectors
**Files**: `scripts/collectors/collect_upstox_daily_10y.py`, `scripts/collectors/collect_global_daily.py`

1. **Upstox daily**: full universe (~172 names) + NSE indices (Nifty 50, Nifty 500, Bank, IT, Auto, Pharma, Metal, FMCG, Energy, Realty) + India VIX. One request per instrument (10y/request supported — confirmed). Store per-instrument parquet under `data/raw_daily_10y/` (these become the `raw_source_glob` for Gauntlet label verification — layout `{YYYY-MM-DD}.parquet` per day OR adapt the glob verifier to per-instrument files; pick one and keep the Gauntlet's raw-source check working).
2. **Global daily** via yfinance: `^GSPC, ^IXIC, ^N225, ^HSI, USDINR=X, BZ=F, GC=F, DX-Y.NYB, ^TNX` — **always pass `timeout=15`** (the Pulse-Scan hang lesson) and retry once.
3. Idempotent + incremental (re-run appends missing days only) — these collectors become part of the daily live refresh.
4. Collect 10 years; training window is a config knob (default **7 years** — survivorship-bias compromise, documented in the dataset spec).

### D2 — Dataset Builder
**File**: `scripts/training/build_daily_macro_dataset.py` → `data/ranking_data_daily_macro_v2.csv`

1. **Single-name block (lean)**: ~40–60 features from the existing `feature_utils` daily-applicable set (returns, distances to SMAs, RSI, vol z-scores, 52w distances, streaks). Do NOT port all 165 blindly — depth-5 trees + 165 noisy features is part of why V1 was mediocre.
2. **Breadth block (from the panel itself, no new data)**: advance/decline ratio, % universe above 50/200-SMA, % at 20-day highs/lows, cross-sectional return dispersion — all at T−1.
3. **Market-context block**: Nifty 50 distance to 20/50/200-SMA, 5/20-day Nifty returns, VIX level + 5-day Δ + 1-year percentile, ticker's sector-index relative strength (5/20-day), USDINR 5-day Δ, overnight US return (per D0 timing), Brent 5-day Δ.
4. **Label**: **3-bar forward close-to-close** — `label = Close(T+2)/Close(T−1) − 1` on the T−1-indexed row, i.e. a standard `shift(-3)` label (`label_horizon_bars=3`, `label_may_cross_session=True`). This stays A0.4-recomputable in the Gauntlet. *Known approximation*: actual entry would be T open, not T−1 close — the overnight gap is inside the label. Documented; acceptable for a gatekeeper (it gates exposure, it doesn't time entries).
5. Query_ID per day, ≥5 tickers/query, cross-sectional z-scoring per the house convention.
6. Register the feature pipeline in `PIPELINE_REGISTRY` for A1.1, or provide an explicit `prefix_invariance_waiver_reason` ONLY for the cross-asset join columns (the per-series features must still pass A1.1). Cross-asset join correctness is covered by D0's table + a dedicated builder unit test: assert no feature column in row T correlates 1.0 with any source value timestamped ≥ 09:00 IST on day T.

### D3 — Gauntlet Criteria v2 (the bundled Track-C fixes — UNIVERSAL, not daily-only)
**Files**: `scripts/gauntlet/contracts.py`, `verdict.py`, `cli.py`, `tests/gauntlet/`

1. **Magnitude-based FILTER alternative**: FILTER_GRADE = rho-test passes AND (recent WR z ≥ 2 **OR** recent top-K raw-return uplift vs same-period universe mean return is significant at t ≥ 2, two-sample/bootstrap). Catches tail-driven edges the WR test is blind to.
2. **Cadence-aware recent window**: `recent_window_months` becomes per-cadence: `{bar_minutes ≤ 60: 12, daily (≥1440): 24}` — restores comparable statistical power across timeframes.
3. Both changes go into **one new canonical GauntletConfig (Criteria v2)** → new config hash → new pre-registration. Old v1 verdicts remain valid as-issued; the report must print which criteria version judged the run.
4. Tests: tail-driven synthetic (high mean, ~50% WR) must FILTER under v2 and DEAD under v1; WR-driven synthetic unchanged; window test asserts 24 distinct months for a daily-cadence spec.
5. ⚠️ **Pre-registration discipline**: D3 must be merged and the v2 config frozen BEFORE the D4 run is started. One ledger trial for the new model. No reruns without a new hypothesis.

### D4 — Train & Certify `daily_macro_v2`
1. Same proven architecture: XGBoost `rank:pairwise`, depth 4–5, dual long/short via label inversion — **no new model math** (house rule: novelty is a bug source). Train via the standard registry path so `models/daily_macro_v2/metadata.json` carries features + params for the Gauntlet.
2. Register the DatasetSpec (`daily_macro_v2`) with `raw_source_glob` pointing at D1 parquets (positive label verification — no waiver), `label_horizon_bars=3`, cadence-aware window.
3. **One** Gauntlet run under Criteria v2. Pass target: FILTER_GRADE on at least one side.
4. **Halt-and-report rule**: if DEAD again, write the evidence to the vault and STOP — do not iterate-rerun (every run deflates the t-thresholds). A second hypothesis needs a new spec.

### D5 — Gatekeeper-Uplift Certification (the deferred Track A, run on the NEW model only)
**Files**: `scripts/gauntlet/uplift.py` + CLI subcommand `gatekeeper-uplift`

1. Inputs: `preds.npz` from the D4 run (daily scores per day×ticker) + `preds.npz` from the v8 and v2_15min R8 runs (downstream intraday OOS trades).
2. Join: daily score from row T−1 gates intraday trades on day T. Two pre-registered gate modes:
   - **Day-level**: market-aggregate daily signal (mean top-decile score spread) → favorable/neutral/unfavorable day terciles.
   - **Symbol-level**: ticker's daily score percentile → gate which symbols the intraday models may trade.
3. Test: net-bps of downstream top-K trades on favorable vs unfavorable tercile; two-sample t + query-bootstrap CI. **Pass: uplift ≥ +2 bps with t ≥ 2** (pre-registered in the same v2 config).
4. Output: uplift report in the run dir + a `gatekeeper_uplift` block appended to the model's Gauntlet stamp (via `registry.py` only).
5. n here is thousands of downstream trades — this is the statistically powerful test of the gatekeeper *function*.

### D6 — Live Integration (USER GATE — no agent action without approval)
- If D5 passes: wire `daily_macro_v2` into `orchestrator.py`'s `update_daily_macro_filters()` as the gatekeeper (soft tilt first), retire `daily_xgb` to `05. Archives/`.
- If D4 passes but D5 fails: model is a FILTER-grade ranker but not a useful gate — present evidence, user decides.
- `GAUNTLET_ENFORCEMENT` flip remains a separate user decision.

---

## 3. Dispersal Summary

| Pkg | Deliverable | Depends on | Est. effort |
|---|---|---|---|
| D0 | Decision-time contract + availability table | — | 0.25 day |
| D1 | Daily collectors (Upstox 10y + global) | D0 | 0.5 day |
| D2 | `build_daily_macro_dataset.py` + pipeline registration | D0, D1 | 1 day |
| D3 | Gauntlet Criteria v2 (magnitude FILTER alt + cadence window) + tests | — (parallel with D1/D2) | 0.5 day |
| D4 | Train `daily_macro_v2` + single pre-registered Gauntlet run | D2, D3 | 0.5 day |
| D5 | `gatekeeper-uplift` mode + certification run | D4 | 1 day |
| D6 | Live wiring decision | D5 | user gate |

**Total ≈ 3.5–4 agent-days.** Sequencing: D0 → (D1→D2) ∥ D3 → D4 → D5 → D6.

### Hard constraints (in addition to the standing Gauntlet ground rules)
1. **Point-in-time above all**: any feature whose availability cannot be proven against the D0 table gets dropped, not lagged-by-guess.
2. One D4 ledger trial. The halt-and-report rule is binding.
3. Criteria v2 changes are universal and frozen before any V2 result is seen.
4. Outstanding Gauntlet fix (stamping opt-in under test + ledger-existence check in `verify_model_stamp`, from the second audit) should ride along with D3 — same module, 30 minutes.
5. Collectors must be live-refresh-safe (the gatekeeper will need these feeds daily in production).

---

## 🔗 Backlinks
- Evidence: Gauntlet run `20260610T102743Z-5f7d069f` (daily_xgb DEAD autopsy) — see [[06. Context & Logs/Daily Logs/2026-06-10]]
- Harness: [[01. Core Architecture/Validation Gauntlet Architecture]] · [[01. Core Architecture/Validation Gauntlet Remediation Plan]]
- Verdict expectations: [[02. Model Suite/Model Performance & Statistics]]
