# 🧠 Meta-Veto Stacking Framework — Engineering & Certification Plan (M0–M5)

> **Status**: 📐 SPEC APPROVED FOR BUILD — disperse to implementing agents
> **Architect**: Claude (Fable 5), 2026-06-10
> **Prereqs**: [[01 — Architecture/Validation Gauntlet/Validation Gauntlet Architecture]], [[01 — Architecture/Validation Gauntlet/Validation Gauntlet Remediation Plan]], [[02 — Models/Daily Gatekeeper/Daily Gatekeeper V2 Rebuild Plan]] — all conventions bind. The uplift harness (`scripts/gauntlet/uplift.py`) is the certification engine this extends.

---

## 1. Motivation & Evidence Base

Manual gating failed certification everywhere (16 pre-registered uplift tests + a 40-cell post-hoc sweep), BUT the accumulated evidence defines exactly what to build next:

- **Uniform direction**: day-gate LONG uplift from the daily macro score is positive in *every* distinct cell ever tested (+1.01 to +5.07 bps, across 7 model families and 2 gatekeeper versions). Hypothesis-grade, not certifiable per-cell. See the audit banner on [[02 — Models/Gauntlet Reports/v3 Gating Sweep Report]].
- **One cell crossed zero**: daily-V3-favorable × v8 1H LONG symbol-gate trades = **+0.21 bps net** — the only net-positive intraday trade subset ever observed. Intersection-of-signals concentration works; single binary gates are too blunt.
- **The signal inventory is ~3 distinct sources, not 12 models**: (1) the 1h price/volume rank family (v8≡v10_native identical; v9/v11/v10_depth4 are siblings), (2) the 15m rank signal, (3) the daily macro signal (the only orthogonal information). Plus two non-model conditioners: time-of-day and the structural macro gate.
- **Therefore**: stop hand-picking gate combinations (40-cell sweeps are noise factories). Train ONE tiny meta-model that learns the combination *inside* the training window, and certify it ONCE on data no sweep has shaped.

**The Meta-Veto's job**: for each candidate intraday trade emitted by a downstream model, output `P(trade is net-positive)`; trades below threshold θ are vetoed before execution (slotting in as a new ML veto stage ahead of the Gemini vetoes).

---

## 2. The Cardinal Risk & Its Firewall

This is a model trained on model outputs, evaluated on a history that the V2/V3 uplift tests and the 40-cell sweep have already *seen*. Without structural protection, any positive result is laundering.

**Time Firewall (non-negotiable):**
- **DEV span**: all panel rows with trade date **< 2025-01-01**. Unlimited experimentation here — feature choices, model class, θ tuning, walk-forward dev metrics. Everything fit/selected on DEV only.
- **VAULT span**: trade dates **≥ 2025-01-01**. Touched **exactly once**, by the single pre-registered M4 certification run, after the meta-model + θ are frozen and hashed. No peeking, no dev metrics computed on it, ever.
- Enforced in code: the panel loader takes a `span` argument; the trainer hard-asserts `max(date) < 2025-01-01`; the certifier hard-asserts the model artifact hash matches the frozen pre-registration before loading VAULT rows.
- Caveat to record honestly in the final report: the sweeps did look at 2025–2026 *aggregates* (uplift means per cell), so the VAULT span is "lightly contaminated" at the cell-statistics level, though never at the per-trade model-fitting level. The true clean test is live data; the M4 verdict should be labeled CONDITIONAL and scheduled for a live-data re-confirmation after ~3 months of paper trading.

---

## 3. Work Packages

### M0 — Candidate Trade Panel Builder
**File**: `scripts/gauntlet/meta/build_trade_panel.py` → `data/gauntlet/meta/trade_panel.parquet`

The panel is the dataset of the meta-model: one row per **candidate trade** = (downstream_model, datetime, ticker, side) for every top-`primary_k` selection in the downstream models' Gauntlet OOS predictions.

1. **Sources (all existing artifacts — no retraining)**: `preds.npz` from the R8/批 runs of `v8_upstox_3y` (1H) and `v2_15min_3y` (15M) = candidate trades + realized forward returns; `preds.npz` from `daily_macro_v2` and `daily_macro_v3` = daily scores; optionally `v10_depth4_1h`, `v11_utility_1h` scores as additional 1h-family inputs (max ONE per identical-spec family — v8≡v10_native counts once).
2. **Feature columns per row** (all must predate trade entry — point-in-time audit per the D0 discipline):
   - Own-model score percentile (within query) and raw score z;
   - Sibling 1h scores (depth4, v11) as percentiles;
   - Cross-timeframe: for 15m trades, the concurrent 1h score; for 1h trades, the same-bar 15m score *of the completed prior 15m bar only*;
   - Daily V2 long/short score percentiles (T−1 join, reuse `uplift.py` join + tz logic);
   - Daily V3 long/short score percentiles (T−1);
   - Context: time-of-day bucket, side, day-of-week, structural macro gate state (Nifty vs 200-SMA), VIX percentile (T−1);
   - Label: realized raw forward return; derived target `y = (raw_return − 10bps) > 0` (binding cost).
3. **Audit block (Stage-0 style, hard assertions)**: no feature may correlate ≥0.95 with the label; every join verified T−1-lagged; row counts per source reconciled; panel SHA256 recorded. Overlap window only (dates where all sources have OOS coverage).
4. Output split markers: `span = DEV | VAULT` column baked in at build time.

### M1 — Orthogonality & Information Audit (zero ledger trials)
**File**: `scripts/gauntlet/meta/orthogonality_audit.py` → vault note `08. Model Analysis/Meta-Veto/Orthogonality Audit.md`

On DEV span only:
1. Pairwise Spearman matrix of all score columns (confirms the true count of independent signals; expect 1h-family ≈ 0.8–0.95 internal correlation).
2. Partial information: incremental rank-IC of each signal on the trade outcome after controlling for own-model score (does daily/15m/ToD add anything the 1h score doesn't have?).
3. Conditional concentration table (diagnostic only, watermarked): net bps of trades in the all-signals-favorable intersection vs base rate.
4. **Kill criterion (pre-declared)**: if no non-own signal adds ≥0.005 incremental IC, the meta-veto is dead on arrival — stop and report, do not proceed to M2.

### M2 — Meta-Veto Model
**File**: `scripts/gauntlet/meta/train_meta_veto.py` → `models/meta_veto_v1/`

1. **Deliberately tiny, in this order of preference**: L2 logistic regression first; only if DEV walk-forward shows clear nonlinear residual, a depth-2 / ≤32-leaf GBM. The capacity ceiling is a feature: the panel has only ~3–8k DEV trades per downstream model — anything bigger memorizes.
2. **Training discipline**: purged walk-forward *within DEV* (embargo ≥ 3 days — the daily features have a 3-day label heritage); per-fold standardization from train folds only; seed from config; one model per downstream family (1H and 15M) OR one pooled model with a downstream-model indicator — choose ON DEV, freeze the choice.
3. **Veto threshold θ**: chosen on DEV to maximize kept-trades net bps subject to **keeping ≥ 25% of trades** (pre-declared floor — prevents the degenerate "veto everything" solution and keeps enough live trade flow to matter).
4. **Primary endpoint (clinical-trial style, pre-declared now)**: **v8 1H LONG trades** — the only cell family with uniform positive evidence. 1H SHORT, 15M LONG/SHORT are *secondary* endpoints (reported, not grounds for certification on their own).
5. Artifact: model + θ + feature list + panel hash frozen into `models/meta_veto_v1/metadata.json`; registry-stamped via `registry.py` after M4 (never manually).

### M3 — Certification Harness Extension
**File**: extend `scripts/gauntlet/uplift.py` (or `scripts/gauntlet/meta/certify_meta_veto.py` reusing its internals)

1. Mechanics: load frozen `meta_veto_v1`, score every VAULT-span candidate trade, split kept (P ≥ θ) vs vetoed, compute the uplift statistics (existing machinery) **plus absolute kept-trade profitability**.
2. Pre-registration plumbing identical to Gauntlet runs: config lock before scoring, ledger `started`/`completed` events, run_id, deflated-t context, report.json/md in the run dir.
3. **Sandbox rules apply** (post-stamping-fix): certification under `GAUNTLET_ROOT`, stamping only by the real run.

### M4 — THE Certification Run (one trial, two-part bar — both pre-registered)
| Criterion | Threshold | Why |
|---|---|---|
| **(a) Uplift** | kept − vetoed ≥ **+3.0 bps net** with **t ≥ 2.0** on the primary endpoint (v8 1H LONG) | the veto must demonstrably separate trades |
| **(b) Absolute** | kept trades pooled **net bps > 0 with t ≥ 2.0 at 10 bps binding cost** | "lose less" is not a strategy — this is the bar every prior gate failed; crossing zero is the entire point |

**Pre-declared verdict map**: (a)+(b) pass → `META_VETO_CERTIFIED (CONDITIONAL — live re-confirmation pending)`; (a) only → FILTER-grade veto (may shrink size on vetoed trades, may not enable/disable trading); neither → line closed, archive, and the honest conclusion is that price/volume+macro information caps out below costs at intraday horizons — redirect effort to new information (options OI/depth/sentiment-as-feature).
Secondary endpoints reported with 4-way-corrected thresholds (t ≥ ~2.5) — informational unless they independently clear correction.

### M-DYN — Dynamic Composition Architecture (makes the framework swappable, not single-purpose)

The framework must support swapping component models in/out and retraining the meta-layer **without code changes** — only configs. Dynamism lives at the engineering layer; certification stays single-shot.

**1. `SignalSource` registry** (`scripts/gauntlet/meta/sources.py` + `data/gauntlet/meta/sources/*.json`):
Every component model is registered by a small manifest, auto-derivable from its Gauntlet run:
```json
{
  "name": "daily_macro_v3",
  "family_id": "daily_macro",            // one-per-family rule enforced by the builder
  "preds_artifact": "data/gauntlet/<run_id>/preds.npz",
  "gauntlet_run_id": "20260610T144343Z-5f7d069f",
  "join_type": "daily_t_minus_1",        // or "intraday_qid"
  "columns": ["long_pct", "short_pct"],
  "live_adapter": "daily_scan_scores"    // how to compute this value in the live engine (M5)
}
```
Adding a future model (CST, options-feature model, anything that passes the Gauntlet) = drop in one manifest. The panel builder **refuses** two sources sharing a `family_id` (kills the v8≡v10 double-count class structurally).

**2. `PanelSpec` config** (YAML/JSON, hashed): declares downstream trade source (which model's top-K candidates), the list of signal sources, context features, firewall date. `build_trade_panel.py` becomes a generic compiler: PanelSpec → panel parquet + SHA256. Swapping the mix = editing the config.

**3. `MetaModelSpec` config**: model class (`logistic` | `gbm_depth2` only — the registry of allowed classes is intentionally closed), hyperparams, θ-selection rule, primary endpoint.

**4. DEV experiment runner** — `python -m scripts.gauntlet.meta dev-run --panel-spec P.yaml --model-spec M.yaml`:
Trains + evaluates on DEV span only (purged WF), writes results to a **separate DEV experiment ledger** (`data/gauntlet/meta/dev_ledger.jsonl`). DEV experiments are unlimited and free — swap, retrain, compare at will — but every one is logged, because **selection intensity must be visible**: the M4 certification report MUST print `n_dev_experiments_tried` so the one VAULT result is read in context of how many candidates competed for it.

**5. Freeze → certify** — `freeze` bundles {panel-spec hash, model artifact hash, θ, endpoint} into an immutable candidate; `certify` runs the single VAULT trial against it. The certifier refuses any candidate whose hashes don't match the freeze.

**6. Re-certification policy**: a *new* certification trial is legitimate only when the candidate contains **genuinely new information** (a new signal source from a new Gauntlet-passed model) — not a re-tuned θ or a reshuffled mix of the same sources. The certification ledger entry must name what's new.

### M5 — Live Integration (USER GATE)
- If certified: new veto stage in `scripts/vanguard/signal_generation.py` ordered **after** candidate generation, **before** the Gemini Stage-1 technical veto (it's local, free, and fast — Gemini quota is precious). Inference path: assemble the same panel features live (daily scores already computed by the engine's daily scan; sibling scores from the loaded models; VIX/ToD from the pulse scan); veto if P < θ.
- 3-month paper-trade/shadow-tracking re-confirmation window (`start_vetoed_tracking()` already exists for exactly this) before any size is risked on it; live results then upgrade or revoke the CONDITIONAL stamp.

---

## 4. Dispersal Summary

| Pkg | Deliverable | Depends on | Effort |
|---|---|---|---|
| M0 | Generic trade panel compiler (PanelSpec-driven) + audit | existing preds.npz | 1 day |
| M-DYN | SignalSource registry + manifests + DEV runner + dev_ledger + freeze/certify CLI | M0 | 1 day |
| M1 | Orthogonality audit + kill criterion | M0 | 0.5 day |
| M2 | Meta-veto model (DEV-only, via dev-run) + frozen θ | M0, M-DYN, M1 pass | 1 day |
| M3 | Certification harness extension | uplift.py | 0.5 day (parallel w/ M2) |
| M4 | Single VAULT certification run (prints `n_dev_experiments_tried`) | M2+M3 | 0.25 day |
| M5 | Live veto stage via `live_adapter` mapping + shadow window | M4 pass + user approval | 0.5 day |

**Total ≈ 4.5–5 agent-days.** Strict order: M0 → M-DYN → M1 (kill gate) → M2 ∥ M3 → M4 → M5.

### Hard constraints
1. **The firewall is sacred**: no VAULT-span metric may be computed before M4; trainer/certifier enforce by assertion. Violating this voids the entire exercise.
2. **One M4 run.** A failed M4 is a closed question, not an invitation to tune θ and retry (AGENTS.md Metric Discipline rule 5).
3. Max one model per identical-spec family in the panel (v8≡v10_native = one).
4. All features point-in-time audited (D0 discipline); no label-derived features; panel hash + config hash + model hash chained into the pre-registration.
5. Tiny model classes only (logistic / depth-2). An agent reaching for depth-6 XGBoost on 5k rows is reproducing the original disease.
6. Engineering Discipline rules (AGENTS.md) apply in full — sandboxed tests, evidence-backed claims, halt-and-report.

---

## 🔗 Backlinks
- Evidence: [[02 — Models/Gauntlet Reports/v3 Gating Sweep Report]] (with multiplicity audit banner) · [[02 — Models/Daily Gatekeeper/Daily Gatekeeper V2 Rebuild and Certification Report]]
- Harness: [[01 — Architecture/Validation Gauntlet/Validation Gauntlet Architecture]] · uplift harness `scripts/gauntlet/uplift.py`
- Discipline: AGENTS.md → Model Metric Discipline + Engineering Discipline
