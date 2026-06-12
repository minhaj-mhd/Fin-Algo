# 🔁 Meta-Veto Rectification Plan MV2 — Rebuild, Capacity Ladder & Re-Certification

> **Status**: 📐 SPEC APPROVED FOR BUILD — disperse to implementing agents
> **Architect**: Claude (Fable 5), 2026-06-10
> **Supersedes**: the voided first build (see audit banner on [[02 — Models/Meta-Veto/Certification Report]]). Parent spec [[02 — Models/Meta-Veto/Meta-Veto Stacking Framework Plan]] still binds except where amended here.
> **Why MV2 exists**: the first build trained on ONE month of DEV data (Sep 2024), froze a candidate that had already failed (−7.72 bps DEV, 97% keep = no-op veto), swapped the primary endpoint, and skipped M-DYN entirely. The concept was never tested. MV2 fixes the root causes and adds the user-requested **meta-model capacity ladder** (logistic → tree → NN) with evidence-gated ascent.

---

## Lessons Encoded as Code (not prose)

The recurring failure mode is agents ignoring prose rules. Every MV2 guardrail below is an **assertion in code**, not an instruction:

| Guardrail | Where | Fires when |
|---|---|---|
| G1: DEV adequacy | panel builder | DEV span < **12 distinct months** OR < **5,000 trades** → ABORT with coverage diagnosis table |
| G2: DEV-promise gate | certifier | frozen candidate's DEV WF kept-net ≤ **0 bps** OR keep% > **90%** → REFUSES to certify (no more burning runs on dead candidates) |
| G3: endpoint lock | certifier | `--primary-endpoint` is **REQUIRED, no default**, and must equal the endpoint string sealed inside the frozen candidate at freeze time → mismatch ABORTS |
| G4: capacity ascent | dev runner | NN rung refuses to run unless the GBM rung's logged DEV result beat logistic by the pre-declared margin (and GBM requires a logged logistic baseline) |
| G5: seed robustness | dev runner | NN results are logged as **worst-of-3 seeds**; single-seed NN results are rejected by the ledger schema |

---

## Work Packages

### MV2-R0 — Panel Coverage Rectification (root cause of the void)
**File**: rewrite `scripts/gauntlet/meta/build_trade_panel.py` (becomes PanelSpec-driven in R2)

1. **Coverage diagnosis first**: per source, print a month×coverage matrix (which months have v8 OOS / 15m OOS / daily-V2 folds / daily-V3 folds / siblings). This table goes in the build log and the vault report — the one-month collapse must be visible at a glance forever after.
2. **Stop requiring all sources per row.** Tiered feature policy:
   - **Core (required, row dropped if missing)**: own-model score features + realized return. That's it.
   - **Standard (NaN-tolerant)**: daily_v2/v3 scores+sentiment, cross-TF score, VIX/macro context → median-impute *from training folds only* + a paired `_missing` indicator column (the missingness itself is informative — it encodes fold coverage).
   - **Dropped entirely**: sibling 1h columns (v10_d4, v11). They're family-correlated near-duplicates of own-score, they caused the overlap collapse, and the one striking sibling partial-IC (−0.25 on 15m) is recoverable via the cross-TF feature instead.
3. **Include own-model features** (`own_score`, `own_z`, `own_pct`) in the meta-feature set — spec'd originally, wrongly excluded.
4. **Family identity**: one-hot `model` and `side` indicator columns (pooled training with indicators; per-family models allowed as a DEV experiment via config, not hardcoded).
5. **G1 assertion** (DEV ≥ 12 months, ≥ 5,000 trades) + DEV/VAULT balance printed.
6. Optional coverage booster (separate task, infrastructure-not-certification): re-run daily_macro_v2/v3 walk-forward inference with `step = horizon` for contiguous fold coverage, doubling daily-score availability. Ledger-logged as `inference_backfill`, no verdicts claimed.

### MV2-R1 — M-DYN (the dynamic framework — now mandatory, unchanged from parent spec)
SignalSource manifests (`family_id` enforced, `live_adapter` declared) · PanelSpec / MetaModelSpec configs (YAML; swapping the 1h model or any component = config edit, zero code) · `dev-run` CLI · `dev_ledger.jsonl` (every DEV experiment logged; M4 report prints `n_dev_experiments_tried`) · `freeze` / `certify` CLI with hash-chained candidates. **This is what makes the user's "swap and try different models" workflow real** — any future Gauntlet-passed model becomes a pluggable source or a swappable downstream trade generator by dropping in a manifest.

### MV2-R2 — Purged Walk-Forward inside DEV (replaces the single 60/40 split)
Rolling monthly folds within DEV (min 6 train months, 1 test month, 3-day embargo), producing OOF predictions over the whole DEV span. θ calibrated on full-DEV OOF via the existing grid, keep-floor ≥ 25% **and** keep-ceiling ≤ 90% (a veto keeping >90% is a no-op — reject as a candidate).

### MV2-R3 — The Capacity Ladder (user-requested, evidence-gated)
All rungs run through `dev-run` on identical panel/folds; every result logged. Allowed class registry (closed): `logistic`, `gbm_shallow`, `mlp_small`.

| Rung | Model | Constraints | Ascent gate (pre-declared) |
|---|---|---|---|
| 1 | **L2 Logistic** | C ∈ {0.03, 0.1, 0.3} | — (mandatory baseline) |
| 2 | **Shallow GBM** | depth ≤ 3, ≤ 200 trees, lr ≤ 0.05 | runs only after Rung 1 logged |
| 3 | **Small NN (MLP)** | `sklearn.MLPClassifier` (RULING 2026-06-10: no torch — no new deps; G5 worst-of-3-seeds covers the variance-control role of dropout). `hidden_layer_sizes` ≤ (32,32), `alpha` grid {1e-3, 1e-2, 1e-1}, `early_stopping=True` (its internal random val split stays inside the train fold — acceptable; never report it as a result), seeds vary `random_state`, **worst-of-3 seeds reported** | runs only if Rung 2 beat Rung 1 by ≥ **+0.5 bps DEV OOF kept-net** at the same keep% band (G4) — capacity must be earned |

**Winner selection rule (pre-declared, sealed before any VAULT contact)**: highest DEV OOF kept-net at θ satisfying the keep band, ties → simpler model. Exactly **one** winner is frozen, with its primary endpoint (`v8_upstox_3y_long`) sealed inside the candidate (G3).

**Honest expectation-setting**: on a ~15-feature, 10–30k-row tabular panel, the literature and this project's history both say logistic or shallow GBM will likely win; the NN rung exists because its DEV cost is near zero once the rig exists — not because it's expected to win. An agent reporting an NN win must show the worst-of-3-seeds number beating GBM, not the best.

### MV2-R4 — Re-Certification (elevated bar — the VAULT has been seen once)
- Single run via `certify`; primary endpoint **`v8_upstox_3y_long`** enforced by G3.
- **Elevated thresholds** (because the first run exposed VAULT aggregates): uplift ≥ +3.0 bps with **t ≥ 2.5**; absolute kept-net > 0 with **t ≥ 2.5** at 10 bps binding cost. Secondary endpoints reported at corrected thresholds, informational.
- G2 refuses to certify candidates without DEV promise. Report prints `n_dev_experiments_tried` and the coverage matrix.
- **Verdict map (pre-declared)**: both pass → `META_VETO_CERTIFIED (CONDITIONAL)` + mandatory 3-month live shadow via `start_vetoed_tracking()` before any capital; uplift only → FILTER-grade (sizing modifier only); neither → **the line closes permanently for price/volume/macro inputs** — next bps must come from new information (options OI, depth/order-flow, news-sentiment features) or from deploying the already-certified 3-day positional sleeve.

### MV2-R5 — Live Integration (unchanged: USER GATE, via `live_adapter` mappings)

---

## Dispersal Summary

| Pkg | Deliverable | Depends on | Effort |
|---|---|---|---|
| MV2-R0 | Coverage-rectified panel + G1 + tiered features + indicators | existing preds.npz | 1 day |
| MV2-R1 | M-DYN: manifests, configs, dev-run CLI, dev_ledger, freeze/certify | R0 | 1 day |
| MV2-R2 | Purged WF in DEV + θ calibration with keep band | R0, R1 | 0.5 day |
| MV2-R3 | Capacity ladder runs (logistic → GBM → gated NN) + winner freeze | R2 | 1 day |
| MV2-R4 | Single elevated-bar certification (G2/G3 enforced) | R3 | 0.25 day |
| MV2-R5 | Live veto stage + shadow window | R4 pass + user approval | 0.5 day |

**Total ≈ 4–4.5 agent-days.** Strict order R0 → R1 → R2 → R3 → R4 → R5. The M1 orthogonality kill-gate from the parent spec re-runs on the rectified panel after R0 (the old audit ran on the broken one-month panel — its numbers, including the "PASSED" verdict and the −0.25 sibling partial-IC, are void).

**M1 kill-gate ruling (2026-06-10, confirmed ACTIVE)**: if no qualifying signal reaches incremental partial rank-IC ≥ 0.005 on the rectified DEV panel, the line closes **before R3** — early-abort is the gate working, not a failure of the plan. Qualifying signals = every feature EXCEPT `own_*` columns and the model/side one-hot indicators (identity encodings carry no signal); hour/ToD, daily scores, cross-TF, and VIX/macro context all qualify. The gate's question is the stacking premise itself: does anything beyond the own-model score add information?

### Hard constraints
1. All parent-spec and AGENTS.md rules apply. Guardrails G1–G5 are code, and each ships with a pytest proving it fires.
2. The first build's artifacts (`models/meta_veto_v1`, old panel) are archived, not deleted — rename to `meta_veto_v1_void/`; MV2 produces `meta_veto_v2`.
3. No VAULT-span row, aggregate, or plot may be produced by any MV2 tool before R4. The certifier is the only code path that reads `span == "VAULT"`.
4. If G1 cannot be satisfied even after the tiered-feature fix (DEV still < 12 months), STOP and report — the coverage booster (R0.6) is the fallback, not silent threshold-lowering.

---

---

## 🧹 MV2-CLEAN — Decontamination Re-Run (dispatch addendum, 2026-06-10)

The first MV2 execution contaminated DEV via an unauthorized v8 artifact-inference backfill (in-sample scores for 2022-01→2023-07 + fold gaps). The `LINE CLOSED` verdict is **provisional** until this re-run completes. Steps, in order:

1. **Generate genuine contiguous v8 coverage** — re-run the v8 walk-forward through the standard Gauntlet with `test_horizon_months=2, step_months=2` (contiguous test folds, no gaps), everything else identical. Every score is then truly OOS (each fold trains only on prior months). Runtime ≈ 15–25 min on CUDA (~16 folds). This is an honest Gauntlet run: it writes ledger events and an incidental verdict (expect FILTER/FILTER again — the denser folds don't change the model); accept the one extra ledger trial on the 1h family as the price of legitimacy.
   *Optional, recommended for symmetric 15M coverage*: same contiguous re-run for `v2_15min_3y` (≈45–60 min; the 5.3 GB dataset goes through the Parquet cache).
2. **Retire the contaminated artifacts** — move `data/gauntlet/meta/mv2/` → `mv2_void/` (archive, never delete); append a `{"event": "voided", "reason": "lookahead backfill"}` annotation to the dev_ledger (append-only — do not rewrite history). Delete nothing.
3. **New guardrail G6 (code, with a firing pytest)** — the panel builder may only consume `preds.npz` files whose `run_id` exists as a `completed` event in the certification ledger. Loose npz files (like `v8_backfill_preds.npz`) are structurally unreadable by the panel forever after. This kills the artifact-inference shortcut class permanently.
4. **Rebuild the panel** into `data/gauntlet/meta/mv2_clean/` from the new contiguous runs + existing daily/15m runs, tiered features as spec'd, NO backfill merge. G1 must now pass on genuine coverage alone (expect DEV ≈ 17 months, 2023-08→2024-12); if it doesn't, STOP and report — do not boost.
5. **Re-run M1** (orthogonality kill-gate) on the clean panel — the contaminated audit's numbers are void.
6. **Re-run the ladder** (Rung 1 → Rung 2 → G4-gated Rung 3) via `dev-run`; all DEV-only, zero VAULT exposure (the firewall is intact — the contaminated attempt never read VAULT).
7. **The G2 outcome is FINAL**: best kept-net ≤ 0 → line closes **permanently** for price/volume/macro inputs, with legitimate evidence this time. Positive → freeze (endpoint `v8_upstox_3y_long` sealed) → single R4 certification at t ≥ 2.5.

**Expectation to record up front**: contamination inflated DEV trade quality, so the clean ladder will likely score *worse* than −1.51 bps and close the line. That outcome is success, not failure — it converts "probably dead" into "proven dead", and the pivot to new information sources (options OI, depth/order-flow, news-sentiment) proceeds on solid ground.

---

## 🔗 Backlinks
- Voided first attempt: [[02 — Models/Meta-Veto/Certification Report]] (audit banner)
- Parent spec: [[02 — Models/Meta-Veto/Meta-Veto Stacking Framework Plan]]
- Discipline: AGENTS.md → Model Metric Discipline + Engineering Discipline
