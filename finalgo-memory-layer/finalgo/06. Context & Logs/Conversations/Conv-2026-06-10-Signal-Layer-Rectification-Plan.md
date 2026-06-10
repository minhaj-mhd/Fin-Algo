# 💬 Conversation Context: Signal Layer Rectification Plan

## 📌 Metadata
- **Conversation ID**: (current Claude session, 2026-06-10)
- **Start Date**: 2026-06-10
- **Status**: 🟢 Active
- **Focus Area**: Model Suite / 1H Signal Generation Strategy

## 🎯 Objectives
- [x] Diagnose why every 1H signal model (v8→v19, TBM, hybrids) "has one or other issue".
- [x] Agree on a rectification path (validation gauntlet + new data sources + role reframing).
- [x] Architect the Validation Gauntlet in depth and publish the spec to the vault for task dispersal.
- [x] Disperse build phases P0–P7 to implementing agents (user-driven) — build completed and audited.
- [x] Audit the agents' P0–P7 build for architecture conformance — core harness verified, 6 critical + 9 moderate gaps documented.
- [x] Architect the remediation spec for dispersal: published [[01. Core Architecture/Validation Gauntlet Remediation Plan]] (R1–R8 work packages with per-package implementation detail, tests, acceptance criteria, dependency graph).
- [ ] Disperse R1–R8 to implementing agents (user-driven); then review R8 expected-vs-actual verdicts before considering enforcement flip.

## 📝 Compacted Session Log
- **Initial Analysis**: User asked how to finally get a bug-free, deployable 1H signal model. Reviewed [[06. Context & Logs/Current Context|Current Context]] and memory: v8 demoted (early-stopping-on-test leakage; true WF ρ≈0.025 decaying, net-negative at 6bps), v10-depth4 net-negative, v17/v18/v19 AUC≈0.50, TBM short "edge" was a cost-sign harness bug, monotonic constraints failed.
- **Diagnosis delivered**: The recurring "bugs" are validation-harness bugs that made dead models look alive — the architecture search is exhausted; 1H OHLCV features carry ~2–3bps gross edge vs 6–10bps costs. No architecture fixes that.
- **Proposed plan**: (1) pull/demote v8 from live triggering, (2) build one canonical purged-WF "model gauntlet" with built-in leakage & cost-sign self-tests as the single pass/fail gate, (3) add genuinely new information (options OI/PCR/IV, market internals, Upstox depth/order-flow, sentiment-as-feature), (4) reframe 1H layer as a ranker/filter inside the pipeline (short-side score z=5–7 is real as a filter), concentrating execution in verified ToD pockets + Sniper Tier B.
- **Gauntlet architected (2026-06-10)**: Studied [v8_walkforward.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/analysis/v8_walkforward.py) (the trusted reference harness), [build_training_data.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/training/build_training_data.py) (label/query construction), `models/v8_upstox_3y/metadata.json` (registry contract), `scripts/feature_utils.py` (902-line feature pipeline), and the existing `tests/` suite. Published the full spec to [[01. Core Architecture/Validation Gauntlet Architecture]]: 6-stage pipeline, 5 historical bug classes → automated assertions (A0.x/A1.x), 3-tier per-side verdicts, pre-registration + deflated-t multiple-testing discipline, registry stamping + live-engine guard, synthetic self-test matrix T1–T8 (T8 = regression vs known v8 WF numbers as build acceptance), and a strictly-ordered P0–P7 build plan (~7–8 agent-days) with per-phase acceptance criteria and hard constraints for implementing agents.
- **Memory layer updated**: Welcome.md nav entry added; Current Context Active Focus + Next Steps updated (Gauntlet build, v8 demotion, 15m re-audit folded into P7).
- **Architecture-conformance audit of the agents' build (2026-06-10)**: Core harness VERIFIED — gauntlet v8 run reproduces the trusted WF reanalysis (ρ 0.0253/0.0243 vs 0.0261/0.0245; short Top-3 net matches to 0.02 bps). Registry stamps + checksums present; live-engine warn guard wired ([model_inference.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/model_inference.py) L77–100, `GAUNTLET_ENFORCEMENT="warn"`). **6 critical gaps found**: (1) A1.1 prefix-invariance never wired into `run_gauntlet` — real feature pipeline never lookahead-tested; (2) overnight guard hollow — checks `dt+bar_minutes`, not the actual next-bar label; empirically all 1,083 last-bar-of-day (13:15) rows/ticker in `1h_v3_3y` have in-file-unverifiable labels that A0.4 silently skipped; (3) T8 regression test absent from the self-test suite (passed manually only); (4) pre-registration broken — `config.lock.json` written AFTER the run; (5) FILTER_GRADE too lenient (z-test vs 0.5 instead of universe baseline WR, best-of-K cherry-pick, 13-mo recent window) — v8 LONG got FILTER_GRADE contradicting the trusted raw-signal audit (longs weak/decayed); (6) `preds.npz` not saved. Moderate: self-test runs pollute production ledger; no `.gitignore` rules for `data/gauntlet` (2.7 GB parquet cache addable); AGENTS.md run-id rule (P6) missing; A0.3 single-max-time heuristic; T6 tests via mock only. **Recommendation: keep enforcement in "warn", treat current FILTER_GRADE stamps (esp. v8 long) as provisional until fixes F1–F6 land.**

## 🔗 Core Memory Links & Backlinks
- [[02. Model Suite/Model Performance & Statistics]]
- [[08. Model Analysis/1-Hour Vanguard Model/Live Trading Configuration & Verdict]]
- [[06. Context & Logs/Conversations/Conv-2026-06-10-V8-Walkforward-Reanalysis]]
