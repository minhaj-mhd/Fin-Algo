---
title: "Active Board"
type: reference
status: active
updated: 2026-06-18
tags: [board]
---
# 🎯 Active Board

> Current focus + next steps only (keep ≤ ~10 live items). Completed items roll into
> [[06 — Logs/Completed Work Archive|Completed Work Archive]]; dead research lines live in
> [[00 — Start Here/Dead-Ends Register|Dead-Ends Register]].

## 🔵 Current Focus

* **Daily Macro Scan Test Pollution Fix (COMPLETED)** — Resolved test suite runs overwriting the production `daily_gatekeepers.json` file with `MagicMock` failures. Prevented tests from executing the daily macro scan by checking if `pytest` is in `sys.modules` or `PYTEST_CURRENT_TEST` is in the environment. Re-ran the scan to restore valid long and short tickers in the live system. See [[06 — Logs/Conversations/Conv-2026-06-18-Daily-Macro-Gatekeeper-Mock-Pollution|Conversation Log]].
* **Shadow tracking peak negative PnL (COMPLETED)** — Implemented tracking of peak negative PnL (drawdown/maximum adverse excursion) in `trades` table, orchestrator tracking loop, risk manager, and dashboard UI templates (`vanguard_v2.html`, `strategy_detail.html`, `ticker_detail.html`). Verified with unit tests. See [[06 — Logs/Conversations/Conv-2026-06-17-Shadow-Peak-Negative|Conversation Log]].
* **Phase-1 Dual-Resolution Retrieval-based Hybrid model (COMPLETED)** — Implemented contrastive dual-resolution transformer pre-training, FAISS index with bucket-filtering, time-decay, and 12-D neighbor returns distribution feature extraction. Trained LightGBM Ranker with graded relevance labels. Diagnostics and walk-forward evaluations completed, showing significant Net PnL (+0.94 bps) and Sharpe (+1.17) uplifts for top-3 LONG picks. See [[06 — Logs/Conversations/Conv-2026-06-17-Transformer-Retrieval-Phase1|Conversation Log]].
* **Memory layer restructure (IN PROGRESS)** — vault reorganized into the `00–09` taxonomy with
  per-doc front-matter and a generated index (`scripts/memory/build_index.py`). See
  [[06 — Logs/Memory Layer Restructure Plan|Restructure Plan]].
* **15-min conviction-flip exit — open decision**: plumbing fixed, but the 15m model is low-signal;
  decide whether the flip exit earns its keep (all 15m overlays are sub-cost). See
  [[06 — Logs/Conversations/Conv-2026-06-12-Fix-15m-Conviction-Flip-Calc|fix log]].
* **Intraday edge — strategic**: four independent lines (CST, DualRes, sided-transformer, gate) all
  confirm the 1h price/volume ceiling is **information**, not model/loss. Next real lever is
  order-flow / microstructure data, not another model.
* **v2 + Transformer 1-Day Hold Backtest (COMPLETED)**: Verified that shortening the hold period from 3 days to 1 day destroys the +32 bps `v2` edge (drops to ~7 bps net, losing significance), while the Transformer still adds zero value (+0.28 bps uplift). Conclusion stands: Deploy `v2-alone` on the 3-day holding period. See [[06 — Logs/Conversations/Conv-2026-06-12-v2-Transformer-1D-Backtest|Conversation Log]].

## Next Steps

* [x] Build the first real `tests/` suite for trade lifecycle, risk math, feature-schema validation, and broker adapter mocks.
* [x] Audit tracked generated artifacts under `data/`, `models/`, and raw cache folders, then move non-canonical outputs out of git tracking.
* [ ] Deprecate or archive random-split training scripts and document walk-forward/temporal validation as the production standard.
* [x] Rewrite `README.md` into a clean operator guide with fixed encoding, current run modes, caveats, token requirements, and safety limits.
* [x] Pin dependencies or add a lock file so future installs reproduce the current working environment.
* [ ] Complete Phase 3 of the **[[04 — Research/Codebase Cleanup Strategy|Codebase Cleanup Strategy]]** (permanent pruning after 5 consecutive days of sandbox observation).
* [ ] Review and deploy the top-performing gated structural strategies (S3, S4, S13, S19, S23, S24) in the production live engine environment.
* [ ] Complete TradingView MCP setup (npm install, launch TV Desktop in debug mode, configure Gemini MCP settings).
* [ ] Build predictive models for Market Psychology using volume profiles and order flow data.
* [x] Design and backtest a 6+ member 1-hour public-data XGBoost ensemble using completed-bar labels, regime gates, and time-of-day pockets.
* [x] Implement Monotonic Constraints in the XGBoost architecture to strictly enforce economic logic (e.g. higher volatility must reduce Long conviction) and aggressively regularize against noise. **Result: FAILED. Human logic contradicts micro-structural mechanics. High relative volatility is required for 1H breakouts. Reverting constraints.**
* [x] Build `v17` Random Forest with bagging to test if an entirely different algorithmic paradigm handles noise better than Gradient Boosting.
* [x] Re-audit v8 with purged walk-forward — **DEMOTED, net-negative & decaying**. See [[02 — Models/_Shared/Model Performance & Statistics|Model Performance & Statistics]].
* [x] Re-audit `v2_15min_3y` with the same purged walk-forward methodology (v8 and v10-depth4 both failed; 15m is the last unaudited "active" model). → Fold into Gauntlet Phase P7 below.
* [x] **Build the Validation Gauntlet** per [[01 — Architecture/Validation Gauntlet/Validation Gauntlet Architecture|the approved spec]] — P0–P7 completed by agents; conformance audit verified the core harness but found 6 critical gaps.
* [x] **Execute the Gauntlet Remediation Plan** per [[01 — Architecture/Validation Gauntlet/Validation Gauntlet Remediation Plan|the R1–R8 spec]]: R1-R8 remediation plan executed, all tests pass, and R8 re-baseline campaign completed on all 4 models with secure SHA-256 stamps in metadata.
* [x] **CST Stage 0**: ran the falsification test — lead-lag features gave Δρ −0.0028/−0.0007 (no lift), **CST killed**; do not build the transformer. Redirect to order-flow/microstructure data. See [[02 — Models/Transformer/Cross-Sectional Transformer Architecture Proposal]] [!failure] callout.
* [x] Demote `v8_upstox_3y` from live trade-trigger duty (its own WF audit shows net-negative & decaying; replaced by `v10_native_1h` in active live trading, demoted to dashboard-only tracking).

Linked to: [[00 — Start Here/Welcome|Main Navigation Index]]

## Archived Daily Logs

Concluded conversations are archived into the dated Daily Logs below (originals removed from `Conversations/` per the teardown protocol):

* [[06 — Logs/Daily Logs/2026-06-16|Daily Log 2026-06-16]] — Fixed Nifty 500 YFinance ticker issue.
* [[06 — Logs/Daily Logs/2026-06-11|Daily Log 2026-06-11]] — V2 Daily Model details query.
* [[06 — Logs/Daily Logs/2026-06-10|Daily Log 2026-06-10]] — Daily Gatekeeper V2/V3 rebuild, standalone Gauntlet v2 certification, downstream gating uplift certification, Gemini rate-limit & sentiment fixes, Vanguard refactor.
* [[06 — Logs/Daily Logs/2026-06-09|Daily Log 2026-06-09]] — Jupyter MCP fix/clean-start, Gemini rotator + 503 handling, skipped-AI-trade tracking, holiday check, v17 tree print, V10/V18 query, TBM validation (⚠️ short "breakthrough" debunked as cost-sign bug), v19 CatBoost.
* [[06 — Logs/Daily Logs/2026-06-08|Daily Log 2026-06-08]] — 1H XGB edge diagnosis, sniper pipeline 3 teardown.
* [[06 — Logs/Daily Logs/2026-06-07|Daily Log 2026-06-07]] — Sniper OOS verification, model audits & edge discovery, re-audit of latest models.
* [[06 — Logs/Daily Logs/2026-06-06|Daily Log 2026-06-06]]
* [[06 — Logs/Daily Logs/2026-06-05|Daily Log 2026-06-05]]
* [[06 — Logs/Daily Logs/2026-06-04|Daily Log 2026-06-04]]

**Currently active (not yet concluded):** Print-v10-Final-Tree (06-08); API-Keys-Performance, Hybrid-Optimization, Hybrid-v10-v17, Test-Jupyter-MCP, v17-Random-Forest, v18-Directional-Random-Forest (06-09); [[06 — Logs/Conversations/Conv-2026-06-17-Check-Tuning-Progress|Check-Tuning-Progress (06-17)]]; [[06 — Logs/Conversations/Conv-2026-06-17-Shadow-Peak-Negative|Shadow-Peak-Negative (06-17)]].


