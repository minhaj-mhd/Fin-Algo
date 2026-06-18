---
title: Price-Action / SMC Transformer
type: log
status: active
updated: 2026-06-15
verdict: DEAD for tradability — explicit price-action/SMC adds no edge at 1h (features ρ≈noise; level-graph gated GCN real≈neg-control). NEW PHASE: closing val→test generalization gap via walk-forward. ⚠️ UNVERIFIED (exploratory, no Gauntlet)
model: pa_transformer / level_gcn
---

# 💬 Conversation Context: Price-Action / Smart-Money-Concept Transformer

## 📌 Metadata
- **Start Date**: 2026-06-15
- **Status**: 🟢 Active
- **Focus Area**: Model Suite — new transformer with explicit price-action / SMC features
- **Driver**: user `/loop` request — a transformer that "has learned" S/R, candlestick
  patterns, chart patterns, order blocks, FVGs, volume profile, delta divergence,
  stop hunts, mitigation/retest.

## 🎯 Objectives
- [ ] Map the user's playbook to what is **computable from data we own** vs needs missing data.
- [ ] Build a lookahead-free price-action / SMC feature module (`scripts/features/price_action.py`).
- [ ] Build an augmented dual-res panel (`data/transformer_panel_smc/`) = 81 TA + PA features.
- [ ] Train the existing DualResCSTransformer on the augmented panel.
- [ ] Honest per-side, net-of-cost eval vs the 81-feature baseline + neg-control + WF.
- [ ] Write verdict to vault (exploratory — NO Gauntlet authority).

## 🧭 Feasibility map (data reality)
**Buildable from OHLCV (have 15m/1h/daily 3y Upstox):**
- Candlestick patterns (hammer, shooting star, engulfing, doji, inside bar, morning/evening star) — pure OHLC geometry. ✅
- Horizontal S/R from confirmed swing pivots (distance, touch-count, round-number magnet). ✅
- Fair Value Gaps / 3-candle imbalance (formation flag + distance to nearest unfilled gap). ✅
- Order blocks (last opposite candle before impulsive displacement). ✅
- Liquidity sweeps / stop hunts (prior swing taken then reclaimed). ✅
- Displacement / impulse strength (body / ATR). ✅
- Chart patterns (H&S, triangles, double tops) — derivable from swing pivots but noisy. ⚠️ later
- Volume-profile POC/VAH/VAL/LVN — crude approx possible from 15m bars only. ⚠️ later

**NOT buildable (missing data):**
- Delta / footprint divergence — needs tick buy/sell volume. ❌
- Historical OI-magnet S/R — Upstox OI paywalled, see [[project_oi_plus_paywall]]. ❌
- True volume profile — 5m 3y cache is empty (`raw_upstox_cache_5min_3y/` has 0 files). ❌

## ⚖️ Prior reality (why this is a real test, not a re-run of a dead end)
Every prior transformer is SUB-COST and the repeated diagnosis is **info-limited, not
arch-limited**: [[project_cst_stage0_killed]], [[project_dualres_transformer_result]],
[[project_sided_transformer_result]], [[project_confirm_v10_cosign_deadend]],
[[project_daily_transformer_veto_deadend]]. BUT all of them fed momentum/oscillator
features. **None fed explicit price-action geometry.** The pre-registered hypothesis:
*do SMC/price-action features (new information from the same OHLCV) lift cross-sectional
rank-IC / net-of-cost edge over the 81-feature TA panel?* Prior = skeptical.

## 💻 Active Code Files
- [price_action.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/features/price_action.py) — new feature module (lookahead-free)
- [build_tensor_panel.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/build_tensor_panel.py) — to be extended → SMC panel
- [model.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/model.py) / [train.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/train.py)

## 📝 Compacted Session Log
- **Initial Analysis**: Existing transformer = 81 TA features (no price-action). Confirmed
  data availability; 5m 3y cache empty, OI=0.0 in raw. Set the new-information hypothesis.
- **Step 1 (iter 1 ✅)**: Built `scripts/features/price_action.py` — 27 lookahead-free PA
  features (candles, S/R from confirmed fractals, FVG w/ unfilled-gap tracking, order
  blocks, sweeps, displacement). Smoke test: all finite, sane dists. **Causality test
  passed: corrupting bars > t left every feature ≤ t unchanged (max diff 0.0).** Swing
  pivots released with a confirm-lag; FVG/OB tracked forward-only.
- **Step 2 (iter 2)**: Built `scripts/transformer/build_tensor_panel_smc.py` — reuses the
  vetted build_tensor_panel helpers (load, z-score, align, macro, pivot) UNCHANGED, only
  appends PA features + writes to `data/transformer_panel_smc/` (production panel untouched).
  Smoke (6 tickers): 108 feats, alignment assertion OK, PA block 99% finite, mean|z| 0.499.
  Full 172-ticker build launched (PID 1556); watcher armed for completion. train.py given an
  env-var panel override (`TRANSFORMER_PANEL`, default unchanged) so it can train on either panel.
- **Step 3 (iter 3)**: Full SMC panel built — X_1h (4510,172,108), 641,942 labels, PA block
  99.3% finite. **Grids verified byte-identical to baseline** (ts_1h, Y_ret, sector_ids,
  tickers all match) → the ONLY difference is the 27 PA features = clean controlled experiment.
  Copied v10 pickmasks. GPU = RTX 5050. Launched matched listwise comparison
  (`scripts/transformer/run_smc_compare.sh`): {baseline, smc} × {long, short}, 15 epochs,
  seed 42 → `data/smc_compare_listwise.log`. Decisive metric = test rank-IC (ρ); baseline to
  beat ≈ 0.006 (DEAD).
- **Step 4 (iter 4) — VERDICT: DEAD-END.** Matched listwise TEST rank-IC: LONG −0.0041→+0.0014,
  SHORT +0.0058→+0.0066 (Δ +0.0008 = noise). Net@10 K=5: short −8.59→−9.35 (worse), long ≈−11bps.
  Only K=1 short marginally net-positive (~+0.7bps) in BOTH panels (pre-existing short signal, not PA)
  and collapses at K≥3. Cost-accounting clean (net−gross=−cost). **Explicit price-action geometry
  adds no tradable info at 1h → confirms info-ceiling** ([[project_pa_smc_transformer_deadend]]).
- **Gated GCN question (user)**: NOT justified — the PA features already encode the SMC relations and
  carry no edge; architecture can't manufacture absent information (cf. [[project_gate1_graph_features]]
  "Gate-2 GNN not justified"). Real lever = new data (order flow/tick/OI, [[project_oi_plus_paywall]]).
- **Reusable kept**: `scripts/features/price_action.py`, `build_tensor_panel_smc.py`,
  `data/transformer_panel_smc/`, `scripts/transformer/run_smc_compare.sh`, `data/smc_compare_listwise.log`.
- **User decision: BUILD the level-graph gated GNN.** Phase 2 begun (despite skeptical prior).
  - `scripts/structural/level_graph.py` — causal market-structure node extractor (NOW + S/R/OB/
    FVG/round nodes; nodes (T,K,13)); smoke + causality verified (max diff 0.0).
  - `scripts/structural/build_graph_panel.py` — node panel aligned 1:1 to transformer_panel_smc
    grid (present-mask agreement 100% on smoke) → `data/graph_panel_smc/` (4510,172,24,13).
  - `scripts/structural/gated_gcn.py` — edge-gated GCN (GGNN-style) over the level graph; NOW
    readout = structural token; SAME cross-sectional + macro + sector + objective as the
    transformer, so only the encoder differs. Forward/backward unit-test OK (130K params).
  - `scripts/structural/train_gcn.py` (+ `run_gcn_compare.sh`) — listwise train/eval with the
    same metrics + built-in negative control (mismatch structure↔label).
  - **GCN VERDICT — DEAD (cleanest evidence yet).** TEST rank-IC: short REAL +0.0032 (WORSE than
    +0.0066 baseline), long −0.0084; net-of-cost negative everywhere (short K5 −10.95). **Negative
    control decisive**: short NEG +0.0029 ≈ short REAL +0.0032 → the level graph contributes ~nothing;
    residual rho is sector/macro/cross-sectional baseline, not market structure. Cost-check clean.
    → A gated GCN does NOT rescue the SMC idea ([[project_pa_smc_transformer_deadend]]).

## ✅ Final verdict (whole conversation)
Explicit price-action/SMC information adds NO tradable edge at 1h — neither as transformer features
(rank-IC Δ≈noise) nor via a purpose-built level-graph gated GCN (real ≈ neg-control). Confirms the
repo-wide info-ceiling. The real lever is new DATA (order flow / tick / historical OI), not architecture.
Status → concluded for tradability. Reusable causal assets kept.

## 🔬 Phase 3 — closing the val→test gap (user /loop "get test rho nearing train")
Diagnostic (`scripts/transformer/analyze_rho_collapse.py`, baseline short): TRAIN rho +0.0276
(t=10.5), VAL +0.0222 (t=4.5), **TEST +0.0058 (t=1.05, n.s., CI spans 0)**; val−test drop
**significant** (Welch p=0.026). Small train→val gap ⇒ NOT overfit; significant val→test drop ⇒
**non-stationarity**. So the lever is recency, not regularization. Reframe: "test≈train" is not a
valid target (would imply leakage); goal = maximize robust TEST rho, while net-of-cost stays the
real bar (ρ≈0.02–0.03 is far sub-cost regardless).
- `scripts/transformer/wf_rho.py` (+ `run_wf.sh`) — expanding walk-forward: each test block trained
  on all history up to its start; reports per-fold train vs TEST rho + test net@10 K5 + t-stat.
- Smoke (2 folds, 1 epoch, short): WF mean TEST rho **+0.0101 > single-split +0.0058**; fold1 t=3.10
  (signif), fold2 t=1.79; net@10 K5 still ≈ −10.7bps (NOT tradable). Recency helps rho, not cost.
- Running full 5-fold×8-epoch short+long (`data/wf_rho.log`, bchcfl081).
- **WF RESULT (baseline):** SHORT mean train ρ +0.0230 → test ρ **+0.0142** (vs single-split 0.0058;
  folds 2-3 significant, fold3 test 0.0218≈train 0.0209), net@10K5 −10.6bps. LONG mean train −0.0129,
  test **−0.0108** (folds t−2.1..−2.9) = consistently INVERTED ranker (long-favored names underperform =
  inverse/reversion signal at model level, ties to the intraday edge). **Conclusion: recency closes the
  test-ρ gap (non-stationarity confirmed, fixable) but net stays −8..−11bps every fold/side → test-ρ-near-
  train achieved yet economically moot at 1h next-bar.** Running SMC-panel WF short (buvusrh6p) as the one
  remaining requested comparison; recommending we SKIP GCN-WF + recency sweeps (confirmatory nulls).
- **THE ACTUAL PAYOFF is elsewhere:** [[project_intraday_overnight_reversal_edge]] — first net-positive
  intraday edge (short overnight winners at 09:15 open → close, net +10.6bps@10bps k=10 t4.5, 2-dataset
  validated). The long-side inverse rho here corroborates that fade-the-strong thesis.

## 🔗 Core Memory Links
- [[project_dualres_transformer_result]] · [[project_sided_transformer_result]] · [[feedback_validate_cost_accounting]] · [[reference_ranking_data_conventions]]
