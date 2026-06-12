# 💬 Conversation Context: Sophisticated Cross-Sectional + Temporal Transformer

## 📌 Metadata
- **Conversation ID**: (loop session, 2026-06-12)
- **Start Date**: 2026-06-12
- **Status**: 🟢 Active
- **Focus Area**: Model Suite — sequence transformer (directional next-candle + confidence)

## 🎯 Objectives
- [ ] Preliminary survey of ALL data feedable to the transformer (DONE — see log)
- [ ] Decide scope/granularity given strong prior negative evidence
- [ ] Build dual-input (temporal sequence + cross-sectional snapshot) transformer
- [ ] GPU training (RTX 5050, 8 GB, CUDA confirmed)
- [ ] Honest cheap falsification BEFORE any expensive Gauntlet run

## 💻 Active Code Files Modified
- (none yet — research phase)

## 📝 Compacted Session Log
- **Request**: Build "the most sophisticated transformer": inputs = last 30×1h candles + volume
  per ticker (temporal self-attention) + last candle of the other 172 tickers (cross-sectional)
  + sector labels + index + VIX, predict next candle long/short + confidence. Maybe add 15m.
- **GPU**: RTX 5050 Laptop, 8151 MiB VRAM, torch 2.11.0+cu128, `cuda.is_available()=True`. Usable
  but VRAM-constrained → modest d_model/batch.
- **Data inventory (what we can actually feed):**
  - Intraday per-ticker OHLCV + ~90 engineered features, 172 tickers, 3y:
    1h (`ranking_data_upstox_1h_3y_clean.csv`, 1.1 GB, 95 cols), 15m (5.3 GB), 30m_v3 (2.7 GB),
    5m_v3 (being collected now). Raw caches: `raw_upstox_cache_1h_v3/` (172 files) etc.
  - Cross-sectional: all 172 tickers aligned; `scripts/sector_map.py` → 16 sectors.
  - Index intraday: `raw_index_cache/nifty500_1h.csv` (⚠️ volume & oi == 0; price only).
  - Macro/global = **DAILY ONLY** (`raw_global_daily/`: SP500, NASDAQ, NIKKEI, HSI, DXY, USDINR,
    BRENT, GOLD, US10Y) and VIX_Level/Change/Percentile + breadth all live only in the DAILY
    fused set `ranking_data_daily_macro_v3.csv` (93 cols). No intraday VIX/macro collected.
- **⚠️ Prior negative evidence this proposal collides with (re-verified, not assumed):**
  - CST Stage-0 killed: `data/research/stage0_leadlag_result.txt` re-read → AUG−BASE rho
    long −0.0028 / short −0.0007 ("CST dead on arrival → redirect to order-flow/microstructure").
    See [[project_cst_stage0_killed]].
  - Directional next-candle classification (v18 XGB / v19 CatBoost) confirmed coin-flip /
    net-negative: [[project_v18_v19_directional_deadend]].
  - 1h price/volume has no post-cost edge: [[project_tbm_1h_ensemble_results]],
    [[project_v8_1h_walkforward_demoted]].
  - The exact macro/VIX/sector/global fusion the user wants ALREADY exists at daily granularity →
    daily_xgb graded DEAD/DEAD by Gauntlet: [[project_validation_gauntlet]].
  - Cost-accounting discipline: [[feedback_validate_cost_accounting]] (RAW vs NET per side).
- **Genuine novelty (why it's not a pure repeat):** nobody has trained a from-scratch end-to-end
  temporal+cross-sectional SEQUENCE transformer — prior CST test only bolted 11 lead-lag features
  onto the existing tree ranker. Intraday technicals × macro fusion also untested at intraday res.
- **DECISION (user, 2026-06-12)**: **Full sophisticated build now** (not the cheap probe) +
  **1h + 15m fused** input. Accept GPU hours + one pre-registered Gauntlet run. Build the most
  capable version, then Gauntlet it. Honest cost-accounting (RAW vs NET per side) is mandatory.
- **Architecture plan (V1)**: dual-resolution temporal encoders (30×1h seq + 60×15m seq, each a
  Transformer encoder w/ positional enc + CLS pool) ‖ cross-sectional encoder (self-attention over
  172 tickers' latest candle + sector embedding) ‖ daily macro/VIX/breadth MLP → fusion MLP →
  binary next-1h-candle logit (long/short) + calibrated confidence. Train on GPU (RTX 5050).

## 🛠️ Build progress (2026-06-12)
- **Panel built** → `data/transformer_panel/` via [build_tensor_panel.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/build_tensor_panel.py):
  X_1h (4510,172,81), X_15m (18937,172,81), single 15m source (reproduces rebuild_aligned_datasets),
  per-query z-scored, **14:15 (2:15–3:15) context candle retained** (slot 5, label NaN), clock-time
  slots emitted (1h 0–5, 15m 0–24). Alignment assertion passed (15m close-time == 1h close-time).
- **Model** [model.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/model.py):
  `DualResCSTransformer` — dual temporal encoders (sinusoidal pos-enc **+ learned time-of-day slot
  emb** so candles are identified by clock time) ‖ cross-sectional encoder over 172 tickers ‖ macro
  FiLM ‖ sector emb → per-ticker P(up). ~233k params @ d_model 64; running d_model 96.
- **Trainer** [train.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/train.py):
  chronological train/val/test (70/15/15, embargo 30), GPU+AMP, AdamW+cosine, early-stop on val AUC.
  Cost-aware eval: Top-1/3/5 per side NET @ 6 & 10 bps; **cost-accounting sanity passed exactly**
  (median(net−gross) == −cost per side → no cost-sign bug, cf. [[feedback_validate_cost_accounting]]).
- **Smoke test (1 epoch, untrained baseline)**: val AUC 0.508; all sides net-negative (expected).
  Full 40-epoch run **in progress** on RTX 5050.
- **GPU confirmed working**: 25–45 s/epoch, CUDA + AMP.

## 📊 RESULT — full GPU run (2026-06-12, exploratory, ⚠️ NO Gauntlet verdict)
- Objective confirmed = **binary next-1h direction** (BCEWithLogits on `Next_Hour_Return>0`),
  per-ticker P(up)=confidence. NOT ranking, NOT magnitude regression. In-flight run was valid.
- Early-stopped; **val AUC 0.526 ≈ TEST AUC 0.521**, test acc 51.6% vs base 49.3% (+2.3pp).
- Cost-aware (binding 10 bps): best Top-1 LONG **gross +6.4 bps → net +0.4 @6bps, −3.6 @10bps**;
  Top-3/5 and short side net-negative. Cost-accounting sanity passed exactly (no sign bug).
- **VERDICT (exploratory): faint real directional signal, gross-positive, but sub-cost** — matches
  the documented dead-end. Artifacts: `artifacts/dualres_transformer.pt` + `_metrics.json`.

## 💸 COST-AWARE OBJECTIVE TEST ("train it to beat 10 bps") — CONCLUSIVE
- Added net-PnL loss `−mean(pos·r − cost·|pos|)`, `pos=2σ(logit)−1` (no-trade at pos→0), early-stop on
  val net-PnL@10. `train.py --objective netpnl --cost_bps 10`.
- **Result: model VOLUNTARILY ABSTAINS** — test `deploy=0.002` (0.2% position), `netPnL@10=−0.022 bps`,
  train loss→0.0000. Every epoch deploy∈[0.002,0.015]. AUC unchanged ~0.513.
- **Why magnitude isn't the lever**: 86% of 1h bars already move >6bps, 78% >10bps → moves are big enough;
  to net-beat 10bps you need ~63–69% directional accuracy, we have 51.6%. The gap is DIRECTION
  (information), not selection. A cost-aware loss free to abstain → abstains. Matches TBM 1h DEAD/DEAD.
- **Bottom line (3 independent tests agree)**: BCE direction (gross<cost), selective-by-confidence
  (tail ~3.4bps gross), and cost-aware net-PnL (deploys ~0) all say: **no cost-beating 1h OHLCV edge.**

## 🛡️ v10 + TRANSFORMER VETO WALK-FORWARD (scripts/transformer/veto_walkforward.py)
- Setup: v10 1h ranker cached WF OOS preds (retrained per fold) Top-1/3/5; transformer vetoes picks it
  disagrees with (long if P(up)>0.5, short if P(up)<0.5). Eval on transformer-OOS dates only
  (Aug2025–May2026, 104,272 rows) → both models OOS. Honest.
- **BCE veto**: probs ~0.5 → vetoes ~8% → ~no effect. Useless as a veto.
- **netPnL veto**: cuts ~70% of longs / ~25% shorts; improves net in EVERY cell (filter value), e.g.
  Top-1 SHORT net@10 −1.1→+1.4, Top-3 SHORT −8.1→−5.2, Top-5 SHORT −7.3→−5.4, trims dead longs.
- **But nothing significantly net-positive @10bps** (best Top-1 SHORT+veto +1.4bps t=0.37 = noise).
  Short side is the only live one (v10 raw short +8.9bps win56%); cost eats it. = FILTER_GRADE confirmed.

## 🚩 20bps "target" probe — small-sample/fat-tail ARTIFACT (not edge)
- Tightening netPnL short-veto (P_up<0.45, Top-3) → ~15 trades/10mo showing +73bps raw / +53 net@20 /
  t=2.44. Fragility check: 15 trades, 9 days; 5 biggest (+158/+237/+167/+239/+179) = 89% of return;
  one day (2026-05-29) = 48%. Cherry-picked from 10-config sweep (multiple testing); t-test invalid
  under +240bps outliers; contradicts global AUC 0.52. Raising cost target = HARDER bar, not easier.
  Do NOT treat as a result — classic Gauntlet-killable artifact.

## 🔍 SKEPTICAL AUDIT (scripts/transformer/audit.py) — corrections to earlier claims
- **Real, not faked**: AUC 0.521 significant (timestamp-cluster bootstrap CI [0.5156,0.5264], excludes
  0.5); negative control clean (shuffle within-hour → AUC 0.502 → no leakage); beats all trivial
  baselines (momentum/reversal/always-short all −4 to −15 bps net).
- **Overclaims fixed**: (a) gross was drift-inflated — always-long gross +1.69 bps/bar, so "+6.4 bps long"
  is really ~+4.7 skill-over-drift; (b) net@10 CIs: LONG [−8.2,+1.0] (straddles 0, not sig profitable),
  SHORT [−12.6,−2.8] (sig neg); (c) cost-aware deploy→0 has a trivial pos=0 optimum → corroborating not
  conclusive; (d) single chronological split, NOT purged WF (the v8 trap) → not Gauntlet-certified.
- Net verdict (sub-cost) stands; confidence + gross framing were too strong.

## 📦 DATASET SUFFICIENCY (evidence, not vibes)
- 641,942 labeled samples / 3,732 decision hours / 3.4y / 81 feats / dual-res. Enough to TRAIN
  (clean convergence, val≈test → not starved, not catastrophically overfit).
- BUT effective N << raw: cross-sectional ρ=0.196 → only **~5 independent names/hour** (of 172);
  next-hour return lag-1 autocorr +0.012 (labels ≈ white noise). Effective budget ~18k, not 642k.
- Repo's own data-size ablation: 1h IC **flat** from 100%→10% data (0.0287→0.0265) →
  **information-limited, not data-limited.** More of the same data won't beat costs.
- Only lever that would move it: order-flow/microstructure (cf. [[project_cst_stage0_killed]]).

## 🔗 Core Memory Links & Backlinks
- Architecture: [[02. Model Suite/DualRes-CrossSectional-Transformer-Architecture]]
- [[02. Model Suite/Cross-Sectional Transformer Architecture Proposal]]
- [[08. Model Analysis/15-Minute Vanguard Model/Dual-TF Entry-Exit Overlay Research]]
- Gauntlet discipline: [[project_validation_gauntlet]]
