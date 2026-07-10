# 💬 Conversation Context: 2-Hour Horizon Models (v20 XGBoost + Transformer)

## 📌 Metadata
- **Conversation ID**: e5d0dcf8-d1d0-4c42-bc4a-870b1aa38fb6
- **Start Date**: 2026-06-30
- **Status**: 🔴 Concluded
- **Focus Area**: Models / Research (rolling-1h ranker @ 2h holding horizon)

## 🎯 Objectives
- [x] Build `Next_2Hour_Return` label on the v21 rolling-1h panel (same features, longer horizon — no feature-window change).
- [x] Check 2-hour holding-period returns on the existing v20/v21 (1h-trained) ranker → gross does NOT compound (≈ same as 1h), still sub-cost.
- [x] Train + eval a 2-hour XGBoost model (v20 recipe, 2h label) → ρ L 0.0202 / S 0.0185, ~3.4bps/side gross, net −6.6 to −7.8bps.
- [x] Train + eval a 2-hour DualRes transformer → ties trees long, loses short; neg-control clean; not justified.

## 🧪 Design (why same features, vary horizon)
- The binding cost is **10bps round-trip regardless of hold length**. At 1h the gross edge is ~4bps/side ⇒ sub-cost. A 2h hold should ~2× the gross (if signal persists) while cost stays fixed ⇒ cost-per-unit-time halves. This is the one clean lever that could cross the cost line without new data.
- To isolate the **horizon** effect, keep the proven v20/v21 rolling-1h FEATURES unchanged and change ONLY the prediction horizon/label. A 2h feature window would confound the comparison.
- RESEARCH ONLY (AGENTS.md): overlapping windows ⇒ effective N ~1/4; point estimates, no t-tests, no Gauntlet, no registry.

## 💻 Active Code Files Modified / Added
- [build_2h_labels.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/build_2h_labels.py) — adds `Next_2Hour_Return` to the v21 panel + builds `Y_ret_2h.npy` for the transformer panel.
- [eval_2h_v20.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/eval_2h_v20.py) — holding-period test (1h-trained, held 1h vs 2h) + 2h-trained model, purged monthly WF.
- [train.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/train.py) — minimal: `TRANSFORMER_LABEL` env var to swap the label tensor.

## 📝 Compacted Session Log
- **Initial Analysis**: v21 panel (1.9M rows, 110 universe, window-closes 10:15→14:30 for 1h labels; full close series to 15:30). 1h transformer baseline ρ L 0.0041 / S 0.0087 vs v21 XGB L 0.0223 / S 0.0192. GPU = RTX 5050.
- **2h label built** (`build_2h_labels.py`): `Next_2Hour_Return` on the identical window-close grid (1h-return reconstruction self-check max|diff| = 1.4e-08 ⇒ byte-identical grid). 2h labels cover window-closes 10:15→13:30 (77.8% of panel rows). Also emitted `Y_ret_2h.npy` (1.49M finite) for the transformer panel.
- **XGBoost 2h eval** (`eval_2h_v20.py`, purged monthly WF, 8 folds, identical both-labeled test entries, cost 10bps, net bps):
  - LONG  K3: A) 1h-model/held-1h gross **+3.78** net −6.22 | B) 1h-model/held-2h gross **+3.48** net −6.52 | C) 2h-model/held-2h gross +2.23 net −7.77 | D) neg-ctrl ρ −0.015 (collapses ✓)
  - SHORT K3: A) gross −0.40 net −10.40 | B) gross +0.49 net −9.51 | C) 2h-model gross **+3.36** net −6.64 | D) neg-ctrl ρ −0.005 (collapses ✓)
  - **READ**: the gross edge does NOT compound with a longer hold — long K3 gross is flat ~3.5bps whether held 1h or 2h (the predictable move is in the FIRST hour; the 2nd hour adds noise/reversion, no return). Short benefits from training on the 2h label (gross −0.40→+3.36) but caps at the same ~3.4bps/side ceiling. ALL variants remain deeply net-negative at 10bps. **Refutes the "longer hold amortizes fixed cost" hypothesis** (gross would need to ~2x; it doesn't grow at all). No leak (both neg-controls collapse).
- **2h DualRes transformer** (`train.py --objective listwise` on `Y_ret_2h`; long+short+long-shuffle, ~20min/side on RTX 5050; horizon-tagged artifacts `dualres_{side}_2h*`):
  - test ρ: long **0.0144** (val_best ~0, faint), short **0.0058**; both net-negative (long K3 −3.17, short K3 −10.88 @10bps).
  - shuffle neg-control (long): test ρ **+0.0008** ≈ 0 ⇒ no leak; the 0.0144 is faint real signal.
  - **same-window head-to-head** (`benchmark_2h.py`, chrono test 2025-10→2026-06): XGBoost 2h L 0.0146 / S 0.0105 vs transformer L 0.0144 / S 0.0058 ⇒ transformer **TIES** XGBoost long, **LOSES** short. Does NOT justify the transformer. (Nuance: at 1h the transformer was ~5× worse on long, 0.0041 vs 0.0223; at 2h the long gap CLOSED — longer horizon is relatively easier for the transformer — but it still only ties the tree and loses on short.)

## 📝 3-HOUR follow-up (user asked "what about 3 hour?")
Parametrized the 3 scripts by `--hours` (kept 2h work intact) and ran the 3h pass. 3h labels cover window-closes 10:15→12:30 (55% of rows; self-check 1.4e-08).
- **XGBoost** (`eval_2h_v20.py --hours 3`, purged WF, K3 net @10bps): LONG A) held-1h gross +3.40 net −6.60 | B) held-3h gross **+3.68** net −6.32 | C) 3h-model gross +3.22 net −6.78 | D) neg-ctrl ρ +0.009. SHORT: A) gross +2.01 net −7.99 | B) held-3h gross +1.52 net −8.48 | C) 3h-model ρ 0.0203 gross +2.20 net −7.80 | D) neg-ctrl ρ −0.004. ⇒ rho FLAT across 1h/2h/3h (~0.02–0.028); gross does NOT scale with hold; net stays −6 to −8bps. Nuance: the 3h LONG shuffle gross is elevated (+1.15) = **market drift over the longer window, not skill** (rho collapses) — so longer "gross" increasingly buys drift, not edge.
- **Transformer 3h** (listwise): test ρ L 0.0070 / S 0.0065; shuffle neg-ctrl L +0.0018 ≈0. Same-window benchmark: XGB 3h L0.0137/S0.0144 vs TF L0.0070/S0.0065 ⇒ TF **loses BOTH sides**. (So the 2h long "tie" was a fluke; trees win outright at 1h and 3h.)

## 🩹 STOP-LOSS follow-up (user: "most trades make money, cut the huge losses → profitable?")
Researched whether cutting the loss tail fixes the sub-cost book. Built an honest INTRABAR trade panel (`build_stop_panel.py` → `data/research/stop_research/trade_path_panel.parquet`, 140k WF-OOS Top-10 legs incl. a random-basket control, 15-min low/high path out to 3h) and swept fixed + trailing stops (`stop_intrabar_sweep.py`); also a fast close-checkpoint pass on the dualtf panel (`stop_loss_sweep.py`).
- **Premise half-right:** GROSS WR is >50% (long 51.7% / short 52.9%) so "most trades make money" is true *before cost*; worst-5% carries ~all the net loss and zeroing it (hindsight) → ~break-even. BUT wins≈losses (|loser|/|winner| 0.98 long, 1.10 short) → near-symmetric, NOT "small wins + huge losses"; and NET WR <50% (cost flips it).
- **A stop can't harvest it:** honest intrabar sweep → EVERY width/side/horizon stays firmly net-NEGATIVE (best 3h-short −0.3% = −7bps, stops 70%); 1h-long stops make it WORSE. Decomp: clip≈save (mean-reversion → stops realize recoverable dips). RANDOM book improves the SAME (deleveraging, not alpha); model's ~+1.2–1.6bps edge is at full-hold and stops ERODE it. The earlier close-checkpoint "+1.56bps" was a 15-min-resolution artifact (−0.3% fires 33% on closes vs 53% on intrabar lows).
- **VERDICT:** cutting losses with a stop does NOT make it profitable. Lever = cut the 10bps COST (limit/passive execution) or a disaster-classifier (meta-label), NOT a price stop. See memory `project_stop_loss_research`.

## 📅 POSITIVE vs NEGATIVE DAY detection (user: separate good/bad days, can we detect/correlate?)
Built a contiguous OOS daily P&L series (`build_daily_pnl.py` → `data/research/regime_days/daily_pnl.csv`, 614 days, expanding-WF Top-10 1h book) and scanned ex-ante predictors (`research_regime_days.py`).
- **Separation:** daily net mean −8.73bps, std 5.86; only **5.2% NET-positive days** (cost sinks the ~360-leg daily average), 60% gross-positive.
- **Undetectable:** autocorr≈0 (no clustering); 28 macro (lag-1, ex-ante) + dow + own-lag → max |Spearman| 0.086 = noise; OOS day-gate rank-IC **+0.034**, best top-30% predicted days still −8.0bps net (vs random −8.75). Long/short daily P&L **−0.755** corr (per-side = market-direction bet); the one in-sample hint (market-trend→side, rho~0.10) DIES OOS (side-timing rank-IC +0.033, timed −8.60 < always-long −7.96).
- **VERDICT:** daily P&L sign is unpredictable ex-ante from price/macro/calendar; net-negative is the 10bps COST, not a detectable bad-day regime → "skip bad days" can't work. See memory `project_regime_day_detection`.

## 🏁 VERDICT (horizon study — research, no Gauntlet)
**NO holding horizon (1h/2h/3h) crosses the cost line.** Cross-sectional skill (rho) is horizon-invariant at ~0.02–0.028; skill-gross does not scale with hold time (the longer hold ties up capital N× for ~0 extra skill-return, the inflating "gross" is market drift); net Top-K stays −6 to −8bps both sides at every horizon at 10bps. The transformer is dominated by the tree at every horizon. **Holding horizon is NOT a lever — the 1h info-ceiling is horizon-invariant.** Best actual intraday trade remains [[project_intraday_overnight_reversal_edge|the 09:15 overnight-reversal short]]. Reusable (horizon-parametrized): `build_2h_labels.py --hours N`, `eval_2h_v20.py --hours N`, `benchmark_2h.py` (reads horizon from `TRANSFORMER_LABEL`), `train.py TRANSFORMER_LABEL=Y_ret_Nh`; panels `panel_{2,3}h.parquet`, tensors `Y_ret_{2,3}h.npy`.

## 🔗 Core Memory Links & Backlinks
- [[06 — Logs/Daily Logs/2026-06-28]] (v21 clean rebuild)
- Related: project_v20_rolling_1h_result, project_v21_clean_rebuild, project_intraday_overnight_reversal_edge (the only net-positive intraday edge so far)
