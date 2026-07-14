# 💬 Conversation Context: Dynamic Probability Floor — Dev/Holdout Validation

## 📌 Metadata
- **Conversation ID**: 5c1cad45-f346-41e0-995c-9f602d018842
- **Start Date**: 2026-07-12
- **Status**: 🔴 Concluded
- **Focus Area**: Research & Strategy Validation (dev/holdout discipline)

## 🎯 Objectives
- [x] Run the "Dynamic Probability Floor" short gate through the DEV window.
- [x] Confirm it once on the sealed HOLDOUT (proxy OOS).
- [x] Isolate the marginal value of the +0.028 macro tightening (static twin control).
- [x] Reconcile against the 2026-07-11 session's "72% WR / +45.98 bps" headline.
- [x] Replicate on the TRUE OOS (independent fresh-Upstox rebuild, 3rd tier).

## 💻 Active Code Files Modified
- [dyn_prob_floor_short.json](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/dev_holdout/configs/dyn_prob_floor_short.json)
- [static_floor_short.json](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/dev_holdout/configs/static_floor_short.json)
- [dyn_floor_lunch.json](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/dev_holdout/configs/dyn_floor_lunch.json)
- [HOLDOUT_LEDGER.md](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/dev_holdout/HOLDOUT_LEDGER.md)

## 📝 Compacted Session Log
- **Spec mapping**: The "Dynamic Probability Floor" is exactly baseline_213's SHORT leg — base floor = 99.92nd pct of raw `ss` (=0.0788 DEV-est ≈ shipped 0.082), penalty +0.028 (=0.110−0.082) triggered on SP500prev>+0.5% AND Nifty2h≥−0.10%. Encoded short-only (longs off, lunch off) to match the user's message literally.
- **DEV**: dyn +9.45 bps/trade (t=1.05, 198 tr); static twin +8.93 (t=1.09, 231 tr). Both insignificant; net-positive 9/11 months but Apr/May 2026 collapse. Neg-control ranking-edge NEGATIVE (−0.80) ⇒ pure pool-carving, no conviction-ranking value.
- **CORRECT 3-TIER SPLIT (`three_tier.py`):** the 11-month window (Aug'25→Jun'26) IS the model test block; BOTH dev + proxy-oos live INSIDE it, TRUE OOS is the fresh-pull month BEYOND it (July). My first pass wrongly ran proxy as Jun11→Jul10, absorbing July (true-OOS territory). Corrected:
  - **DEV** (Aug'25→May'26, inside 11mo): dyn +8.58 bps (t0.91, 185 tr) / static +8.16 — insignificant (t<1) from the start.
  - **PROXY OOS** (June 2026, inside 11mo): dyn **−8.15** (t−0.35, 25 tr) / static −9.61 — DEV edge does NOT replicate on the sealed within-frame month. ⚠️ boundary-sensitive: reserve only Jun11–30 as proxy → ≈−40 bps (early-June was a positive stretch) = window-dependent noise, no stable edge.
  - **TRUE OOS** (Jul1–10, beyond 11mo, fresh Upstox rebuild `true_oos_dyn_floor.py`): dyn **+14.69** (t0.55, 8 tr) / static identical +14.69 — POSITIVE but 8 trades t0.55 = statistically ≈0, can't conclude anything, and identical for static ⇒ says nothing about the +0.028 lever.
- **+0.028 tightening INERT at every tier** (dyn vs static ~equal). Full 7-gate stack on the fresh pipeline (`oos_jul10_backtest.py`) = −23.65% portfolio, short WR 43.8% vs 72% historical.
- **Headline reconciliation**: the 07-11 "72% WR / +45.98 bps" was in-sample whole-window on the SUSPECT `panel_backfilled.parquet` + extra lunch/nifty gates. Adding just `lunch_veto` to the clean floor: DEV +36.20 (t3.69) but derived holdout (all−dev, 144×27.67−131×36.20) ≈ **−58 bps / 13 tr** — the gate that manufactures the in-sample glory DEEPENS the OOS loss.
- **Root cause**: per-signal every short conviction band is net-negative (−6.9..−3.8 bps, info-ceiling); the floor selects the least-bad extreme tail (conv ≥0.089, disjoint from the inverted-U 0.04 cap → 0 trades). Gates extract a DEV regime artifact that reverts OOS.
- **Verdict**: DEAD. Do not deploy; do not burn more holdout on threshold variants. Winnable levers remain sizing/cost or new data (framework rule), and the certified above-cost generators: daily_macro_v2 LONG + gap-fade.

## 🔗 Core Memory Links & Backlinks
- [[04 — Research/Gate Dev-Holdout Validation Framework (2026-07)]]
- [[06 — Logs/Conversations/Conv-2026-07-11-Macro-Dynamic-Thresholds|Prior macro-thresholds session]]
- [[06 — Logs/Active Board]]
