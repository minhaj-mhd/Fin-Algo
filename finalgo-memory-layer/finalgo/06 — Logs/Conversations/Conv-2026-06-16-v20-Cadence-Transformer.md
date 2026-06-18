---
title: "Conv 2026-06-16 v20-Cadence Transformer"
type: log
status: concluded
updated: 2026-06-16
model: dualres-transformer
---

# 💬 Conversation Context: v20-Cadence Dual-Res Transformer + BCE Veto

## 📌 Metadata
- **Conversation ID**: 4a8ccffa-b6ee-4728-a511-25b5e5af0e09
- **Start Date**: 2026-06-16
- **Status**: 🔴 Concluded
- **Focus Area**: Model Suite / Research — transformer objective selection + veto layer design

## 🎯 Objectives
- [x] Determine best transformer objective for veto layer role (BCE vs listwise vs DualRes)
- [x] Train BCE transformer on v20 rolling-1h panel (`data/transformer_panel_v20`)
- [x] Run walk-forward veto eval: v20 XGB + BCE transformer veto on genuine OOS window
- [x] Record findings in Ray of Hope + research note

## 💻 Active Code Files Modified / Created
- [train.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/train.py) — run via `TRANSFORMER_PANEL=data/transformer_panel_v20 --objective bce`
- [v20_bce_veto_walkforward.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/v20_bce_veto_walkforward.py) — NEW: walk-forward veto eval script
- Artifact: `artifacts/dualres_transformer.pt` (BCE model)
- Artifact: `artifacts/v20_bce_veto_walkforward.json` (full results)

## 📝 Compacted Session Log

- **Objective selection**: BCE chosen over listwise for veto role — listwise LONG had negative val rho
  across all 15 epochs (anti-ranked), BCE LONG test rho +0.020 (best). BCE outputs calibrated P(winner)
  → natural threshold semantics for veto (P < threshold → reject trade).

- **All 3 objectives compared on v20 panel** (LONG K=1 gross): BCE +4.52 bps, Listwise +4.88 bps,
  DualRes ~+4.5 bps — identical information ceiling ~AUC 0.520 / rho 0.020 across all objectives.

- **BCE training on v20 panel**: AUC 0.5264 val (best e6), AUC 0.5204 test. LONG K=1 gross +4.52 bps,
  SHORT K=1 gross +3.23 bps. All net-negative at 6 bps+ cost. Saved `artifacts/dualres_transformer.pt`.

- **Walk-forward veto eval** (`v20_bce_veto_walkforward.py`):
  - OOS: Sep 2025 → Jun 2026, 2,616 timestamps, 447k joined rows
  - Bugs fixed: XGBRanker predict() wrapping, ts_ns µs→ns ×1000, .NS ticker stripping
  - **K=3 LONG @ th=0.50 @6bps**: ALL=−3.56, KEPT=−2.43, VETOED=−5.65
    **Δnet = +1.14 bps, t = +2.27, CI [+0.2, +2.1]** ✅ CI lo > 0, neg-ctrl OK
  - K=1, K=5 LONG: positive Δ but CI crosses 0 (not significant)
  - SHORT: no significant uplift at any K/threshold
  - Pre-registered WIN (kept_net ≥ +1 bps): 0 hits — v20 XGB itself net-negative in this OOS period

- **Verdict**: Veto has **real, statistically significant discriminative power** on LONG K=3.
  WIN condition fails only because v20's base gross edge (~3.5 bps) is below the 6 bps cost floor in
  this period — not a transformer failure. Recorded in Ray of Hope Tier 3.

- **Path to net-positive**: reduce effective cost to ≤ 3 bps (limit orders). At that threshold,
  KEPT gross ~3.5 bps would cross net-positive with the veto in place.

## 🔗 Core Memory Links & Backlinks
- [[00 — Start Here/Ray of Hope]] — added to Tier 3
- [[04 — Research/BCE-Transformer-V20-Veto]] — new research note
- [[02 — Models/DualRes Transformer]] — BCE objective result logged
- [[project_v20_rolling_1h_result]] [[feedback_validate_cost_accounting]]
