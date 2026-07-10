---
title: Kronos Zero-Shot Veto (2026-07)
type: research
status: dead
updated: 2026-07-03
model: kronos-base (external foundation model, zero-shot)
verdict: DEAD as backtest edge — ⚠️ UNVERIFIED by Gauntlet (exploratory tier only, no run_id; pre-registered kill criteria applied instead)
---

# Kronos Zero-Shot Veto — DEAD (backtest); live shadow-counterfactual experiment continues

**Verdict (2026-07-03, final pre-registered eval on the complete post-cutoff window):**
zero-shot Kronos-base (NeoQuasar, 102.3M, 12B-bar/45-exchange pretraining) adds **no
significant net uplift** as a veto on the 1h ranker book. The one hopeful interim cell
regressed exactly like noise does.

## Design (pre-registered before scoring; vault conv log has the full trail)
- Host book: `data/research/entry_exit/dualtf_trade_panel.csv` (13,020 OOS walk-forward
  1h top-3 trades, 2024-09→2026-06; host model retrained per fold — verified from
  `build_dualtf_panel.py`, not assumed).
- Score: p_up = fraction of R=30 sampled OHLCV forecast paths (480×15m-bar context,
  pred_len=4, T=1.0, top_p=0.9) ending above entry. Side-aligned; coverage-matched veto.
- Leakage split at **2025-09-09** (HF weight upload date). POST window = honest read.
- Kill criteria: Δnet(keep-70@10bps)>0 with day-clustered bootstrap t>2 on ≥1 side, AND
  neg-control (within-dt1 shuffle) does not reproduce it, AND sign holds at keep-50.

## Result (post-cutoff, n=4,776 trades; scripts + full tables in repo)
| cell | Δnet @10bps | boot t | verdict |
|---|---|---|---|
| LONG keep-70 (primary) | **+1.03bps** | **+1.34** | fails t>2; CI[−0.45,+2.58] straddles 0 |
| SHORT keep-70 (primary) | −0.56bps | −0.52 | negative |
| LONG keep-50 (secondary) | +0.50bps | +0.46 | fades with tightness |
| SHORT keep-50 (secondary) | −1.40bps | −0.79 | negative |

- Interim (n=3,525) long cell was +1.73 t=2.02 → regressed to +1.03 t=1.34 at full n. 
- PRE-cutoff (possible pretraining overlap) shows 2–4× larger "uplift" (long +2.7 t2.2,
  short +4.3 t2.1) **but the neg-controls reproduce it** (nulls +1.8 / +4.7): timing +
  probable memorization, not selection.
- Supporting diagnostics, all pointing the same way: gross of the kept book **never
  crosses +6bps at any coverage** (peak +4.1bps @keep-70 long, then INVERTS — top-decile
  Kronos-conviction trades are net-negative); 25bps predicted-magnitude veto makes the
  book WORSE (−2.7bps); p_up is badly **miscalibrated** (Brier 0.348 vs 0.250 base-rate;
  realized up-rate flat ~48% across all predicted bins); rank-IC of the aligned score
  ρ≈0.01–0.02 post-cutoff — at/below the familiar 1h info-ceiling.

## Interpretation
The cross-market pretrained prior was the last untested "free data" lever for the 1h
book. It sees the same OHLCV our from-scratch models saw and finds the same ~ρ0.02–0.03
ceiling — consistent with [[Dead-Ends Register]] entries for co-sign v10, daily veto,
BCE veto, CST, DualRes, PA/SMC. The lever remains **new data (order-flow /
microstructure) or execution (gap-fade fills)**, not architecture or pretraining.

## Live status (separate experiment, not a verdict input)
Deployed in Vanguard per user decision BEFORE the final verdict landed: keep-50%
thresholds (LONG p_up≥0.50 / SHORT 1−p_up≥0.70), **enforce mode**, ahead of Gemini
S1/S2; dashboard "Kronos Avoided" chip tracks the counterfactual P&L of blocked trades.
Backtest expectation at this operating point: ≈0 to slightly negative. Review the chip
after ~2 weeks; revert with `KRONOS_VETO_MODE=shadow`.

## Artifacts
- Scoring/eval: `scripts/research/kronos_veto_score.py`, `scripts/research/kronos_veto_eval.py`
- Scores (7,176 trades incl. 2,405 pre-cutoff): `data/research/kronos_veto/scores.csv`
- Tables: `data/research/kronos_veto/results_2026-07-03.txt`, `artifacts/kronos_veto.json`
- Live layer: `scripts/vanguard/kronos_veto.py`, log `data/kronos_veto_live.jsonl`
- Conv log: [[06 — Logs/Conversations/Conv-2026-07-02-Kronos-Zero-Shot-Veto|Conversation Log]]
