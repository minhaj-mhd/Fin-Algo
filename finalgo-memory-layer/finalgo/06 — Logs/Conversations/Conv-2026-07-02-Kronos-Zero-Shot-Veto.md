---
title: Conv 2026-07-02 Kronos Zero-Shot Veto
type: log
status: active
updated: 2026-07-02
model: kronos-base (external, zero-shot)
---

# 💬 Conversation Context: Kronos Zero-Shot Veto Backtest (1h book)

## 📌 Metadata
- **Conversation ID**: 50c37bfb-0923-4fd2-a6cc-7b092da75a10
- **Start Date**: 2026-07-02
- **Status**: 🟢 Active
- **Focus Area**: Research — external foundation-model veto overlay

## 🎯 Objectives
- [x] Assess Kronos (github.com/shiyu-coder/Kronos) for use in this repo
- [x] Zero-shot Kronos-base veto backtest on the 1h ranker book (dualtf panel)
- [x] Verdict vs pre-registered kill criteria, archived either way → **DEAD** (see [[04 — Research/Kronos Zero-Shot Veto (2026-07)]])
- [x] Live deployment as Vanguard veto layer (user decision; enforce + dashboard tracking)
- [ ] Review live "Kronos Avoided" counterfactual after ~2 weeks (~2026-07-17)

## 🧪 Pre-registered hypothesis (logged BEFORE the run)
*Zero-shot Kronos-small next-hour forecasts carry veto-grade information on the 1h book's top-K picks that our from-scratch models (co-sign v10, daily veto, BCE veto — all dead) did not. Only new ingredient: 12B-bar / 45-exchange pretrained prior.*

- **Primary cell**: coverage-matched keep-70% by side-aligned score, Δnet(KEPT−ALL) @10bps, day-clustered bootstrap t.
- **PASS iff** Δnet>0 with t>2 on ≥1 side, AND within-timestamp score-shuffle neg-control shows no comparable uplift, AND sign holds at keep-50%, AND effect exists post-2025-09-09 (HF weight upload date; earlier trades may sit inside Kronos's pretraining corpus → memorization leakage).
- **Fixed spec, no sweeps**: **Kronos-base (102.3M)** + Tokenizer-base (HF rev of 2025-09-09), 15m bars, lookback=480 fixed, pred_len=4, T=1.0, top_p=0.9, R=30 independent sample paths (predict_batch averages over sample_count internally, so each trade is duplicated R× with sample_count=1 — compute-identical).
- *Amendment 2026-07-03 (pre-run, before any scoring)*: model upgraded small→base per user /goal; smoke test had validated the pipeline with small; no score data existed at amendment time, so this is a pre-registration change, not a sweep.
- ⚠️ **Exploratory only — NO Gauntlet verdict authority. No fine-tuning. No threshold sweeps.**

## 💻 Active Code Files Modified
- [kronos_veto_score.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/kronos_veto_score.py) (new — GPU scoring pass)
- [kronos_veto_eval.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/kronos_veto_eval.py) (new — veto metrics + neg-control + leakage split)
- Kronos clone (external, read-only dependency): `C:/Users/loq/Desktop/Trading/Kronos` (upstream shiyu-coder/Kronos, shallow, NOT in this repo)

## 📝 Compacted Session Log
- **Initial Analysis**: Kronos = open-source K-line foundation model (decoder-only AR transformer over hierarchically-quantized OHLCV tokens; pretrained 12B bars / 45 exchanges; paper arXiv 2508.02739). User's fork is an unmodified snapshot; using upstream latest per user decision.
- **Step 1 — Security review before execution**: `model/` package (3 files) has clean imports (numpy/pandas/torch/einops/tqdm/hf_hub), no exec/subprocess/network beyond HF download; weights are safetensors (no pickle code-exec risk). Cleared to run.
- **Step 2 — Env**: installed einops 0.8.1, huggingface_hub 0.33.1, safetensors 0.6.2 only; pinned torch 2.11.0+cu128 / pandas 3.0.x / numpy 2.4.4 untouched (Kronos pins pandas 2.2.2 — will patch local clone if 3.x trips).
- **Step 3 — Design**: host trade list = `data/research/entry_exit/dualtf_trade_panel.csv` (13,020 OOS WF trades 2024-09→2026-06, no-look-ahead by construction); context = 15m cache bars labeled ≤ dt1+45 (left-labeled ⇒ closes at entry dt1+60); veto metrics mirror `veto_lib.py` coverage-matched KEPT-vs-ALL with day-clustered bootstrap; cost 10bps round-trip once per trade.
- **Step 4 — Smoke test (Kronos-small, 2026-07-02)**: GPU forecast sane (0.77% dev from last close), 1.6s/call. Pipeline validated before the model-size amendment.
- **Step 5 — Batch calibration (2026-07-03)**: 8GB RTX 5050 shared with the LIVE `vanguard_signal_engine.py` (PID holding CUDA since 07-02 — left untouched). Effective batch 360 spills into Windows shared memory (24 trades / 36 min — unusable); batch 4 (eff 120) is the sweet spot: 72 trades / 3.1 min ≈ 2.6 s/trade; batch 8 spills again. Full pass = 12,999 distinct (ticker,dt1) contexts ≈ 9.5 h, launched resumable (`--batch 4`), scores → `data/research/kronos_veto/scores.csv`.
- **Step 6 — Eval dry-run on 96 partial rows**: code path validated (cost sanity assert passes; day-clustered bootstrap + neg-control wired). Partial outputs deleted — numbers meaningless at that n. Known small-sample artifact understood: neg-control is degenerate while scored trades are ticker-sorted (1 trade/dt1 ⇒ identity permutation); must de-degenerate on full data — verify.
- **Data quality note**: ~10% of trades skip with `no_align` (missing 15m bar at dt1+45 in the 3y cache); skipped conservatively, counts reported by the scorer.
- **Step 7b — Host-model OOS verification (user challenge, 2026-07-03)**: re-derived from `build_dualtf_panel.py` (not assumed): expanding-window monthly walk-forward, XGB rankers RETRAINED per fold on months strictly before a val month strictly before the 4 disjoint test months; sL/sS predicted only on test months ⇒ every panel trade's host score is from a model that never saw that month. Residual caveat: v10 HPs/feature list from `models/v10_native_1h/metadata.json` were selected historically (second-order HP-selection leakage, shared by all panel consumers; HP tuning is info-ceiling-bound per 2026-06-17 study, so low risk).
- **Step 7 — Interim checkpoint, declared BEFORE any look (2026-07-03, user asked for a smaller test first)**: scoring queue reordered to POST-cutoff-first, seeded-random within window (random prefix = unbiased subsample; the old alphabetical order made partials unreadable). Post-cutoff universe = 5,424 contexts (of 12,999); 640 were already scored under the old alphabetical order (ticker-biased composition — noted, ~21% of the interim sample). **Interim rule**: one look when post-cutoff scored ≥ 3,000. Directional only: if Δnet(keep70@10bps) ≤ +1bps on BOTH sides or t<1 on both → early kill (stop scoring, record dead-end). Otherwise continue to full post-cutoff for the pre-registered primary cell; pre-cutoff continues after, for the leakage contrast only. No other interim looks.

- **Step 8 — INTERIM LOOK (2026-07-03 09:40, per declared rule at post≥3,000; n=3,525 post-cutoff)**:
  - Sanity ✔: book baseline reproduces known level (ALL net@10 ≈ −7.2 long / −7.7 short bps); p_up non-degenerate (mean .449, sd .333); neg-control de-degenerated as predicted.
  - POST-cutoff keep70@10bps: LONG dNET **+1.73bps t=+2.02** CI[+0.01,+3.32]; SHORT +0.52 t=+0.41.
  - **Kill-rule check**: kill requires (both sides ≤+1bps) OR (both t<1) → NOT met (long exceeds both) → **continue to full post-cutoff per rule**.
  - ⚠️ BUT two of three final PASS conditions already failing at interim: (1) neg-control (within-dt1 shuffle) null = +0.90 [−0.16,+1.95] — observed long +1.73 sits INSIDE the null band; most of the "uplift" is hour-TIMING (keeping more trades in hours Kronos likes), not within-hour selection; (2) keep-50 sign FLIPS for longs (−0.28). Diag spearman(aligned, gross) post-cutoff ≈ +0.02 — at the familiar info-ceiling.
  - Leakage contrast already visible: PRE-cutoff dNET much larger (long +3.7 t2.0, short +7.4 t1.7, nulls +3.6/+7.2) — consistent with pretraining-corpus memorization inflating pre-2025-09 performance; reinforces POST as the only honest window.
  - Magnitude reality-check vs the goal (net returns): even the best cell keeps a −7.2bps book at −5.5bps — nowhere near cost-line rescue (needs >+7bps).

- **Step 9 — LIVE DEPLOYMENT as Vanguard veto layer (user decision 2026-07-03, "anyway lets add it")**: despite weak interim evidence, deployed for live observation. New module [kronos_veto.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/kronos_veto.py); hook in [orchestrator.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/orchestrator.py) AHEAD of the Gemini S1/S2 audit (user: "put it ahead of s2 news veto layer"; placed before the whole Gemini call — local GPU before API quota). Config block in `scripts/vanguard/config.py` (⚠️ UNCERTIFIED banner): **shadow mode default** (logs every decision to `data/kronos_veto_live.jsonl`, never blocks); flip with `KRONOS_VETO_MODE=enforce`. Rule = backtest keep-70% op point: LONG keep iff p_up≥0.30, SHORT keep iff 1−p_up≥0.4333; R=30 paths, 480×15m-bar context; fail-safe pass-through on any error. Also fixed `upstox_broker.get_recent_candles` REST fallback to scale calendar days with requested count (was starving deep 15m requests at 1 day). Functional test ✔ (load 3.9s cuda, LONG pass / SHORT would-veto, JSONL written). Extra diagnostics logged: 25bps magnitude veto INVERTS (−2.7bps); gross NEVER crosses +6bps at any coverage (peak +4.1bps @keep-70 long); p_up badly miscalibrated (Brier 0.348 vs base-rate 0.250; realized up-rate flat ~48% across all predicted bins) — only signal is faint avoid-the-worst tilt (ρ≈0.04).
  - ⚠️ Engine restart required to activate (live engine PID predates the code change). Recommend restarting AFTER the backtest scoring pass completes (GPU contention: veto latency 8–26s while scoring runs vs ~2–3s solo).
  - **Live hardening (2026-07-03 ~10:20)**: HDFCBANK abstain (KeyError timestamp) traced to the yfinance fallback candle format → added `_normalize_candles()` (handles Upstox v3 / WS cache / yfinance MultiIndex) + drops trailing in-progress 15m bar (completed-bars-only, matches backtest spec). All 3 formats tested + end-to-end GPU call ✔.
  - **Startup entry-hold added per user (2026-07-03)**: on engine restart the immediate catch-up scan is scan-only (`[STARTUP-HOLD]` log); new entries resume at the next quarter-hour scheduler boundary (boot exactly on a boundary defers to the following one). Exits/monitoring unaffected. `self.entries_resume_at` in orchestrator `__init__`; boundary math mirrors the scheduler sleep.
  - **ATR stop widened 1.0×→2.0× 15m-ATR per user (2026-07-03)** in `compute_15min_atr` (orchestrator): old 1× stop ≈ 0.8σ of the HOURLY move → ~40% noise-touch odds per hold (cf. 3 stop-outs in 2h, 2026-06-16). New: SL clamp [0.50, 2.00]%, fallback default 0.50→1.00%; TP unchanged 1.8×. Framed as disaster brake per stop-loss research (no stop width improves net; [[project_stop_loss_research]] finding). Sizing auto-halves (qty = risk/SL-distance), keeping ₹-risk/trade constant.
  - **Tightened keep-70% → keep-50% per user (2026-07-03)**: `KRONOS_THR_LONG 0.30→0.50`, `KRONOS_THR_SHORT 0.4333→0.70` (LONG kept iff ≥15/30 paths up; SHORT kept iff ≥21/30 paths down). ⚠️ backtest read at keep-50%: uplift flips NEGATIVE (long −0.28 / short −0.63bps) — shadow observation will arbitrate. Old op point documented in config for rollback.

- **Step 10 — ENFORCE mode + dashboard tracking per user (2026-07-03)**: `KRONOS_VETO_MODE` default shadow→**enforce** (would-veto candidates now BLOCKED, tracked as VETOED with counterfactual P&L via `start_vetoed_tracking`; revert via env `KRONOS_VETO_MODE=shadow`). Dashboard: `get_performance_stats()` gained `daily_kronos` (comment LIKE `%[KRONOS-%`); vanguard_v2.html gained a "Kronos Avoided" chip + green KRONOS stage badge in the AI Vetoed table. ⚠️ Combined with keep-50% thresholds, ~half of all candidates will now be blocked by an UNCERTIFIED layer whose backtest read at this tightness was NEGATIVE (−0.28/−0.63bps) — the "Kronos Avoided" counterfactual chip is the arbiter; review after ~2 weeks of live data.

- **Step 11 — FINAL PRE-REGISTERED VERDICT (2026-07-03, post-cutoff window COMPLETE, n=4,776)**: **DEAD.** Primary cell LONG keep70@10bps Δnet +1.03bps t=+1.34 (CI[−0.45,+2.58]) — fails t>2; SHORT −0.56 t=−0.52; keep-50 cells +0.50/−1.40. Interim +1.73 t=2.02 regressed as noise does. Pre-cutoff "uplift" (+2.7/+4.3, t≈2.1) fully reproduced by its neg-controls (nulls +1.8/+4.7) → timing + probable pretraining memorization. Full write-up: [[04 — Research/Kronos Zero-Shot Veto (2026-07)]]. Scorer died at chunk 660 (external kill, exit 127) AFTER completing post-cutoff — pre-cutoff left partial at 2,405 (sufficient for the secondary contrast; not restarting during market hours).

## 🔗 Core Memory Links & Backlinks
- [[04 — Research/Dual-TF Entry-Exit Overlay|Dual-TF panel provenance]]
- Prior dead veto attempts: co-sign v10, daily transformer veto, BCE veto (see Dead-Ends Register)
- Discipline: exploratory tier per AGENTS.md Model Metric Discipline §2
