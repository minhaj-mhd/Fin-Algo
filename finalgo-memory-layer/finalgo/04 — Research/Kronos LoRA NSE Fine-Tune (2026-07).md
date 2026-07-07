---
title: Kronos LoRA NSE Fine-Tune
type: research
status: dead
updated: 2026-07-07
model: kronos-lora
---

# Kronos LoRA NSE Fine-Tune — veto is a dead-end (13:15-short "lead" was a bug)

**Verdict (⚠️ UNVERIFIED — no Gauntlet):** LoRA-fine-tuning Kronos on NSE candles gave a real
**model-quality** gain but **no trading edge**. Keep the live veto on **zero-shot base**.

Standalone repo: `C:\Users\loq\Desktop\Trading\kronos-nse-lora` (GitHub + HF, private, LFS,
vendored Kronos MIT). Two predictor-only LoRA adapters (tokenizer frozen, r16/α32, attn+ffn):
**1h** (2022–2026, val 2.3445) and **daily** (2016–2026, val 3.0767, overfits after ep1).

## Forecast gate (teacher-forced CE, OOS ≥ 2025-09-09)
- **1h**: base 2.6205 → LoRA **2.3684 (−9.6%)** — generalises (OOS CE ≈ val).
- **daily**: base 3.0511 → LoRA 3.0319 (**−0.6%**, marginal).

## 1h veto verdict (5,424 post-cutoff contexts, `scores_1h_base/lora.csv`)
**LoRA does NOT beat zero-shot base** — long keep70@10bps base **+1.20** t1.87 vs LoRA **+0.63**
t1.00; short dead both; **both inside the shuffle neg-control band**. rank-IC nudged (long
0.032→0.039) but did not convert to net-bps. **CE↓ ≠ tradeable edge; 1h info-ceiling reconfirmed.**

## ⚠️ CORRECTION — the "13:15-short lead" was a top-1 mis-selection (RETRACTED)
An earlier pass (`kronos_veto_top1_by_tod.py`, `kronos_1315_short_deepdive.py`,
`kronos_1315_short_stop.py`) reported a 13:15-short edge (+11 raw, +26 LoRA-kept, split-half
stable, p=0.007). **It was a bug:** "top-1" was selected by the **15-minute model's rank
`rkS_0` (min)** instead of the **host ranker's short score `sS` (max)** — wrong model AND wrong
sign. Corrected results:
- **Correct top-1 (host #1 = max `sS`), post-cutoff n=905/side:** LONG no-veto −5.89 (t−2.9);
  SHORT no-veto −2.99. Kronos veto is **neutral on longs** (dNET +0.8/+1.1) and
  **INVERTED/harmful on shorts** (dNET −5: keeps losers −8, vetoes winners +2 = Kronos
  "conviction inverts"). 13:15-short under correct top-1: +4.56@6 but **+0.56@10** (nothing).
- **min-`rkS_0` is itself an ENTRY-PRINT ARTIFACT** (dual-TF rank-lag collapse): lag test picks
  short by min `rkS_m` — edge lives ONLY at `rkS_60` (the entry-price rank, +19.5 t6.3) and is
  −7.9 at `rkS_45` (one bar earlier), −3.2 at `rkS_0`; pre-cutoff min-`rkS_0` = −9.3 t−3.9.
  Selecting on `rkS_60` = selecting on the entry price itself → not tradeable
  (cf. [[Dual-TF entry/exit research]], "rank-lag collapse" screen).

**Net: no tradeable edge from either route** — host #1 (veto inverted on shorts) or min-`rkS`
(entry-print artifact). The 5x-leverage / stop-loss / split-half analyses were all built on the
mis-selection and are **void**. Main verdict (LoRA = better model, no veto edge) is reinforced.

## Scripts (finalgo)
`scripts/research/kronos_veto_score_1h.py` (1h-native scorer, `--adapter`/`--post-only`),
`kronos_veto_compare_1h.py`, `kronos_veto_top1_by_tod.py`, `kronos_1315_short_deepdive.py`,
`kronos_1315_short_stop.py`. Env: `peft`/`transformers<5` with **`huggingface-hub==0.33.1` pinned**
(protects the live veto).

Related: [[Kronos Zero-Shot Veto (2026-07)]], memory `project_kronos_lora_nse`.
