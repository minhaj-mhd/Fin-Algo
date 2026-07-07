---
title: Kronos LoRA NSE Fine-Tune
type: research
status: dead
updated: 2026-07-07
model: kronos-lora
---

# Kronos LoRA NSE Fine-Tune — veto is a dead-end; 13:15-short is an in-sample lead

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

## 13:15-short deep-dive (grid-search standout, POST-HOC)
Top-1 short, entry 13:15 (dt1 12:15), post-cutoff n=160:
- **Raw book** (no Kronos): +11.10 bps@6 t1.99, **8/10 months positive** (consistent) — BUT
  **−4.04 pre-cutoff** (sign flip → period-specific regime, not structural) and **t1.26 @10bps**.
- **LoRA veto** on the cell: kept +25.98@6 t2.85, beats neg-control **p=0.007**, holds in both
  halves — BUT at monthly resolution it is **erratic** (Jan −15.00 vs book +5.02; big wins ride
  on n=1–6 months) and **fails multiple-comparison correction** (0.007 × 20 cells ≈ 0.14).
- **2×ATR stop** (intrabar high-triggered): adds only +1.7 bps (tighter 1×ATR +3.1); pure
  tail-cap (worst −260.95 → −133 bps), WR unchanged → deleveraging, not alpha (cf.
  [[project_stop_loss_research]]).

**Durable piece = "13:15 is a good hour to be short (post-cutoff regime)"**; the LoRA adapter is
a noisy sharpener at best. Not deployable; needs a **forward test** (unlevered, shadow). Do NOT
5x-leverage a triple-selected in-sample cell; S2 can't be relied on ([[project_s2_veto_live_falsified]]
— S2's live veto cut winners).

## Scripts (finalgo)
`scripts/research/kronos_veto_score_1h.py` (1h-native scorer, `--adapter`/`--post-only`),
`kronos_veto_compare_1h.py`, `kronos_veto_top1_by_tod.py`, `kronos_1315_short_deepdive.py`,
`kronos_1315_short_stop.py`. Env: `peft`/`transformers<5` with **`huggingface-hub==0.33.1` pinned**
(protects the live veto).

Related: [[Kronos Zero-Shot Veto (2026-07)]], memory `project_kronos_lora_nse`.
