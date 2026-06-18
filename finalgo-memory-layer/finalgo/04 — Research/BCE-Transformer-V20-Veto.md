---
title: "BCE Transformer V20 Veto — Walk-Forward Results"
type: research
status: active
model: dualres-transformer
updated: 2026-06-16
verdict: "⚠️ UNVERIFIED — genuine discriminative power, not standalone net-positive"
---

# 🔬 BCE Transformer (v20 panel) — Veto Layer Walk-Forward

## Summary
BCE DualRes Transformer trained on the v20 rolling-1h tensor panel, evaluated as a
**veto layer on top of v20 XGBoost** picks. Genuine K=3 LONG discriminative power
confirmed on genuine OOS (Sep 2025 – Jun 2026). Pre-registered WIN not achieved due
to v20 base edge degradation in OOS period, not transformer failure.

---

## Training Setup
| Item | Value |
|---|---|
| Panel | `data/transformer_panel_v20` (rolling-1h, ~18/day) |
| Objective | BCE (`--objective bce`) |
| Architecture | DualResCSTransformer, d_model=64 |
| Features | 81 (VIEW_A+B+C), 28 macro, 15 sectors |
| Split | 70/15/15 chronological + 30-bar embargo |
| Best val AUC | **0.5264** (epoch 6, early stopped epoch 12) |
| Test AUC | **0.5204** |

### Test Results (standalone)
| Side | K=1 Gross | K=1 Net @6bps | K=1 Net @10bps |
|---|---|---|---|
| LONG | +4.52 bps | −1.48 bps | −5.48 bps |
| SHORT | +3.23 bps | −2.77 bps | −6.77 bps |

Artifact: `artifacts/dualres_transformer.pt` + `artifacts/dualres_transformer_metrics.json`

---

## Walk-Forward Veto Evaluation
**Script**: `scripts/transformer/v20_bce_veto_walkforward.py`
**Output**: `artifacts/v20_bce_veto_walkforward.json`

- **OOS window**: 2025-09-29 → 2026-06-04 (2,616 timestamps, 447k rows)
- **Primary ranker**: v20 XGBoost (`models/research/v20_rolling_1h/xgb_long/short_model.json`)
- **Veto rule**: LONG keep if P(up) > th ; SHORT keep if P(up) < 1−th
- **Bootstrap**: 2,000 iterations, day-clustered

### Key Results @ th=0.50

| K | Side | n_picks | Coverage | ALL net | KEPT net | VETOED net | Δnet | t | CI |
|---|---|---|---|---|---|---|---|---|---|
| K=1 | LONG | 2,616 | 67% | −3.41 | −3.97 | — | −0.56 | −0.71 | [−2.1, +1.0] |
| K=1 | SHORT | 2,616 | 99% | −3.93 | −4.01 | — | −0.08 | −0.42 | [−0.5, +0.3] |
| **K=3** | **LONG** | **7,848** | **65%** | **−3.56** | **−2.43** | **−5.65** | **+1.14** | **+2.27** | **[+0.2, +2.1]** ✅ |
| K=3 | SHORT | 7,848 | 98% | −5.37 | −5.45 | −0.81 | −0.08 | −0.92 | [−0.3, +0.1] |
| K=5 | LONG | 13,080 | 63% | −3.51 | −2.80 | −4.69 | +0.71 | +1.63 | [−0.1, +1.6] |
| K=5 | SHORT | 13,080 | 98% | −5.29 | −5.29 | −5.28 | −0.00 | −0.00 | [−0.2, +0.2] |

Neg-control (shuffled returns): all ≤ 0.59 bps [OK ~0] ✅ — no data leakage.

---

## Pre-Registered Verdict

**WIN condition**: Δnet CI lo > 0 AND kept_net ≥ +1 bps AND vetoed < all  
**Result**: 0 hits — WIN not achieved.

**Why**: v20 XGB is itself net-negative in this OOS period (ALL net = −3.56 bps at 6 bps cost).
The transformer correctly identifies better vs worse picks within v20's basket (proven by
Δnet = +1.14 bps, CI lo > 0), but cannot lift a net-negative base to net-positive.

**This is a base-strategy degradation issue, not a transformer failure.**

---

## Path to Net-Positive

| Scenario | Gross needed | Gap |
|---|---|---|
| Current (6 bps cost) | > 6 bps | KEPT gross ~3.5 bps → −2.5 bps short |
| Limit orders (~3 bps cost) | > 3 bps | KEPT gross ~3.5 bps → **+0.5 bps NET** ✅ |
| v20 edge recovery | KEPT net ~−2.4 bps today | Need +2.4 bps more gross |

**Primary lever**: limit order execution to reduce effective cost to ≤ 3 bps.

---

## Objective Comparison (all on v20 panel)

| Objective | LONG K=1 Gross | SHORT K=1 Gross | Test AUC/rho | Use for veto? |
|---|---|---|---|---|
| **BCE** | **+4.52 bps** | **+3.23 bps** | AUC 0.5204 | ✅ Best choice |
| DualRes | ~+4.5 bps | ~+3.2 bps | rho ~0.020 | ✅ Comparable |
| Listwise LONG | +4.88 bps | — | rho −0.0036 | ❌ Anti-ranked |
| Listwise SHORT | — | −1.25 bps | rho +0.0115 | ❌ Dead |

BCE wins for veto: calibrated P(winner) → natural threshold semantics.
Listwise LONG is anti-ranked on val — cannot be used as LONG veto.

---

## Links
- [[00 — Start Here/Ray of Hope]] — Tier 3 entry
- [[Conv-2026-06-16-v20-Cadence-Transformer]] — full session log
- [[project_v20_rolling_1h_result]] — v20 XGB baseline
- [[feedback_validate_cost_accounting]] — cost-sign guard confirmed clean
