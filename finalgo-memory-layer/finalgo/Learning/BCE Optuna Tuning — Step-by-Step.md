---
title: "BCE Veto Transformer — Optuna Hyperparameter Tuning (learning walkthrough)"
type: reference
status: 🟢 Active
updated: 2026-06-16
tags: [learning, automl, optuna, transformer]
---

# 📚 BCE Veto Transformer — Optuna Tuning (plain-language walkthrough)

> A learning-oriented, step-by-step explanation of what we are doing to **automatically tune** the
> BCE veto transformer with Optuna AutoML. Written to be followed without prior AutoML background.
> Newest steps are appended at the bottom (**Progress Log**).
> Lives in the `Learning/` folder; the terse agent log is in [[Conv-2026-06-16-BCE-Optuna-Tuning]].

---

## 1. What are we even trying to do? (the 30-second version)

We have a small neural network (the **BCE DualRes Transformer**) that looks at each stock and
outputs a number `P(up)` = "how likely this stock goes up next hour". We don't trade it directly.
Instead we use it as a **veto**: the v20 XGBoost model picks stocks to trade, and the transformer
**vetoes** (skips) the picks it thinks are bad.

On genuinely unseen data it already shows a faint-but-real skill: among v20's Top-3 LONG picks,
the ones the transformer keeps beat the ones it vetoes by **+1.14 bps (t = +2.27)**. That's not
yet profitable after costs, but it's a real signal. **The question this project answers:** can we
make that veto signal *stronger and more reliable* by choosing better settings for the network?

---

## 2. What is a "hyperparameter", and what is "AutoML / Optuna"?

- A **parameter** is something the network learns by itself during training (its internal weights).
- A **hyperparameter** is a setting *we* choose *before* training that controls *how* it learns —
  e.g. how big the network is, how fast it learns, how much it's allowed to "memorise".
  Pick them badly and a good idea looks dead; pick them well and a faint signal becomes usable.
- There are too many combinations to try by hand. **Optuna** is an "AutoML" tool that **searches**
  hyperparameter combinations *intelligently*: it tries some, sees which did well, and spends its
  next tries near the promising ones (this smart search is called **TPE**). Each attempt is a
  **trial**. To save time it also **prunes** (kills early) trials that are clearly going nowhere.

So "Optuna AutoML" here just means: *let the computer systematically hunt for the network settings
that make the veto signal strongest, instead of us guessing.*

---

## 3. The hyperparameters we will let Optuna search

Two groups.

**(A) The network's shape & learning settings**

| Setting | Plain meaning |
|---|---|
| `lr` (learning rate) | how big each learning step is |
| `batch` | how many timestamps it studies at once |
| `d_model` | how "wide"/expressive the network is |
| `nhead` | how many parallel "attention" views it uses (must divide `d_model`) |
| `t_layers` / `c_layers` | how "deep" the two parts of the network are |
| `dropout` | random switch-off during training to prevent memorising |
| `weight_decay` | gentle pressure to keep the network simple |

**(B) The loss function — *how we score the network's mistakes during training***

The "loss" is the rule that tells the network how wrong it was, which is what it tries to minimise.
Different rules emphasise different things:

| Loss | What it emphasises | Why it might help the veto |
|---|---|---|
| `plain_bce` | be well-calibrated on every name equally | baseline |
| `weighted_bce` | pay **more attention to big movers** (large % moves) | the veto's payoff comes from the big winners/losers |
| `focal` | pay **more attention to the hard, uncertain names** | stop wasting effort on easy/obvious names |
| `bce_profit_hybrid` | mix "be right" with "actually make money after cost" | trains directly on the economics we care about |

Each of those has its own little knobs (e.g. focal's `gamma`/`alpha`, hybrid's `mix`), which
Optuna only tunes when that loss is chosen.

---

## 4. What does Optuna try to maximise? (the objective)

This is the most important design choice. The training loss (above) is *not* the final goal — the
final goal is the **veto's edge**. So each trial is scored by:

> the **veto Δnet on the validation data** (how much better "kept" picks are than "all" picks),
> measured for the Top-3 LONG basket.

We use the "downstream" objective (rather than the simpler but weaker "just maximise accuracy").
Two refinements, both folded in:

1. **Reward *consistency*, not just a high average.** We optimise the **t-statistic** of the edge
   (average ÷ wobble) instead of the raw average, and we add a **stability floor**: we split the
   validation period into 3 calendar blocks and also look at the *worst* block. A setting that's
   great in one month and dead the next gets penalised.
2. **A cheating-check on every trial.** We re-run with the returns **shuffled** (a "negative
   control"). If the "edge" survives shuffling, it was fake/leakage — that trial is rejected.

---

## 5. The rules we must NOT break (why this is careful, not just fast)

This repo has been burned before by numbers that looked great but were illusions. So:

- 🚫 **Never tune on the final test data.** We tune only on *validation* data. The *test* data
  (the genuine "future", Sep 2025–Jun 2026) is touched **exactly once** at the very end to report
  an honest result. If we tuned on it, the famous +2.27 result would become a self-fulfilling fake.
- 🚫 **Never overwrite the existing trained model** during experiments (we added a `--no_save`
  safety switch for that).
- ✅ **Calibration vs threshold:** because different losses shift the `P(up)` numbers around, we
  compare them by *ranking* (keep the top X% of picks) rather than a fixed `P>0.50` cutoff. The
  real cutoff is chosen on validation and frozen for the final test.
- ✅ Everything here is **exploratory** — it gets **no official "Gauntlet" verdict** and writes no
  model stamps.

---

## 6. The plan, in phases

- **Phase 0 — Preparation (make the code tunable).** ✅ *done this session* — see Progress Log.
- **Phase 1 — The search.** Write the Optuna script, let it run ~30–50 trials on validation data
  for ~2–3 hours, pruning weak ones early.
- **Phase 2 — Honest confirmation.** Take the single best setting, retrain it 2–3 times (different
  random seeds) to be sure it wasn't luck, then run it **once** on the frozen test data and compare
  to the current +1.14 bps / t +2.27 baseline. Report whatever we find — better OR not.

---

## 7. Glossary (quick reference)

- **bps** — "basis points", 1 bps = 0.01%. Edges and costs are measured here.
- **Δnet (delta-net)** — how much better the KEPT picks are than ALL picks, after costs.
- **t-statistic** — edge size relative to its noise; |t| > 2 ≈ "probably not luck".
- **trial** — one full attempt with one set of hyperparameters.
- **pruning** — stopping a clearly-bad trial early to save time.
- **negative control** — a deliberately-broken re-run (shuffled returns); a real edge must vanish.
- **validation vs test** — validation = data we're allowed to tune against; test = the sealed
  "future" we only look at once.

---

## 8. Progress Log (newest at bottom)

### Phase 0 — done ✅
**Goal:** make the existing code controllable by Optuna without breaking anything.

1. **Added the 4 loss functions** to [train.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/train.py)
   (`bce_family_loss`: plain / weighted / focal / profit-hybrid) and exposed every hyperparameter
   on the command line (`--loss`, `--t_layers`, `--c_layers`, `--nhead`, `--weight_decay`, `--patience`, …).
   - *Checked numerically:* all 4 losses produce sensible, finite values and gradients. For the
     profit-hybrid we verified the two pieces are the **same size** (BCE ≈ 0.73, profit ≈ 0.15) so
     the "mix" dial really does something — a classic bug here is one piece silently dominating.
2. **Added a `--no_save` safety switch** so experiments can't overwrite the production model file.
3. **Added one shared definition of the train/val/test split** (`chrono_split`) so the training
   code and the evaluation code can never disagree about which dates are "test" (that disagreement
   would be a hidden data-leak).
4. **Built [veto_lib.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/veto_lib.py)** — the reusable scoring engine. It can score the veto edge on
   *any* time window (so the tuner can score the **validation** window each trial), with the
   coverage-matching, the day-clustered confidence interval, the shuffle cheating-check, and the
   per-block stability floor described above.
5. **Kicked off a timing run** (2 epochs) to measure how long one trial takes — this tells us how
   many trials we can afford. *(result pending)*

### Phase 0 results
- Timing run worked end-to-end: **~110 s/epoch** on the RTX 5050, cost-accounting clean, the
  `--no_save` switch protected the production model. ✅

### Phase 1 — the search (RUNNING)
- Installed Optuna 4.9.0 and wrote [tune_bce_optuna.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/tune_bce_optuna.py).
- Hit (and fixed) a Windows-specific crash: `torch` must be imported *after* `pandas` or the
  process segfaults at startup (a known OpenMP/MKL clash on this machine).
- **Sanity trial passed**: one 2-epoch trial ran the whole train→veto-score loop cleanly
  (coverage 0.63 ≈ target, the cheating-check came back ~0). The machinery is correct.
- **Reality of the budget**: a full 10-epoch trial is ~20 min, so 2.5h only fits ~12–16 trials.
  To reach ~20–30 we train each epoch on a **resampled 70% of timestamps** — this only needs to
  *rank* settings correctly; the winner is retrained on 100% of the data in Phase 2.
- **Launched the real search**: 10 epochs/trial, up to 50 trials, hard stop at 2.5 hours, saved
  continuously to `artifacts/optuna_bce_study.db`. Results will land in `artifacts/optuna_bce_best.json`.

### Phase 1 — batch 1 result (13 trials)
The first 2.5h ran 13 trials (7 scored, 6 pruned early). What we learned:
- **The new losses did NOT help.** The best 4 settings were all `plain_bce`; `weighted_bce` was the
  *worst*, and `bce_profit_hybrid` / `focal` trailed. So the loss zoo you asked for was worth
  *ruling out* — and it's ruled out. This fits the long-standing pattern that the limit here is
  *information in the data*, not the training recipe.
- **Best setting (so far):** `plain_bce`, a wide-but-shallow network (d_model 128, 1 temporal layer,
  2 cross-sectional layers), validation edge Δt ≈ +2.86 / +1.42 bps.
- **Two honest caveats:** the search was shallow (only 13 trials, mostly random), and the winner's
  cheating-check wasn't perfectly clean (+0.41 bps), so its *real* validation edge is nearer +1.0 bps.

### Phase 1 — batch 2 (RUNNING)
You chose to search a bit more before spending the one-time TEST check. Resumed the same study for
another ~2.5h; these trials are now *guided* by what batch 1 learned (more focused, less random).

### Phase 1 — batch 2 result (now 31 trials total)
- **The winner did not change, but it's now confirmed solid.** The smart search piled more attempts
  into the same neighbourhood (`plain_bce`, wide-shallow net) and several *independent* trials landed
  at nearly the same edge (Δt ≈ +2.6 to +2.8). So it's a stable sweet-spot, not a fluke.
- `focal` got a proper try this round and still lost (+0.17). **plain_bce wins, decisively.**
- **A measurement bug we caught:** our "cheating-check" used a *single* shuffle with a fixed random
  seed, so every trial reused the *same* shuffle — which happened to read +0.4 bps. That's one noisy
  draw, not a real bias. **Fix:** the check now averages over *many* shuffles (mean ± spread) and also
  reports the edge *after subtracting* the control. This makes the final TEST verdict trustworthy.

### Phase 2 — sealed TEST confirmation (RUNNING)
Retraining the winning setting on **100% of the training data** across 3 random seeds, then scoring it
**once** on the sealed TEST period — both the way we tuned (keep top 65%) and the baseline's exact way
(keep P>0.50) — each with the upgraded multi-shuffle control. We report the **median across seeds** so
a lucky initialisation can't flatter the result. Output → `artifacts/phase2_test_confirmation.json`.

### Next
- Read the TEST verdict; write the honest result (better OR not) into [[BCE-Transformer-V20-Veto]].

---

## 🔗 Links
- [[BCE-Transformer-V20-Veto]] — the underlying veto research + baseline numbers
- [[project_dualres_transformer_result]] — earlier finding that this signal is "info-limited"
- [[Conv-2026-06-16-v20-Cadence-Transformer]] — the session that produced the baseline
