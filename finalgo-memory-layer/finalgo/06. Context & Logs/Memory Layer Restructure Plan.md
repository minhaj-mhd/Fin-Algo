# 🗂️ Memory Layer Restructure Plan (handoff spec for executing agents)

> **Status:** 🟢 Plan ready for execution · **Author:** Claude (2026-06-12) · **Type:** spec
> **Goal:** make the Obsidian vault (`finalgo-memory-layer/finalgo/`) fast to understand for **both humans and AI agents**, with zero orphaned content and a self-maintaining lifecycle.
> **For the executing agent:** do the Work Packages (WP0→WP8) in order, on a branch, with the safety rules in §7. Each WP has explicit acceptance criteria. Do **not** delete content without the §7 checks.

---

## 1 · Why (the diagnosis — audited 2026-06-12)
- **194 `.md` files; `Welcome.md` links only ~30** → ~85% of the vault is undiscoverable by browsing.
- **Folder numbering is broken:** three folders share the `07.` prefix (`07. Cluster Research`, `07. MCP Integrations`, `07. Research & Backtests`); `Welcome.md` points to a non-existent `10. MCP Integrations` and to stale "latest" logs.
- **Overlap & fuzzy boundaries:** `02. Model Suite` (21) vs `08. Model Analysis` (66) both hold model docs; one model's spec + card + analysis is split across both. Meta-Veto spans `02` + `08/Meta-Veto` with near-duplicate files (`M1 Orthogonality Audit MV2` vs `M1_Orthogonality_Audit_v10_v19`). 5 transformer files sit loose in `02` root.
- **Naming chaos:** `Title Case.md`, `snake_case.md`, and `Hyphen-Case.md` coexist, sometimes in one folder.
- **Lifecycle not enforced:** `Current Context.md` is a 115-line append-only "COMPLETED" log; 17 conversation notes accumulate instead of being archived+deleted per the protocol; root holds junk (`to remove.md`, duplicate `agent.md`).

**Root cause:** discovery depends on filenames + a hand-maintained index that nobody keeps current, and there is no machine-readable status/type on any file. Fix = (a) one clear taxonomy, (b) per-file front-matter as the backbone, (c) generated indexes, (d) an enforced lifecycle.

## 2 · Design principles
1. **One topic → one home.** Each subject lives in exactly one folder; everything about a model lives under that model.
2. **Front-matter is the source of truth, not the filename.** Agents filter by metadata; humans read MOCs. Renames never silently break discovery.
3. **Indexes are generated, not hand-curated.** A script rebuilds `Welcome.md`, per-folder MOCs, and a machine index from front-matter, so they can't rot.
4. **Status is always visible.** Every doc declares `active | concluded | superseded | dead | wip`. Dead-ends are first-class (a register), so agents never re-run a closed line.
5. **Logs decay, knowledge persists.** Conversations and "completed" bullets are transient and get archived; durable findings get promoted to reference docs.

## 3 · Target taxonomy
Clean, collision-free top level. Every move uses `git mv` (preserve history). Display names = **Title Case With Spaces** (Obsidian-native); the snake_case/Hyphen outliers get normalized.

```
00 — Start Here/
   Welcome.md                    # generated MOC hub (links to per-domain MOCs, NOT every file)
   AI Operating Protocol.md      # THIN pointer to repo-root AGENTS.md (no duplicated body)
   How To Navigate.md            # 1 page: humans + "Agent Quickstart" (how to grep front-matter)
   Naming & Front-matter Standard.md
   Dead-Ends Register.md         # one-line "don't retry X — see [[…]]" table
01 — Architecture/
   Global System Architecture.md
   Validation Gauntlet/          # Architecture, Remediation Plan
   Execution & Runtime/          # Shadow Tracker & Execution Loop, AI Veto & Gemini Audit   (from 04)
   Data & Code/                  # Database Architecture, Codebase File Directory             (from 04)
02 — Models/
   _Shared/                      # Feature Engineering, Model Inference Data Structure, Model Registry,
                                 #   Training Data & Regime Requirements, Model Performance & Statistics,
                                 #   V8 Microstructure Feature Comparison, Advanced Tree Models Roadmap,
                                 #   Multi-Timeframe Models
   1H/                           # Model Card v10 + all 08/1-Hour analysis
   15m/                          # Model Card v3 + 08/15-Minute analysis + 15m Conviction Audit
   30m/                          # 08/30-Minute analysis
   Daily Gatekeeper/             # V2 plan+report, V3 report, Gatekeeper V2 Feature Availability
   Transformer/                  # CST Proposal, DualRes Architecture/Flowchart/netPnL10 Report,
                                 #   Sided-Transformer-Preregistration
   Meta-Veto/                    # Rectification Plan MV2, Stacking Framework Plan, Certification,
                                 #   Orthogonality Audits (DEDUPE the two M1 files)
   Gauntlet Reports/             # the 30 per-model reports + a single generated Master Index table
03 — Strategies/
   Strategy Catalog, Market Friction & Slippage, Upstox Fees & Statutory Taxes,
   Strategy March 2026 Revision, MTF Limit Order Architecture, Time-Based ATR Targets,
   Strategy Dashboard UI, Empirical Regime Simulation Results   (last one from 02)
04 — Research/                   # exploratory; every file carries a VERDICT header + status
   Cluster Research/*, TBM 1h Ensemble (plan+results), V18 Hybrid Veto Scalability,
   Dominance Variance Analysis, Dual-TF Entry-Exit Overlay Research (from 08), 1030 Strategy/*
05 — Operations/
   MCP/                          # Jupyter, Obsidian, SQLite, TradingView, MCP Registry
   Run & Environment.md          # live-run guides (from Welcome), env/reproducibility
06 — Logs/
   Active Board.md               # the slimmed Current Context (≤10 LIVE items only)
   Daily Logs/
   Conversations/                # ACTIVE only; concluded ones fold into the day's log then delete
09 — Archive/                    # current 05 Archives + legacy_archive + newly-superseded docs
```

**Boundary rules (resolve the 02-vs-08 fuzziness):**
- `02 — Models/` holds everything **about a model**: spec, card, analysis, gauntlet stamp. (`08. Model Analysis` is dissolved into it.)
- `04 — Research/` holds **exploratory dead-ends / line-of-inquiry** work that isn't a single production model.
- A doc that is a **plan** for a model still lives under that model; when executed it gains `status: superseded` or is merged into the report (don't keep stale plans as if live).

## 4 · Front-matter standard (the backbone)
Every `.md` (except generated indexes) starts with YAML:
```yaml
---
title: Human Readable Title
type: spec | report | reference | research | log | archive | moc
status: active | concluded | superseded | dead | wip
model: v10_native_1h            # optional; model id if model-specific
verdict: FILTER_GRADE | DEAD | sub-cost | inconclusive   # optional
gauntlet_run_id: 20260610T...   # optional; REQUIRED if a verdict is cited (Metric Discipline)
updated: 2026-06-12
tags: [transformer, veto, short-side]
related: ["[[Other Note]]"]
---
```
Rules: `verdict` without a `gauntlet_run_id` (or an explicit `⚠️ UNVERIFIED`) is a protocol violation. `status: dead` files MUST have a one-line "why / don't retry" at the top and an entry in the Dead-Ends Register.

## 5 · Generated indexes (so they never rot)
Build `scripts/memory/build_index.py` (read-only over the vault) that scans front-matter and regenerates:
1. **`00 — Start Here/Welcome.md`** — hub: one section per top folder, linking to each folder's MOC + a 1-line status summary.
2. **`<folder>/_MOC.md`** in every top folder — table of its docs: `title | type | status | verdict | updated | one-liner`.
3. **`00 — Start Here/INDEX.json`** (or `.md` table) — machine-readable: `path, title, type, status, model, verdict, updated`. This is what agents grep first.
4. **`00 — Start Here/Dead-Ends Register.md`** — every `status: dead` doc as a one-liner.
Re-run it as the last step of any memory edit (add to the lifecycle checklist in §8).

## 6 · Migration — Work Packages (assign to agents, do in order)

**WP0 · Inventory & branch (read-only).** Create branch `memory-restructure`. Script-list every `.md`: `path, bytes, last-commit-date, first-H1, detected-topic` → `06 — Logs/restructure_inventory.csv`. *Accept:* CSV has all 194 files; nothing moved yet.

**WP1 · Freeze decisions.** Confirm §3 taxonomy, §4 front-matter, §3 naming with the user. Resolve open calls: keep numbering scheme `00–09`? merge `08→02`? *Accept:* user sign-off recorded in this doc's changelog.

**WP2 · Front-matter pass.** Add the §4 YAML to every file; agents read each doc to set `type/status/verdict` honestly (cite `gauntlet_run_id` where a verdict exists; mark unverifiable ones `⚠️ UNVERIFIED`). *Accept:* 0 files without front-matter; every `verdict` has a run-id or UNVERIFIED tag.

**WP3 · Move & consolidate.** `git mv` files into §3 taxonomy per a mapping table the agent appends here. Merge duplicate-topic files (e.g., the two Meta-Veto M1 audits → one, the other → `09 — Archive/` with `status: superseded`). *Accept:* every file in its §3 home; duplicates resolved; `git log --follow` still shows history.

**WP4 · Fix all links.** Update every `[[wikilink]]` and `](path)` to the new locations (script-assisted find/replace from the WP3 mapping). Run a **link-checker** (see §7) → 0 broken internal links. *Accept:* link-checker passes; Obsidian graph has no dangling-but-referenced nodes from moves.

**WP5 · Rebuild indexes.** Write & run `build_index.py` (§5). Replace the hand-written `Welcome.md`. *Accept:* Welcome + every `_MOC.md` + `INDEX.json` regenerated; **0 orphans** (every non-index file appears in exactly one MOC).

**WP6 · Lifecycle cleanup.** Slim `Current Context.md` → `Active Board.md` (≤10 live items; move the ~50 completed bullets into the relevant Daily Logs). Archive every 🔴 Concluded conversation into its day's Daily Log and delete the original (enforce the protocol). *Accept:* Active Board ≤10 items; `Conversations/` holds only 🟢 Active notes.

**WP7 · Kill junk & de-dup protocol.** Delete `to remove.md`; reconcile `tasks.md` (fold into Active Board or archive); replace vault `agent.md` with a thin pointer to repo-root `AGENTS.md` (no duplicated body — prevents divergence). *Accept:* no junk at root; single protocol source.

**WP8 · Lock it in.** Add a **Memory Hygiene checklist** to `AGENTS.md` Phase 4 (run `build_index.py`; conversations archived; Active Board pruned; front-matter present). Optionally add a CI/pre-commit check that fails on missing front-matter or broken links. Commit; ask user to review the rendered Obsidian graph. *Accept:* hygiene rules in AGENTS.md; clean commit; user review.

## 7 · Safety rules (binding — this is live project memory)
- **Branch + git mv only.** Never lose history; no `rm` of unique content — superseded → `09 — Archive/` with `status: superseded`, not deletion.
- **Content-preserving moves.** Moves/merges may reorganize and add front-matter but must not drop facts, `run_id`s, or numbers. If two docs conflict, keep both claims and flag the discrepancy (do not silently pick one).
- **Link integrity gate.** A move isn't done until the link-checker is green. Minimal checker: for every `[[Name]]`/`](rel/path)`, assert the target exists post-move.
- **UTF-8 + no shell-redirect edits** to `.md` (Windows UTF-16 corruption rule). Use file tools.
- **Don't touch `.obsidian/`** config or the separate `.claude` auto-memory (`MEMORY.md`) — that index is a *different* system (the agent's cross-session memory); this plan covers only the Obsidian vault. Note the relationship in `How To Navigate.md` so the two aren't confused.

## 8 · Definition of done
- 0 files without front-matter; 0 broken internal links; 0 orphans (all in a MOC).
- Top level matches §3; no numbering collisions; one protocol source.
- `Active Board.md` ≤10 live items; `Conversations/` only active; root junk gone.
- `Welcome.md`, all `_MOC.md`, `INDEX.json`, `Dead-Ends Register.md` are generated and current.
- A human can find any topic in ≤2 clicks from Welcome; an agent can answer "what's the status of model X / is line Y dead?" from `INDEX.json` + front-matter in one grep.

## 9 · Decisions (LOCKED 2026-06-12 by user)
1. **Folder numbering:** KEEP `00–09` numeric prefixes (deliberate sidebar order).
2. **Naming:** **Title Case With Spaces** for all display filenames; normalize snake_case/Hyphen outliers.
3. **Model docs:** DISSOLVE `08. Model Analysis` into `02 — Models/<model>/` (one home per model).
4. **Index:** BUILD the `build_index.py` generator (Welcome + per-folder MOCs + `INDEX.json` + Dead-Ends Register from front-matter); run it after every memory edit.

→ WP1 is complete; executing agents proceed from WP0 inventory under these locked decisions.
