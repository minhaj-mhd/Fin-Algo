"""
WP3 draft mapping: old path -> new path under the locked 00-09 taxonomy. Read-only; emits a CSV
the agent reviews before any git mv. Files needing a human/agent routing decision are flagged
needs_review=yes (08 research-vs-card, 06 plans, _Shared vs per-model, etc.).

Usage: python scripts/memory/build_mapping.py
Output: finalgo-memory-layer/finalgo/06. Context & Logs/restructure_mapping.csv
"""
import os, csv, re

VAULT = 'finalgo-memory-layer/finalgo'
INV = f'{VAULT}/06. Context & Logs/restructure_inventory.csv'
OUT = f'{VAULT}/06. Context & Logs/restructure_mapping.csv'

# ---- explicit per-file overrides (judgment calls) -------------------------------------------
# key = relpath under VAULT (forward slashes); value = (new_relpath, reason)
OVERRIDES = {
    'to remove.md': ('__DELETE__', 'junk'),
    'agent.md': ('00 — Start Here/AI Operating Protocol.md', 'thin pointer to repo-root AGENTS.md'),
    'Welcome.md': ('00 — Start Here/Welcome.md', 'regenerated MOC hub'),
    'tasks.md': ('06 — Logs/tasks.md', 'review: fold into Active Board or archive'),
    # 02 Model Suite -> _Shared
    '02. Model Suite/Feature Engineering & Normalization.md': ('02 — Models/_Shared/Feature Engineering & Normalization.md', ''),
    '02. Model Suite/Model Inference Data Structure.md': ('02 — Models/_Shared/Model Inference Data Structure.md', ''),
    '02. Model Suite/Model Registry & File Structures.md': ('02 — Models/_Shared/Model Registry & File Structures.md', ''),
    '02. Model Suite/Training Data & Regime Requirements.md': ('02 — Models/_Shared/Training Data & Regime Requirements.md', ''),
    '02. Model Suite/Model Performance & Statistics.md': ('02 — Models/_Shared/Model Performance & Statistics.md', ''),
    '02. Model Suite/V8 Microstructure Feature Comparison.md': ('02 — Models/_Shared/V8 Microstructure Feature Comparison.md', ''),
    '02. Model Suite/Advanced Tree Models Roadmap.md': ('02 — Models/_Shared/Advanced Tree Models Roadmap.md', ''),
    '02. Model Suite/Multi-Timeframe Models.md': ('02 — Models/_Shared/Multi-Timeframe Models.md', ''),
    # 02 Model Suite -> Daily Gatekeeper
    '02. Model Suite/Daily Gatekeeper V2 Rebuild Plan.md': ('02 — Models/Daily Gatekeeper/Daily Gatekeeper V2 Rebuild Plan.md', ''),
    '02. Model Suite/Daily Gatekeeper V2 Rebuild and Certification Report.md': ('02 — Models/Daily Gatekeeper/Daily Gatekeeper V2 Rebuild and Certification Report.md', ''),
    '02. Model Suite/Daily Gatekeeper V3 Rebuild and Certification Report.md': ('02 — Models/Daily Gatekeeper/Daily Gatekeeper V3 Rebuild and Certification Report.md', ''),
    '02. Model Suite/Gatekeeper V2 Feature Availability.md': ('02 — Models/Daily Gatekeeper/Gatekeeper V2 Feature Availability.md', ''),
    # 02 Model Suite -> Transformer
    '02. Model Suite/Cross-Sectional Transformer Architecture Proposal.md': ('02 — Models/Transformer/Cross-Sectional Transformer Architecture Proposal.md', ''),
    '02. Model Suite/DualRes-CrossSectional-Transformer-Architecture.md': ('02 — Models/Transformer/DualRes Cross-Sectional Transformer Architecture.md', 'normalize name'),
    '02. Model Suite/DualRes-Transformer-Flowchart.md': ('02 — Models/Transformer/DualRes Transformer Flowchart.md', 'normalize name'),
    '02. Model Suite/DualRes-Transformer-netPnL10-Report.md': ('02 — Models/Transformer/DualRes Transformer netPnL10 Report.md', 'normalize name'),
    '02. Model Suite/Sided-Transformer-Preregistration.md': ('02 — Models/Transformer/Sided Transformer Preregistration.md', 'normalize name'),
    # 02 Model Suite -> Meta-Veto
    '02. Model Suite/Meta-Veto Rectification Plan MV2.md': ('02 — Models/Meta-Veto/Meta-Veto Rectification Plan MV2.md', ''),
    '02. Model Suite/Meta-Veto Stacking Framework Plan.md': ('02 — Models/Meta-Veto/Meta-Veto Stacking Framework Plan.md', ''),
    # 02 Model Suite -> per-model / strategies
    '02. Model Suite/15m_Conviction_Audit_Report.md': ('02 — Models/15m/15m Conviction Audit Report.md', 'normalize name'),
    '02. Model Suite/Empirical Regime Simulation Results.md': ('03 — Strategies/Empirical Regime Simulation Results.md', 'strategy routing'),
    # 01 Core Architecture
    '01. Core Architecture/Global System Architecture.md': ('01 — Architecture/Global System Architecture.md', ''),
    '01. Core Architecture/Vanguard System Features.md': ('01 — Architecture/Vanguard System Features.md', ''),
    '01. Core Architecture/Validation Gauntlet Architecture.md': ('01 — Architecture/Validation Gauntlet/Validation Gauntlet Architecture.md', ''),
    '01. Core Architecture/Validation Gauntlet Remediation Plan.md': ('01 — Architecture/Validation Gauntlet/Validation Gauntlet Remediation Plan.md', ''),
    # 04 Data & Code Map -> 01 Architecture / 05 Operations
    '04. Data & Code Map/Shadow Tracker & Execution Loop.md': ('01 — Architecture/Execution & Runtime/Shadow Tracker & Execution Loop.md', ''),
    '04. Data & Code Map/AI Veto & Gemini Audit.md': ('01 — Architecture/Execution & Runtime/AI Veto & Gemini Audit.md', ''),
    '04. Data & Code Map/Database Architecture.md': ('01 — Architecture/Data & Code/Database Architecture.md', ''),
    '04. Data & Code Map/Codebase File Directory.md': ('01 — Architecture/Data & Code/Codebase File Directory.md', ''),
    '04. Data & Code Map/Upstox Brokerage API Plan.md': ('05 — Operations/Upstox Brokerage API Plan.md', ''),
    # 06 plans -> Architecture/Research, Restructure plan stays in 06
    '06. Context & Logs/Vanguard Engine Refactor Roadmap.md': ('01 — Architecture/Vanguard Engine Refactor Roadmap.md', ''),
    '06. Context & Logs/Codebase Cleanup Strategy.md': ('04 — Research/Codebase Cleanup Strategy.md', ''),
    '06. Context & Logs/Current Context.md': ('06 — Logs/Active Board.md', 'slim to <=10 live items in WP6'),
    # 08 research-vs-card splits -> 04 Research
    '08. Model Analysis/Dominance Variance Analysis.md': ('04 — Research/Dominance Variance Analysis.md', ''),
    '08. Model Analysis/1-Hour Vanguard Model/TBM-1h-Ensemble-Results.md': ('04 — Research/TBM 1h Ensemble Results.md', 'dead-end research; normalize name'),
    '08. Model Analysis/15-Minute Vanguard Model/Dual-TF Entry-Exit Overlay Research.md': ('04 — Research/Dual-TF Entry-Exit Overlay Research.md', 'research line'),
    '07. Research & Backtests/TBM-1h-Ensemble-Implementation-Plan.md': ('04 — Research/TBM 1h Ensemble Implementation Plan.md', 'normalize name'),
    '07. Research & Backtests/V18-Hybrid-Veto-Scalability.md': ('04 — Research/V18 Hybrid Veto Scalability.md', 'normalize name'),
    # resolved review items
    '08. Model Analysis/Gauntlet_V10_V19_Master_Report.md': ('02 — Models/Gauntlet Reports/Gauntlet V10 V19 Master Report.md', 'gauntlet master report'),
    '08. Model Analysis/1-Hour Vanguard Model/V10_V18_Independent_Findings.md': ('04 — Research/V10 V18 Independent Findings.md', 'cross-model research, not a card'),
    '08. Model Analysis/1-Hour Vanguard Model/V10_vs_V18_Edge_Evaluation.md': ('04 — Research/V10 vs V18 Edge Evaluation.md', 'cross-model research, not a card'),
}

# folder-level prefix rules (applied if no override). (old_prefix, new_prefix)
PREFIX_RULES = [
    ('08. Model Analysis/1-Hour Vanguard Model/', '02 — Models/1H/'),
    ('08. Model Analysis/15-Minute Vanguard Model/', '02 — Models/15m/'),
    ('08. Model Analysis/30-Minute Vanguard Model/', '02 — Models/30m/'),
    ('08. Model Analysis/Gauntlet Reports/', '02 — Models/Gauntlet Reports/'),
    ('08. Model Analysis/Meta-Veto/', '02 — Models/Meta-Veto/'),
    ('07. Cluster Research/', '04 — Research/Cluster Research/'),
    ('07. MCP Integrations/', '05 — Operations/MCP/'),
    ('05. Archives/', '09 — Archive/'),
    ('research_1030_strategy/', '04 — Research/1030 Strategy/'),
    ('03. Trading Strategies/', '03 — Strategies/'),
    ('06. Context & Logs/Conversations/', '06 — Logs/Conversations/'),
    ('06. Context & Logs/Daily Logs/', '06 — Logs/Daily Logs/'),
    ('06. Context & Logs/', '06 — Logs/'),
]

# names that still use snake/Hyphen and should normalize to Title Case With Spaces
def normalize_name(path):
    d, f = os.path.split(path)
    base, ext = os.path.splitext(f)
    # PRESERVE protocol conventions: dated daily logs (YYYY-MM-DD) and Conv-YYYY-MM-DD notes
    if re.match(r'^\d{4}-\d{2}-\d{2}', base) or base.startswith('Conv-'):
        return path
    if '_' in base or (base.count('-') >= 2 and ' ' not in base):
        base = base.replace('_', ' ').replace('-', ' ')
        base = ' '.join(w if (w.isupper() or any(c.isdigit() for c in w)) else w.capitalize() for w in base.split())
    return f"{d}/{base}{ext}" if d else f"{base}{ext}"

rows = []
with open(INV, encoding='utf-8') as fh:
    for r in csv.DictReader(fh):
        old = r['path']
        review = ''
        if old in OVERRIDES:
            new, reason = OVERRIDES[old]
        else:
            new = None
            for op, npref in PREFIX_RULES:
                if old.startswith(op):
                    new = npref + old[len(op):]
                    break
            reason = 'folder-rule'
            if new is None:
                new = old; reason = 'UNMAPPED'; review = 'yes'
            else:
                new = normalize_name(new)
            # flag 08 1H files that look like research not cards
            if old.startswith('08. Model Analysis/1-Hour') and any(k in old for k in ['V10_V18', 'V10_vs_V18', 'Independent']):
                review = 'yes'; reason = 'card-vs-research?'
        rows.append([old, new, r['guess_type'], r['guess_status'], reason, review])

with open(OUT, 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    w.writerow(['old_path', 'new_path', 'type', 'status', 'reason', 'needs_review'])
    w.writerows(rows)

from collections import Counter
print(f"mapped {len(rows)} files -> {OUT}")
print("new top-level distribution:")
for k, v in sorted(Counter(r[1].split('/')[0] for r in rows if r[1] != '__DELETE__').items()):
    print(f"  {v:3d}  {k}")
print("\nDELETE:", [r[0] for r in rows if r[1] == '__DELETE__'])
print("needs_review:", sum(1 for r in rows if r[5] == 'yes'))
for r in rows:
    if r[5] == 'yes':
        print(f"  [{r[4]}] {r[0]}")
