"""
Phase 5d ablation — WHICH v21 cleaning element moved long/short?

Leave-one-out on the same 110-name universe / same purged WF folds, graph OFF (already shown
dead). Every variant uses the SAME production-faithful feature set (all non-meta columns incl.
Relative_*/Hour/Time_To_Close) so the comparison is clean — this also CORRECTS the headline
eval, where v20@110 was accidentally given only its z-scored columns.

Variants (hygiene, clean_feats[mask+wall-clock], gap_feats, robust scoring):
  BASE  (v20@110)  F F F F
  FULL  (clean)    T T T T
  -clean_feats     T F T T   -> isolates mask+wall-clock lookback
  -hygiene         F T T T   -> isolates frozen/zero-vol bar drop
  -gap             T T F T   -> isolates causal session/gap features
  -robust          T T T F   -> isolates median/MAD winsorized scoring (vs mean/std)

Each element's contribution = FULL - (that element turned off). RESEARCH ONLY; overlapping
windows => point estimates, no t-tests.

Run: python scripts/research/ablate_v21_cleaning.py
"""
import os, sys, json
import pandas as pd
from tqdm import tqdm
sys.path.append(os.getcwd())

from scripts.research.build_v21_rolling_1h_panel import _load_raw, build_ticker, build_ranking, SRC_DIR
from scripts.research.eval_v21_vs_v20 import wf_eval

UNIV_JSON = 'data/research/v21_rolling_1h/universe.json'
EXCL = {'DateTime', 'DateTime_15Min', 'DateTime_Hour', 'Query_ID', 'Ticker', 'Open', 'High',
        'Low', 'Close', 'Volume', 'Next_Hour_Return', 'YearMonth', 'candle_type', 'tradable'}

# name -> (hygiene, clean_feats, gap_feats, robust)
VARIANTS = {
    'BASE(v20@110)': (False, False, False, False),
    'FULL(clean)':   (True,  True,  True,  True),
    '-clean_feats':  (True,  False, True,  True),
    '-hygiene':      (False, True,  True,  True),
    '-gap':          (True,  True,  False, True),
    '-robust':       (True,  True,  True,  False),
}


def assemble(universe, hygiene, clean_feats, gap_feats):
    frames = []
    for tk in tqdm(universe, desc=f'  build(h={int(hygiene)},c={int(clean_feats)},g={int(gap_feats)})', leave=False):
        fp = os.path.join(SRC_DIR, tk + '.csv')
        if not os.path.exists(fp):
            continue
        f, _ = build_ticker(tk, _load_raw(fp, hygiene=hygiene), clean_feats=clean_feats, gap_feats=gap_feats)
        if f is not None and len(f):
            frames.append(f)
    return pd.concat(frames, ignore_index=True)


def main():
    universe = json.load(open(UNIV_JSON))['tickers']
    print(f"universe: {len(universe)} tickers | leave-one-out ablation (graph off)\n")
    cache, res = {}, {}
    for name, (hy, cf, gf, rob) in VARIANTS.items():
        key = (hy, cf, gf)
        if key not in cache:
            cache[key] = assemble(universe, hy, cf, gf)
        final, _ = build_ranking(cache[key].copy(), robust=rob)
        feats = [c for c in final.columns if c not in EXCL]
        res[name] = wf_eval(final, feats, label=name)

    print("\n=== leave-one-out contributions (FULL - element_off), rho ===")
    full = res['FULL(clean)']
    for name in ['-clean_feats', '-hygiene', '-gap', '-robust']:
        dL = full['L_rho'] - res[name]['L_rho']
        dS = full['S_rho'] - res[name]['S_rho']
        elem = name[1:]
        print(f"  {elem:12s}: L_rho {dL:+.4f}  S_rho {dS:+.4f}")
    print(f"\n  FULL vs BASE: L_rho {full['L_rho']-res['BASE(v20@110)']['L_rho']:+.4f}  "
          f"S_rho {full['S_rho']-res['BASE(v20@110)']['S_rho']:+.4f}  "
          f"(corrected baseline; same feature set for all)")
    json.dump(res, open('data/research/v21_rolling_1h/ablation_summary.json', 'w'), indent=2, default=float)
    print("\nsaved data/research/v21_rolling_1h/ablation_summary.json")


if __name__ == '__main__':
    main()
