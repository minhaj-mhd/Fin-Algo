#!/usr/bin/env bash
for side in short long; do
  echo "########## WF $side (baseline panel) ##########"
  python scripts/transformer/wf_rho.py --target $side --folds 5 --epochs 8
done
echo "########## WF ALL DONE ##########"
