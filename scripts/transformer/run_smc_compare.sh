#!/usr/bin/env bash
# Matched listwise comparison: baseline (81 TA) vs SMC (108 = 81 TA + 27 PA), both sides.
# Only difference between panels is the 27 price-action features (grids + labels identical).
for panel in data/transformer_panel data/transformer_panel_smc; do
  tag=$(basename "$panel")
  for side in long short; do
    echo "########## PANEL=$tag SIDE=$side ##########"
    TRANSFORMER_PANEL="$panel" python scripts/transformer/train.py \
        --objective listwise --target "$side" --epochs 15 --batch 16
  done
done
echo "########## ALL DONE ##########"
