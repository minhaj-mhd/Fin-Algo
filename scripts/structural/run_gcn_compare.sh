#!/usr/bin/env bash
# Level-graph gated GCN vs listwise transformer baseline (TEST rank-IC: long +0.0014, short +0.0066).
# Same decision universe + objective + metrics; short is the side with prior signal.
echo "########## GCN short (REAL) ##########"
python scripts/structural/train_gcn.py --target short --epochs 15
echo "########## GCN short (NEG-CONTROL) ##########"
python scripts/structural/train_gcn.py --target short --epochs 15 --neg_control
echo "########## GCN long (REAL) ##########"
python scripts/structural/train_gcn.py --target long --epochs 15
echo "########## GCN ALL DONE ##########"
