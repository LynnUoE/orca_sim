#!/bin/bash
# Safety layer off vs on for both v1 policies, then the safety-margin sweep on run23
# (results/2026-10-12-safety-layer.md). Run from Orca_rl/ with the safety environment:
#   PY=../../safety_venv/bin/python bash scripts/eval_safety_all.sh
set -e
PY=${PY:-python}
$PY -m orca_rl.evaluate_safety --model models/run23/final_model.zip --goal-angle 45 --episodes 100 --out results/safety/run23_45deg.json
$PY -m orca_rl.evaluate_safety --model models/run22/final_model.zip --goal-angle 45 --episodes 100 --out results/safety/run22_45deg.json
for mm in 2 1 0.2; do
  $PY -m orca_rl.evaluate_safety --model models/run23/final_model.zip --goal-angle 45 --episodes 100 --modes on \
      --safety-config scripts/safety_configs/margin_${mm}mm.yaml --out results/safety/run23_45deg_margin${mm}mm.json
done
