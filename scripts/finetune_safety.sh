#!/bin/bash
# Fine-tune a v1 policy for 5M more steps with and without the safety filter in the loop, then evaluate both
# (results/2026-10-13-safety-finetune.md). Run from Orca_rl/ with the safety environment:
#   PY=../../safety_venv/bin/python BASE=run23 SEED=1 bash scripts/finetune_safety.sh     # Mac, 10 cores
#   PY=~/safety_venv/bin/python BASE=run22 SEED=0 NENVS=14 NSTEPS=292 bash scripts/finetune_safety.sh   # desktop
# Takes ~15 min (control) + ~8-10 h (with the filter). Models land in runs/ (not tracked).
set -e
PY=${PY:-python}
BASE=${BASE:-run23}; SEED=${SEED:-1}; NENVS=${NENVS:-10}; NSTEPS=${NSTEPS:-409}
COMMON="--version v1 --obs-rotvec --obs-fingertips --seed $SEED --n-envs $NENVS --n-steps $NSTEPS \
        --timesteps 5000000 --curriculum-start-deg 45 --resume models/$BASE/final_model.zip"
$PY -m orca_rl.train --name ${BASE}_ft_ctrl $COMMON
$PY -m orca_rl.train --name ${BASE}_ft_safety $COMMON --safety
for r in ${BASE}_ft_ctrl ${BASE}_ft_safety; do
  $PY -m orca_rl.evaluate_safety --model runs/$r/final_model.zip --goal-angle 45 --episodes 100 \
      --out results/safety/${r}_45deg.json
done
