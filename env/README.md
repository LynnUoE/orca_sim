# Environments

| File | What it pins | Used for |
|---|---|---|
| `../requirements.txt` | the packages, loosely | starting a new environment |
| `requirements-train.lock.txt` | exact versions on the Mac (macOS, Apple M4, Python 3.11.16), exported 2026-10-13 | the Mac training runs; versions were not recorded per run |
| `requirements-safety.lock.txt` | exact versions on the Mac, plus pinocchio / OSQP | safety-layer evaluations, `train.py --safety`, `record_safety` |

```bash
# training environment
python3.11 -m venv venv && venv/bin/pip install -r Orca_rl/env/requirements-train.lock.txt
# safety environment (from the Orcahand-Safe-Assembly/ root; Orca_safety stays editable)
python3.11 -m venv safety_venv && safety_venv/bin/pip install \
    -r Orca_rl/env/requirements-safety.lock.txt -e './Orca_safety[geometry,safety]'
```

Both install orca_sim editable from git at `05e00da` (unmodified upstream); `uv pip` refuses that line, so use
the venv's own `pip` (`uv venv --seed` gives it one).

The lock files are macOS builds. On Linux, `torch==2.13.0` from PyPI is the CUDA build (several GB); the
network is small and trains faster on CPU, so install `torch` from `https://download.pytorch.org/whl/cpu` first.
The desktop runs (WSL2 Ubuntu, i9-11900F) used torch 2.14.0, mujoco 3.13.0, numpy 2.4.6, stable-baselines3 2.9.0,
gymnasium 1.3.0.

**MuJoCo version changes individual episodes.** The same policy and reset seeds give different episodes under
3.12.0 and 3.15.0 (see `../SAFETY.md`), so compare runs within one MuJoCo version.

## Experiment scripts (`../scripts/`)

- `eval_safety_all.sh`: off / on for run22 and run23, and the 2 / 1 / 0.2 mm margin sweep (`safety_configs/`).
- `finetune_safety.sh`: 5M-step fine-tune with and without the filter, then evaluation.
