"""PPO baseline for continuous in-hand cube reorientation.

    python -m orca_rl.train --timesteps 20_000_000 --n-envs 8

Everything lands in ``runs/<name>/``: checkpoints, the VecNormalize statistics
(needed at evaluation time), and TensorBoard logs.

    tensorboard --logdir runs/
"""

from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)

from orca_rl.task import (
    ENV_KWARGS_FILE,
    LEGACY_OBS_DIM,
    CubeReorientContinuous,
    resolve_stats_path,
)


class TaskMetricsCallback(BaseCallback):
    """Log what actually matters: solves per episode, drop rate, hold quality.

    Reward alone is a poor progress signal on this task -- it is exactly the
    number a broken reward function inflates. These are the numbers to watch.
    """

    def __init__(self, window: int = 100) -> None:
        super().__init__()
        self.window = window
        self._successes: list[int] = []
        self._drops: list[float] = []
        self._lengths: list[int] = []
        self._goal_angles: list[float] = []
        self._solve_rates: list[float] = []
        self._goals: list[int] = []
        self._drop_counts: list[int] = []
        self._goal_solves: list[int] = []

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if not done:
                continue
            self._successes.append(int(info.get("episode_successes", 0)))
            # Episodes with at least one drop. Reading the terminal "dropped"
            # flag would report 0% under drop_mode="reset_cube", where the cube
            # is already back in the hand by the time the episode ends.
            n_drops = int(info.get("episode_drops", int(bool(info.get("dropped", False)))))
            self._drops.append(float(n_drops > 0))
            self._drop_counts.append(n_drops)
            self._lengths.append(int(info.get("elapsed_steps", 0)))
            self._goal_angles.append(float(info.get("goal_angle_deg", 0.0)))
            self._solve_rates.append(float(info.get("curriculum_solve_rate", 0.0)))
            self._goals.append(int(info.get("episode_goals", 0)))
            self._goal_solves.append(int(info.get("episode_goal_solves", 0)))

        if len(self._successes) >= self.window:
            self.logger.record("task/solves_per_episode", float(np.mean(self._successes)))
            self.logger.record("task/solved_any_frac", float(np.mean([s > 0 for s in self._successes])))
            self.logger.record("task/drop_rate", float(np.mean(self._drops)))
            self.logger.record("task/drops_per_episode", float(np.mean(self._drop_counts)))
            self.logger.record("task/episode_length", float(np.mean(self._lengths)))
            self.logger.record("task/goal_angle_deg", float(np.mean(self._goal_angles)))
            self.logger.record("task/curriculum_solve_rate", float(np.mean(self._solve_rates)))
            # Goal attempts are the unit of opportunity now: how many the policy
            # gets through per episode, and what fraction of them it converts.
            goals = float(np.sum(self._goals))
            self.logger.record("task/goals_per_episode", goals / len(self._goals))
            self.logger.record(
                "task/goal_success_rate",
                float(np.sum(self._goal_solves)) / goals if goals else 0.0,
            )
            self._goals.clear()
            self._drop_counts.clear()
            self._goal_solves.clear()
            self._successes.clear()
            self._drops.clear()
            self._lengths.clear()
            self._goal_angles.clear()
            self._solve_rates.clear()
        return True


class EpochsUsedCallback(BaseCallback):
    """Log how many of `--n-epochs` PPO actually ran before target_kl stopped it.

    Not cosmetic. SB3 checks the KL *inside* the minibatch loop and bails out of
    the whole epoch loop, so a rollout can be abandoned partway through its first
    pass. Measured on run8's first 457 updates: 456 of them stopped at epoch 0.
    `--n-epochs 10` was decorative -- every sample collected was used once or
    less, which is the worst sample reuse PPO can have, and it had been that way
    for the entire 460M steps of run1-run6.

    If this sits at 1.0, the policy is moving the full `target_kl` budget on the
    first pass: lower `--lr` or raise `--batch-size` until a few epochs survive.
    """

    def __init__(self) -> None:
        super().__init__()
        self._last = 0

    def _on_rollout_end(self) -> None:
        n = int(getattr(self.model, "_n_updates", 0))
        if n > self._last:
            self.logger.record("train/epochs_used", n - self._last)
        self._last = n

    def _on_step(self) -> bool:
        return True


class ClampLogStdCallback(BaseCallback):
    """Hard cap on the policy's action std.

    A diagonal Gaussian's entropy has no upper bound in sigma. When the task
    reward is nearly flat -- which it is until the policy solves anything --
    an entropy bonus is the only consistent gradient in the loss, and PPO
    happily pushes sigma toward infinity. Observed on this task: std went
    0.6 -> 11 over 20M steps, so every sampled action saturated at the action
    limits and exploration died even though the curves looked healthy.
    """

    def __init__(self, max_log_std: float = 0.0) -> None:
        super().__init__()
        self.max_log_std = max_log_std

    def _on_rollout_end(self) -> None:
        log_std = getattr(self.model.policy, "log_std", None)
        if log_std is not None:
            log_std.data.clamp_(max=self.max_log_std)

    def _on_step(self) -> bool:
        return True


def resolve_tensorboard_dir(run_dir: Path) -> str | None:
    """TensorBoard is optional; SB3 raises if it is missing, so check first."""
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        print("tensorboard not installed -- logging to stdout only "
              "(`pip install tensorboard` to get curves)")
        return None
    return str(run_dir / "tb")


def make_env_fn(seed: int, rank: int, env_kwargs: dict):
    def _init():
        env = CubeReorientContinuous(**env_kwargs)
        env.reset(seed=seed + rank)
        return env

    return _init


def build_vec_env(
    n_envs: int,
    seed: int,
    env_kwargs: dict,
    subproc: bool,
    vecnormalize_path: Path | None = None,
    gamma: float = 0.99,
):
    fns = [make_env_fn(seed, i, env_kwargs) for i in range(n_envs)]
    venv = SubprocVecEnv(fns) if (subproc and n_envs > 1) else DummyVecEnv(fns)
    venv = VecMonitor(venv)

    if vecnormalize_path is not None and vecnormalize_path.exists():
        # Carry the running obs/reward statistics across. Starting them from
        # scratch would feed the resumed policy differently-scaled observations
        # than it was trained on, which looks exactly like the policy forgetting.
        normalized = VecNormalize.load(str(vecnormalize_path), venv)
        normalized.training = True
        normalized.norm_reward = True
        normalized.gamma = gamma
        print(f"resumed normalization stats from {vecnormalize_path}")
        return normalized

    # VecNormalize scales rewards by a running estimate of the *discounted*
    # return, with its own gamma (default 0.99) -- keep it in step with PPO's.
    return VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=gamma)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="ppo_reorient")
    p.add_argument("--timesteps", type=int, default=20_000_000,
                   help="steps to train in THIS invocation. With --resume they are added "
                        "on top of the checkpoint's count (SB3 reset_num_timesteps=False), "
                        "so resuming a 160M model with --timesteps 8_000_000 stops at 168M")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--no-subproc", action="store_true", help="single process (easier to debug)")

    # PPO
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1024,
                   help="was 256. At 256 / lr 3e-4 every one of run8's 628 updates hit "
                        "target_kl inside the first epoch, so --n-epochs 10 never ran; "
                        "replaying one rollout, 1024 / 1.5e-4 completes 4 epochs for the "
                        "same 16 gradient steps. Watch train/epochs_used")
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--gamma", type=float, default=0.995,
                   help="0.99 is a 1s horizon at this 100Hz control rate, but a goal "
                        "attempt runs up to 1.5s; the success bonus was being discounted "
                        "away before the policy could act on it")
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0,
                   help="0 by default: with a sparse reward the entropy bonus is the only\n                         consistent gradient and drives the action std to infinity")
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.02,
                   help="stop the epoch loop early when the policy has moved this far")
    p.add_argument("--log-std-init", type=float, default=-0.5,
                   help="initial action std = exp(this). SB3 default 0 (std=1) saturates this action space")

    # task
    p.add_argument("--action-mode", default="relative", choices=["relative", "absolute"])
    p.add_argument("--action-scale", type=float, default=0.15)
    p.add_argument("--max-episode-steps", type=int, default=400)
    p.add_argument("--randomize-physics", action="store_true")
    p.add_argument("--goal-mode", default="curriculum", choices=["curriculum", "axis", "random"])
    p.add_argument("--drop-mode", default="terminate", choices=["terminate", "reset_cube"],
                   help="reset_cube: a drop costs --drop-penalty, the hand and cube go back "
                        "to the episode's settled start with a new goal, and the episode "
                        "continues. Under terminate the real price of a drop is the penalty "
                        "plus the rest of the episode. Evaluation always uses terminate")
    p.add_argument("--goal-timeout", type=int, default=150,
                   help="steps a single goal attempt gets before it is retired unsolved "
                        "and a fresh one is drawn. 0 disables (the old behaviour, where "
                        "one unreachable goal burned the rest of the episode)")
    p.add_argument("--shaping-mode", default="angle", choices=["angle", "cos", "lookahead"],
                   help="'cos' is the original potential and its gradient vanishes at "
                        "the goal; 'angle' pays the same per degree everywhere")
    p.add_argument("--lookahead-s", type=float, default=0.15,
                   help="lookahead shaping: how far ahead (s) the cube's spin is extrapolated")
    p.add_argument("--lookahead-mix", type=float, default=0.5,
                   help="lookahead shaping: weight on the predicted angle vs the current one")
    p.add_argument("--freeze-potential-off-hand", action="store_true",
                   help="bill potential changes that happen while the cube is out of the hand "
                        "on the step it is regrasped. Required with --shaping-mode lookahead: "
                        "without it checks.py farms +4.8 per 300 steps by flicking the cube")
    p.add_argument("--curriculum-metric", default="goal_success",
                   choices=["goal_success", "episode_rate"])
    p.add_argument("--curriculum-window", type=int, default=40,
                   help="goal attempts (or episodes, for episode_rate) per decision")
    p.add_argument("--curriculum-start-deg", type=float, default=30.0)
    p.add_argument("--curriculum-step-deg", type=float, default=5.0,
                   help="2.5 gives a smoother difficulty curve than the default 5")
    p.add_argument("--curriculum-up-rate", type=float, default=0.55,
                   help="goal_success: fraction of attempts solved needed to add "
                        "--curriculum-step-deg. The old episode_rate bar of 1.0 solve "
                        "per episode was never reached and pinned difficulty at ~45 deg")
    p.add_argument("--curriculum-down-rate", type=float, default=0.25)
    p.add_argument("--hold-steps", type=int, default=10,
                   help="consecutive aligned steps a solve requires")
    p.add_argument("--success-tolerance-deg", type=float, default=15.0)
    p.add_argument("--spin-penalty", type=float, default=0.0,
                   help="penalise cube angular speed inside the goal cone. Off by default: "
                        "run9 (0.05, with --align-bonus 0.1) finished below both seeds of "
                        "the baseline without them (run10/run12) at 30, 45 and 60 deg")
    p.add_argument("--spin-band", type=float, default=1.0,
                   help="width of that band, in multiples of the success tolerance. "
                        "Keep it <= 1: at 2.0 the band swallows the whole 30-degree "
                        "curriculum goal range, and idling is taxed 2.6 per episode -- "
                        "which is how you get a policy that clamps the cube and freezes")
    p.add_argument("--align-bonus", type=float, default=0.0,
                   help="per-step reward while inside the cone, budgeted to hold_steps "
                        "payouts per goal so edge-hovering cannot farm it")
    p.add_argument("--max-log-std", type=float, default=0.0,
                   help="hard cap on log_std (0 => std <= 1). Guards the runaway above")
    p.add_argument("--checkpoint-every", type=int, default=500_000)
    p.add_argument("--resume", default=None, metavar="MODEL.zip",
                   help="continue training from a saved model; its vecnormalize.pkl "
                        "is picked up from the same folder")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path("runs") / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(
        action_mode=args.action_mode,
        action_scale=args.action_scale,
        max_episode_steps=args.max_episode_steps,
        randomize_physics=args.randomize_physics,
        goal_mode=args.goal_mode,
        goal_timeout_steps=args.goal_timeout,
        drop_mode=args.drop_mode,
        shaping_mode=args.shaping_mode,
        lookahead_s=args.lookahead_s,
        lookahead_mix=args.lookahead_mix,
        freeze_potential_off_hand=args.freeze_potential_off_hand,
        curriculum_start_deg=args.curriculum_start_deg,
        curriculum_step_deg=args.curriculum_step_deg,
        curriculum_metric=args.curriculum_metric,
        curriculum_window=args.curriculum_window,
        curriculum_up_rate=args.curriculum_up_rate,
        curriculum_down_rate=args.curriculum_down_rate,
        hold_steps=args.hold_steps,
        success_tolerance_rad=np.deg2rad(args.success_tolerance_deg),
        align_bonus=args.align_bonus,
        spin_penalty=args.spin_penalty,
        spin_band=args.spin_band,
    )

    resume_path = Path(args.resume) if args.resume else None
    stats_path = resolve_stats_path(resume_path) if resume_path else None
    (run_dir / ENV_KWARGS_FILE).write_text(json.dumps(env_kwargs, indent=2, default=float))
    if resume_path is not None and resume_path.exists():
        # Checkpoints from before the controller target joined the observation
        # (runs 1-8, 54-dim) can still be resumed, in their own layout.
        from stable_baselines3.common.save_util import load_from_zip_file
        saved, _, _ = load_from_zip_file(str(resume_path), load_data=True, device="cpu")
        if int(saved["observation_space"].shape[0]) == LEGACY_OBS_DIM:
            env_kwargs["obs_include_target"] = False
            print("resuming a legacy 54-dim policy: controller target NOT observed. "
                  "Start fresh to get the fixed observation.")

    venv = build_vec_env(
        args.n_envs, args.seed, env_kwargs,
        subproc=not args.no_subproc,
        vecnormalize_path=stats_path,
        gamma=args.gamma,
    )
    if stats_path is not None and not stats_path.exists():
        print(f"WARNING: {stats_path} not found -- observation statistics restart from "
              "zero, and the resumed policy will see differently-scaled inputs")

    if resume_path is not None:
        if not resume_path.exists():
            raise SystemExit(f"--resume: {resume_path} not found")
        model = PPO.load(
            str(resume_path),
            env=venv,
            device=args.device,
            # Hyperparameters are re-read from the command line, so a resumed run
            # picks up flag changes instead of silently reusing the old values.
            custom_objects=dict(
                learning_rate=args.lr,
                clip_range=args.clip_range,
                ent_coef=args.ent_coef,
                target_kl=args.target_kl,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                # These two were missing, so --gamma / --gae-lambda were
                # silently ignored on every resumed run despite the docstring.
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            ),
        )
        model.tensorboard_log = resolve_tensorboard_dir(run_dir)
        print(f"resumed policy from {resume_path} ({model.num_timesteps:,} steps already trained)")
    else:
        model = PPO(
            "MlpPolicy",
            venv,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                log_std_init=args.log_std_init,
            ),
            tensorboard_log=resolve_tensorboard_dir(run_dir),
            seed=args.seed,
            device=args.device,
            verbose=1,
        )

    callbacks = [
        TaskMetricsCallback(),
        EpochsUsedCallback(),
        ClampLogStdCallback(args.max_log_std),
        CheckpointCallback(
            save_freq=max(args.checkpoint_every // args.n_envs, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo",
            save_vecnormalize=True,
        ),
    ]

    print(f"obs {venv.observation_space.shape}  act {venv.action_space.shape}  envs {args.n_envs}")
    print(f"rollout = {args.n_envs} x {args.n_steps} = {args.n_envs * args.n_steps} transitions")
    print(f"logging to {run_dir}")

    # A run started in the background (`nohup ... &`) inherits SIGINT as
    # *ignored*, so Ctrl-C / `kill -INT` does nothing and the save below never
    # runs -- that is how run8 had to be killed without a final save. Route
    # SIGTERM (plain `kill <pid>`) into the same save-and-exit path.
    def _stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _stop)

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            reset_num_timesteps=resume_path is None,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\ninterrupted -- saving what we have")

    model.save(run_dir / "final_model")
    venv.save(str(run_dir / "vecnormalize.pkl"))
    venv.close()
    print(f"saved {run_dir/'final_model.zip'} and {run_dir/'vecnormalize.pkl'}")


if __name__ == "__main__":
    main()
