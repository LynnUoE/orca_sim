"""PPO baseline for continuous in-hand cube reorientation.

    python -m orca_rl.train --timesteps 20_000_000 --n-envs 8

Everything lands in ``runs/<name>/``: checkpoints, the VecNormalize statistics
(needed at evaluation time), and TensorBoard logs.

    tensorboard --logdir runs/
"""

from __future__ import annotations

import argparse
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

from orca_rl.task import CubeReorientContinuous


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

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if not done:
                continue
            self._successes.append(int(info.get("episode_successes", 0)))
            self._drops.append(float(bool(info.get("dropped", False))))
            self._lengths.append(int(info.get("elapsed_steps", 0)))
            self._goal_angles.append(float(info.get("goal_angle_deg", 0.0)))
            self._solve_rates.append(float(info.get("curriculum_solve_rate", 0.0)))

        if len(self._successes) >= self.window:
            self.logger.record("task/solves_per_episode", float(np.mean(self._successes)))
            self.logger.record("task/solved_any_frac", float(np.mean([s > 0 for s in self._successes])))
            self.logger.record("task/drop_rate", float(np.mean(self._drops)))
            self.logger.record("task/episode_length", float(np.mean(self._lengths)))
            self.logger.record("task/goal_angle_deg", float(np.mean(self._goal_angles)))
            self.logger.record("task/curriculum_solve_rate", float(np.mean(self._solve_rates)))
            self._successes.clear()
            self._drops.clear()
            self._lengths.clear()
            self._goal_angles.clear()
            self._solve_rates.clear()
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
        print(f"resumed normalization stats from {vecnormalize_path}")
        return normalized

    return VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="ppo_reorient")
    p.add_argument("--timesteps", type=int, default=20_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--no-subproc", action="store_true", help="single process (easier to debug)")

    # PPO
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
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
    p.add_argument("--curriculum-start-deg", type=float, default=30.0)
    p.add_argument("--curriculum-step-deg", type=float, default=5.0,
                   help="2.5 gives a smoother difficulty curve than the default 5")
    p.add_argument("--curriculum-up-rate", type=float, default=1.0)
    p.add_argument("--curriculum-down-rate", type=float, default=0.25)
    p.add_argument("--hold-steps", type=int, default=10,
                   help="consecutive aligned steps a solve requires")
    p.add_argument("--success-tolerance-deg", type=float, default=15.0)
    p.add_argument("--spin-penalty", type=float, default=0.0,
                   help="penalise cube angular speed while inside the tolerance cone. "
                        "Targets overshoot, which diagnose.py measures as ~85%% of failed holds")
    p.add_argument("--align-bonus", type=float, default=0.0,
                   help="per-step reward while inside the cone; capped by hold_steps "
                        "because the solve then fires and the goal moves")
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
        curriculum_start_deg=args.curriculum_start_deg,
        curriculum_step_deg=args.curriculum_step_deg,
        curriculum_up_rate=args.curriculum_up_rate,
        curriculum_down_rate=args.curriculum_down_rate,
        hold_steps=args.hold_steps,
        success_tolerance_rad=np.deg2rad(args.success_tolerance_deg),
        align_bonus=args.align_bonus,
        spin_penalty=args.spin_penalty,
    )

    resume_path = Path(args.resume) if args.resume else None
    stats_path = (resume_path.parent / "vecnormalize.pkl") if resume_path else None

    venv = build_vec_env(
        args.n_envs, args.seed, env_kwargs,
        subproc=not args.no_subproc,
        vecnormalize_path=stats_path,
    )

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
