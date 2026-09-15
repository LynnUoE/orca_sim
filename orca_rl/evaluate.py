"""Honest evaluation for the reorientation task.

    python -m orca_rl.evaluate --model runs/ppo_reorient/final_model.zip
    python -m orca_rl.evaluate --policy random        # baseline, no model needed
    python -m orca_rl.evaluate --policy zero          # do-nothing baseline

Every reported success requires the cube to be **in contact with the hand**,
**above ``in_hand_height``**, and **aligned for ``hold_steps`` consecutive
steps**. Under the stock ``orca_sim`` criterion a random policy "succeeds" 21%
of the time by flinging the cube and catching a lucky frame mid-tumble; those
do not count here.

Always compare a trained policy against both baselines. A policy that cannot
beat `zero` has learned nothing except to avoid dropping.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from orca_rl.task import CubeReorientContinuous


def load_policy(args):
    if args.policy == "random":
        return lambda env, obs: env.action_space.sample(), None
    if args.policy == "zero":
        return lambda env, obs: np.zeros(env.action_space.shape, dtype=np.float32), None

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model = PPO.load(args.model, device="cpu")

    normalizer = None
    if args.vecnormalize:
        stats = Path(args.vecnormalize)
        if stats.exists():
            dummy = DummyVecEnv([lambda: CubeReorientContinuous()])
            normalizer = VecNormalize.load(str(stats), dummy)
            normalizer.training = False
            normalizer.norm_reward = False
        else:
            print(f"warning: {stats} not found, evaluating without obs normalization")

    def act(env, obs):
        x = obs[None, :]
        if normalizer is not None:
            x = normalizer.normalize_obs(x)
        action, _ = model.predict(x, deterministic=args.deterministic)
        return action[0]

    return act, model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None, help="path to a saved PPO .zip")
    p.add_argument("--vecnormalize", default=None, help="path to vecnormalize.pkl")
    p.add_argument("--policy", default="model", choices=["model", "random", "zero"])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-episode-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--stochastic", dest="deterministic", action="store_false")
    p.add_argument("--randomize-physics", action="store_true")
    p.add_argument("--goal-angle", type=float, default=None,
                   help="pin the curriculum at this difficulty for the whole run. Without it "
                        "the curriculum keeps adapting DURING evaluation, so a stronger policy "
                        "gets handed harder goals and cross-run numbers are not comparable")
    p.add_argument("--hold-steps", type=int, default=10,
                   help="evaluate at the standard bar (10) even if training used a\n                         looser one, so numbers stay comparable across runs")
    p.add_argument("--action-mode", default="relative", choices=["relative", "absolute"])
    args = p.parse_args()

    if args.policy == "model" and args.model is None:
        p.error("--model is required unless --policy is random or zero")
    if args.model and args.vecnormalize is None:
        guess = Path(args.model).parent / "vecnormalize.pkl"
        args.vecnormalize = str(guess) if guess.exists() else None

    act, _ = load_policy(args)

    env = CubeReorientContinuous(
        max_episode_steps=args.max_episode_steps,
        action_mode=args.action_mode,
        randomize_physics=args.randomize_physics,
        hold_steps=args.hold_steps,
    )
    if args.goal_angle is not None:
        # Freeze the curriculum: same difficulty for every episode and every run.
        env.goal_angle_deg = float(args.goal_angle)
        env.curriculum_min_deg = env.curriculum_max_deg = float(args.goal_angle)

    returns, solves, drops, lengths = [], [], [], []
    first_solve_steps, in_hand_frac = [], []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        if args.goal_angle is not None:
            env.goal_angle_deg = float(args.goal_angle)
        total, steps_in_hand, first_solve = 0.0, 0, None

        for step in range(args.max_episode_steps):
            obs, reward, terminated, truncated, info = env.step(act(env, obs))
            total += reward
            steps_in_hand += int(info["in_hand"])
            if info["solved_this_step"] and first_solve is None:
                first_solve = step + 1
            if terminated or truncated:
                break

        returns.append(total)
        solves.append(info["successes"])
        drops.append(bool(info["dropped"]))
        lengths.append(info["elapsed_steps"])
        in_hand_frac.append(steps_in_hand / max(info["elapsed_steps"], 1))
        if first_solve is not None:
            first_solve_steps.append(first_solve)

    env.close()

    solves = np.asarray(solves)
    label = args.policy if args.policy != "model" else Path(args.model).stem

    pinned = f"goal angle {args.goal_angle:.0f} deg" if args.goal_angle is not None else "curriculum adapting (NOT comparable across runs)"
    print(f"\n=== {label}  ({args.episodes} episodes | {pinned} | hold {args.hold_steps}) ===")
    print(f"  solves / episode      : {solves.mean():.2f}  (max {solves.max()})")
    print(f"  episodes with >=1     : {100 * (solves > 0).mean():.0f}%")
    print(f"  drop rate             : {100 * np.mean(drops):.0f}%")
    print(f"  mean episode length   : {np.mean(lengths):.0f} / {args.max_episode_steps}")
    print(f"  fraction of steps held: {100 * np.mean(in_hand_frac):.0f}%")
    if first_solve_steps:
        print(f"  steps to first solve  : {np.mean(first_solve_steps):.0f}  (n={len(first_solve_steps)})")
    else:
        print("  steps to first solve  : never")
    print(f"  mean return           : {np.mean(returns):.2f} +- {np.std(returns):.2f}")
    print()
    print("  reference: zero policy ~0.0 solves, 0% drop | random ~0.03 solves, ~88% drop")


if __name__ == "__main__":
    main()
