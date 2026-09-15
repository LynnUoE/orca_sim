"""Find out where a policy loses the task.

    python -m orca_rl.diagnose --model runs/run3/final_model.zip

A solve needs two separate things to go right: the red face has to enter the
15-degree cone, and it has to *stay* there for `hold_steps`. A single
solves/episode number cannot tell those apart, and the fixes point in opposite
directions -- so measure them separately before tuning anything.

Reads out:

  reach      how close the policy gets to each goal it is handed
  convert    of the times it does enter the cone, how often it holds long enough
  where time goes

Interpretation is printed at the bottom.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from orca_rl.task import CubeReorientContinuous


def build_actor(args):
    if args.policy == "random":
        return lambda env, obs: env.action_space.sample()
    if args.policy == "zero":
        return lambda env, obs: np.zeros(env.action_space.shape, dtype=np.float32)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model = PPO.load(args.model, device="cpu")
    stats = Path(args.vecnormalize) if args.vecnormalize else Path(args.model).parent / "vecnormalize.pkl"
    normalizer = None
    if stats.exists():
        normalizer = VecNormalize.load(str(stats), DummyVecEnv([lambda: CubeReorientContinuous()]))
        normalizer.training = False
        normalizer.norm_reward = False
    else:
        print(f"warning: {stats} not found -- observations will be unnormalized")

    def act(env, obs):
        x = obs[None, :]
        if normalizer is not None:
            x = normalizer.normalize_obs(x)
        action, _ = model.predict(x, deterministic=True)
        return action[0]

    return act


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None)
    p.add_argument("--vecnormalize", default=None)
    p.add_argument("--policy", default="model", choices=["model", "random", "zero"])
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--max-episode-steps", type=int, default=400)
    p.add_argument("--goal-angle", type=float, default=None,
                   help="pin the curriculum at this difficulty (default: whatever the env starts at)")
    p.add_argument("--hold-steps", type=int, default=10,
                   help="evaluate at the standard bar (10) even if training used a\n                         looser one, so numbers stay comparable across runs")
    p.add_argument("--seed", type=int, default=4321)
    args = p.parse_args()

    if args.policy == "model" and args.model is None:
        p.error("--model is required unless --policy is random or zero")

    act = build_actor(args)
    env = CubeReorientContinuous(max_episode_steps=args.max_episode_steps,
                                 hold_steps=args.hold_steps)
    if args.goal_angle is not None:
        env.goal_angle_deg = float(args.goal_angle)
        env.curriculum_min_deg = env.curriculum_max_deg = float(args.goal_angle)

    tol_deg = float(np.degrees(env.success_tolerance_rad))

    # Why in-cone runs end: overshoot (cube still spinning) vs slip (contact lost)
    exit_angvel: list[float] = []
    exit_reason: Counter[str] = Counter()

    goal_min_angles: list[float] = []     # closest approach per goal handed out
    goal_reached: list[bool] = []         # did that goal ever enter the cone
    hold_runs: list[int] = []             # length of each contiguous in-cone run
    solves = drops = 0
    steps_total = steps_aligned = steps_in_hand = 0
    end_reasons: Counter[str] = Counter()

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        if args.goal_angle is not None:
            env.goal_angle_deg = float(args.goal_angle)

        current_goal = tuple(info["goal_dir"])
        min_angle = info["angle_to_goal_deg"]
        entered = False
        run_len = 0

        for _ in range(args.max_episode_steps):
            obs, _, terminated, truncated, info = env.step(act(env, obs))
            steps_total += 1
            steps_in_hand += int(info["in_hand"])

            aligned_now = bool(info["aligned"] and info["in_hand"])
            steps_aligned += int(aligned_now)
            min_angle = min(min_angle, info["angle_to_goal_deg"])
            entered |= aligned_now

            if aligned_now:
                run_len += 1
            elif run_len:
                # A hold just ended short. Record why, and how fast the cube was
                # turning as it left -- that separates "arrived with too much
                # momentum and slid past" from "lost the grasp".
                hold_runs.append(run_len)
                angvel = float(np.linalg.norm(info["cube_qvel"][3:]))
                exit_angvel.append(angvel)
                if not info["in_hand"]:
                    exit_reason["lost contact / dropped below palm"] += 1
                elif angvel > 1.0:
                    exit_reason["overshoot (cube still spinning)"] += 1
                else:
                    exit_reason["drifted out slowly"] += 1
                run_len = 0

            if info["solved_this_step"]:
                solves += 1
                hold_runs.append(env.hold_steps)
                run_len = 0

            goal_now = tuple(info["goal_dir"])
            if goal_now != current_goal:                 # goal retired: record it
                goal_min_angles.append(min_angle)
                goal_reached.append(entered)
                current_goal = goal_now
                min_angle = info["angle_to_goal_deg"]
                entered = False

            if terminated or truncated:
                goal_min_angles.append(min_angle)
                goal_reached.append(entered)
                drops += int(bool(info["dropped"]))
                end_reasons["drop" if terminated else "timeout"] += 1
                break

        if run_len:
            hold_runs.append(run_len)

    env.close()

    goal_min_angles = np.asarray(goal_min_angles)
    reached = np.asarray(goal_reached)
    hold_runs_arr = np.asarray(hold_runs) if hold_runs else np.zeros(0)
    n_goals = len(goal_min_angles)
    # Runs that entered the cone but fell out before hold_steps.
    short_runs = hold_runs_arr[hold_runs_arr < env.hold_steps]

    label = args.policy if args.policy != "model" else Path(args.model).stem
    print(f"\n=== {label} | {args.episodes} episodes | goal angle {env.goal_angle_deg:.0f} deg "
          f"| tolerance {tol_deg:.0f} deg | hold {env.hold_steps} steps ===\n")

    print("REACH -- how close it gets to each goal")
    print(f"  goals handed out          : {n_goals}")
    if n_goals:
        pct = np.percentile(goal_min_angles, [10, 50, 90])
        print(f"  closest approach (deg)    : p10 {pct[0]:.1f} | median {pct[1]:.1f} | p90 {pct[2]:.1f}")
        print(f"  goals that entered the cone: {100 * reached.mean():.0f}%")

    print("\nCONVERT -- of the entries, how many hold long enough")
    print(f"  in-cone episodes (runs)   : {len(hold_runs_arr)}")
    print(f"  solves                    : {solves}")
    if len(hold_runs_arr):
        print(f"  conversion rate           : {100 * solves / len(hold_runs_arr):.0f}%")
        print(f"  median run length         : {np.median(hold_runs_arr):.0f} steps "
              f"(needs {env.hold_steps})")
    if len(short_runs):
        print(f"  failed runs, median length: {np.median(short_runs):.0f} steps "
              f"({len(short_runs)} of them)")

    if exit_angvel:
        ang = np.asarray(exit_angvel)
        print("\nWHY THE HOLD ENDS  (only runs that fell short)")
        print(f"  cube angular speed at exit : median {np.median(ang):.2f} rad/s "
              f"| p90 {np.percentile(ang, 90):.2f}")
        total_exits = sum(exit_reason.values())
        for reason, n in exit_reason.most_common():
            print(f"  {reason:<34}: {n:>3}  ({100 * n / total_exits:.0f}%)")

    print("\nWHERE TIME GOES")
    print(f"  steps                     : {steps_total}")
    print(f"  in hand                   : {100 * steps_in_hand / steps_total:.0f}%")
    print(f"  aligned and in hand       : {100 * steps_aligned / steps_total:.0f}%")
    print(f"  episode endings           : {dict(end_reasons)}")

    # ---- verdict -------------------------------------------------------
    print("\nVERDICT")
    enter_rate = reached.mean() if n_goals else 0.0
    conv = solves / len(hold_runs_arr) if len(hold_runs_arr) else 0.0

    if enter_rate < 0.35:
        print("  REACH-limited. It rarely gets the red face into the cone at all.")
        print("  -> lower the difficulty (--curriculum-start-deg), widen the tolerance,")
        print("     or give it more time (--max-episode-steps 800). Loosening hold_steps")
        print("     will not help; it is not getting there in the first place.")
    elif conv < 0.5:
        print("  HOLD-limited. It reaches the cone but slides out before the hold completes.")
        print("  -> shorten hold_steps, or add a reward term for staying aligned so the")
        print("     policy is paid for stabilizing rather than only for arriving.")
    else:
        print("  Balanced -- neither reaching nor holding is the clear bottleneck.")
        print("  -> the limit is the sample budget. More steps, or more parallel envs.")

    if exit_angvel and conv < 0.5:
        overshoot = exit_reason["overshoot (cube still spinning)"]
        slip = exit_reason["lost contact / dropped below palm"]
        if overshoot >= max(slip, 1) :
            print("  Failures look like OVERSHOOT: the cube is still turning when it")
            print("  leaves the cone. Near the goal the shaping term is ~0, so nothing")
            print("  currently pays the policy to brake. Try penalising cube angular")
            print("  speed while inside the cone -- unlike a flat in-cone bonus, holding")
            print("  the fingers still does not stop an already-spinning cube, so this")
            print("  rewards an actual skill rather than rewarding stillness.")
        elif slip > overshoot:
            print("  Failures look like SLIP: contact is lost while aligned. Look at grasp")
            print("  quality -- friction, finger placement, or a penalty on losing contact.")
    print()


if __name__ == "__main__":
    main()
