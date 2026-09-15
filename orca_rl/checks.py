"""Prove the reward exploit is gone.

    python -m orca_rl.checks

Runs four scripted policies against both the stock ``orca_sim`` task and the
fixed one, and prints what each earns. The point of the table is a single
comparison: under the stock reward, *stalling outranks solving by ~20x*.
Under the fixed reward it does not.

Run this before trusting any training curve.
"""

from __future__ import annotations

import numpy as np
import mujoco

from orca_sim import OrcaHandRightCubeOrientation
from orca_rl.task import CubeReorientContinuous

TOL = 1e-9


def stock_stall_vs_solve() -> tuple[float, float]:
    """Score the two strategies analytically on the stock reward.

    Stock reward is 0.5*(alignment+1) per step plus a small lift bonus, and
    _get_terminated() ends the episode the moment the goal is reached.
    """
    env = OrcaHandRightCubeOrientation(version="v2")
    env.reset(seed=0)
    tol = env.success_tolerance_rad
    horizon = env.max_episode_steps

    # Stall: sit just outside the tolerance for the whole episode.
    just_outside = np.cos(tol + np.deg2rad(1.0))
    stall = horizon * 0.5 * (just_outside + 1.0)

    # Solve: ramp from red-face-down to solved over 20 steps, then terminate.
    n = 20
    ramp = np.linspace(-1.0, np.cos(tol), n)
    solve = float(np.sum(0.5 * (ramp + 1.0)))

    env.close()
    return stall, solve


def run(env, policy, steps: int, seed: int = 0) -> dict:
    obs, info = env.reset(seed=seed)
    total, solves = 0.0, 0
    for t in range(steps):
        obs, reward, terminated, truncated, info = env.step(policy(env, t))
        total += reward
        solves += int(info.get("solved_this_step", False))
        if terminated or truncated:
            break
    return {"return": total, "solves": solves, "steps": t + 1, "info": info}


def shortest_arc_quat(target: np.ndarray) -> np.ndarray:
    """Quaternion rotating the cube's local red-face normal (0,0,1) onto `target`.

    Exact for any target direction, unlike picking the nearest axis-aligned
    orientation -- curriculum goals are arbitrary directions, not axis-aligned.
    """
    a = np.array([0.0, 0.0, 1.0])
    b = np.asarray(target, dtype=np.float64)
    b = b / (np.linalg.norm(b) + 1e-12)
    dot = float(np.dot(a, b))
    if dot < -1.0 + 1e-8:                      # antipodal: any perpendicular axis
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(a, b)
    quat = np.array([1.0 + dot, axis[0], axis[1], axis[2]])
    return quat / (np.linalg.norm(quat) + 1e-12)


def oracle_policy(env):
    """Pin the cube on the goal each step, hand held still.

    Not a real policy -- it is the upper bound the reward should pay out.

    Only the *orientation* is forced (plus zeroed angular velocity). Position is
    left to physics so the cube keeps resting on the palm: pinning it to the
    spawn height instead leaves it hovering once the hand sags, contact is lost,
    and `_cube_in_hand` stops counting -- which silently caps the oracle at one
    solve.
    """
    adr = env._cube_qpos_adr

    def policy(e, t):
        e.data.qpos[adr + 3 : adr + 7] = shortest_arc_quat(e._goal_dir)
        ang = e._cube_qvel_adr + 3
        e.data.qvel[ang : ang + 3] = 0.0
        mujoco.mj_forward(e.model, e.data)
        return np.zeros(e.action_space.shape, dtype=np.float32)

    return policy


def main() -> None:
    print("\n" + "=" * 68)
    print("STOCK TASK  (orca_sim.OrcaHandRightCubeOrientation)")
    print("=" * 68)
    stall, solve = stock_stall_vs_solve()
    print(f"  hover just outside tolerance for 200 steps : {stall:8.1f}")
    print(f"  actually solve in 20 steps, then terminate : {solve:8.1f}")
    print(f"  --> stalling is {stall / max(solve, TOL):.0f}x better. PPO will stall.")

    print("\n" + "=" * 68)
    print("FIXED TASK  (orca_rl.task.CubeReorientContinuous)")
    print("=" * 68)

    steps = 400
    rows = []

    env = CubeReorientContinuous(max_episode_steps=steps)
    rows.append(("do nothing", run(env, lambda e, t: np.zeros(e.action_space.shape), steps)))
    rows.append(("random", run(env, lambda e, t: e.action_space.sample(), steps, seed=1)))
    env.close()

    oracle_env = CubeReorientContinuous(max_episode_steps=steps)
    rows.append(("oracle (pinned on goal)", run(oracle_env, oracle_policy(oracle_env), steps, seed=2)))
    oracle_env.close()

    print(f"  {'policy':<28}{'return':>10}{'solves':>9}{'steps':>8}")
    print("  " + "-" * 55)
    for label, res in rows:
        print(f"  {label:<28}{res['return']:>10.2f}{res['solves']:>9}{res['steps']:>8}")

    do_nothing = rows[0][1]
    oracle = rows[2][1]

    print("\n  assertions:")
    ok = True

    check = abs(do_nothing["return"]) < 1.0
    ok &= check
    print(f"    [{'ok' if check else 'FAIL'}] doing nothing earns ~0 "
          f"(got {do_nothing['return']:+.2f}) -- no stalling exploit")

    check = oracle["solves"] >= 2
    ok &= check
    print(f"    [{'ok' if check else 'FAIL'}] solving repeatedly is possible "
          f"({oracle['solves']} solves) -- goals resample")

    check = oracle["return"] > do_nothing["return"] + 5.0
    ok &= check
    print(f"    [{'ok' if check else 'FAIL'}] solving pays far more than idling "
          f"({oracle['return']:+.2f} vs {do_nothing['return']:+.2f})")

    print("\n  " + ("all checks passed" if ok else "SOMETHING IS WRONG -- do not train"))
    print()


if __name__ == "__main__":
    main()
