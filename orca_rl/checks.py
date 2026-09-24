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


def edge_hover_policy(env, inside_steps: int = 5, outside_steps: int = 5):
    """Oscillate across the tolerance cone edge without ever completing a hold.

    This is the exploit `align_bonus` invites. The hold counter resets every
    time the cube leaves the cone, so an unbudgeted per-step in-cone bonus pays
    out forever: it is the stock task's stalling exploit wearing a new hat. The
    budget in `CubeReorientContinuous.step` caps the payout at `hold_steps` per
    goal, which this measures.
    """
    adr = env._cube_qpos_adr
    period = inside_steps + outside_steps

    def policy(e, t):
        tol = e.success_tolerance_rad
        offset = tol * (0.5 if (t % period) < inside_steps else 1.5)
        # Tip the red face `offset` off the goal, around an arbitrary axis.
        goal = e._goal_dir / (np.linalg.norm(e._goal_dir) + 1e-12)
        axis = np.cross(goal, [0.0, 0.0, 1.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(goal, [1.0, 0.0, 0.0])
        axis /= np.linalg.norm(axis)
        target = (
            goal * np.cos(offset)
            + np.cross(axis, goal) * np.sin(offset)
        )
        e.data.qpos[adr + 3 : adr + 7] = shortest_arc_quat(target)
        ang = e._cube_qvel_adr + 3
        e.data.qvel[ang : ang + 3] = 0.0
        mujoco.mj_forward(e.model, e.data)
        return np.zeros(e.action_space.shape, dtype=np.float32)

    return policy


def zero_policy_free_solves(episodes: int = 30) -> int:
    """Solves a motionless hand collects at the easiest curriculum angle.

    Must be zero. It was 26% of episodes at 30 degrees: the cube tipped ~21
    degrees while dropping onto the palm after reset, and the goal had been
    drawn from its pre-drop orientation. A single seed-0 episode (the check
    above) happens to miss it, which is how it survived runs 1-8.
    """
    env = CubeReorientContinuous()
    env.goal_angle_deg = env.curriculum_max_deg = env.curriculum_min_deg
    solves = 0
    for ep in range(episodes):
        solves += run(env, lambda e, t: np.zeros(e.action_space.shape), 400, seed=100 + ep)["solves"]
    env.close()
    return solves


def reset_cube_drops(episodes: int = 5) -> dict:
    """drop_mode="reset_cube" under a random policy (which drops constantly).

    A drop must not end the episode, must cost exactly `drop_penalty` (plus the
    usual small action-rate term), must leave the cube back in the hand, and
    must not hand out a free solve on the step after it is put back.
    """
    env = CubeReorientContinuous(drop_mode="reset_cube")
    out = {"drops": 0, "terminated": 0, "steps": 0, "worst_excess": 0.0,
           "back_in_hand": 0, "solve_after_drop": 0}
    for ep in range(episodes):
        env.reset(seed=200 + ep)
        just_dropped = False
        for t in range(400):
            _, reward, terminated, truncated, info = env.step(env.action_space.sample())
            out["steps"] += 1
            out["terminated"] += int(terminated)
            if just_dropped and info["solved_this_step"]:
                out["solve_after_drop"] += 1
            just_dropped = info["dropped_this_step"]
            if just_dropped:
                out["drops"] += 1
                out["back_in_hand"] += int(info["in_hand"])
                # shaping is 0 on a drop step (cube not held), so everything
                # beyond -drop_penalty is the action-rate term, which is tiny
                out["worst_excess"] = max(out["worst_excess"], abs(reward + env.drop_penalty))
            if terminated or truncated:
                break
    env.close()
    return out


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

    hover_env = CubeReorientContinuous(max_episode_steps=steps)
    rows.append(("hover on the cone edge", run(hover_env, edge_hover_policy(hover_env), steps, seed=3)))
    hover_env.close()

    print(f"  {'policy':<28}{'return':>10}{'solves':>9}{'steps':>8}")
    print("  " + "-" * 55)
    for label, res in rows:
        print(f"  {label:<28}{res['return']:>10.2f}{res['solves']:>9}{res['steps']:>8}")

    do_nothing = rows[0][1]
    oracle = rows[2][1]
    hover = rows[3][1]

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

    free = zero_policy_free_solves()
    check = free == 0
    ok &= check
    print(f"    [{'ok' if check else 'FAIL'}] a motionless hand never solves, even at the "
          f"easiest curriculum angle ({free} solves in 30 episodes)")

    rc = reset_cube_drops()
    check = (rc["drops"] > 0 and rc["terminated"] == 0 and rc["steps"] == 5 * 400
             and rc["back_in_hand"] == rc["drops"] and rc["worst_excess"] < 0.05
             and rc["solve_after_drop"] == 0)
    ok &= check
    print(f"    [{'ok' if check else 'FAIL'}] drop_mode=reset_cube: {rc['drops']} random-policy drops, "
          f"0 terminations ({rc['terminated']}), each costs drop_penalty "
          f"(max deviation {rc['worst_excess']:.4f}), cube back in hand "
          f"{rc['back_in_hand']}/{rc['drops']}, free solves after a drop {rc['solve_after_drop']}")

    # 400 steps of edge-hovering would collect 40 payouts if align_bonus were
    # unbudgeted. Budgeted, it can only ever collect hold_steps per goal, and
    # hovering never solves so the goal only changes on the timeout.
    env_ref = CubeReorientContinuous()
    ceiling = env_ref.align_bonus * env_ref.hold_steps * (
        1 + steps // max(env_ref.goal_timeout_steps or steps, 1)
    )
    env_ref.close()
    check = hover["solves"] == 0 and hover["return"] <= ceiling + 1.0
    ok &= check
    print(f"    [{'ok' if check else 'FAIL'}] hovering on the cone edge cannot be farmed "
          f"({hover['return']:+.2f}, {hover['solves']} solves, budget ceiling {ceiling:.1f})")

    print("\n  " + ("all checks passed" if ok else "SOMETHING IS WRONG -- do not train"))
    print()


if __name__ == "__main__":
    main()
