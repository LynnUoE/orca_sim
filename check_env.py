"""Sanity check for orca_sim: prints the env's shapes and runs a short random rollout."""

import time

import numpy as np

from orca_sim import OrcaHandRight, OrcaHandRightCubeOrientation


def report(name, env):
    obs, info = env.reset(seed=0)
    print(f"\n[{name}]")
    print(f"  obs shape      : {obs.shape}")
    print(f"  action shape   : {env.action_space.shape}")
    print(f"  nq / nv / nu   : {env.model.nq} / {env.model.nv} / {env.model.nu}")
    dt = env.model.opt.timestep * env.frame_skip
    print(f"  control period : {env.model.opt.timestep} s x {env.frame_skip} = {dt} s  ({1/dt:.0f} Hz)")
    return obs, info


def main():
    hand = OrcaHandRight()
    report("OrcaHandRight", hand)
    hand.close()

    env = OrcaHandRightCubeOrientation(version="v2")
    obs, info = report("OrcaHandRightCubeOrientation", env)
    print(f"  info keys      : {sorted(info)}")
    print(f"  start alignment: {info['red_face_up_alignment']:+.3f}  (-1 = red face down)")

    n_steps, successes, drops, timeouts = 2000, 0, 0, 0
    returns, ep_return = [], 0.0

    start = time.time()
    for step in range(n_steps):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        ep_return += reward
        if terminated or truncated:
            successes += bool(info["is_success"])
            drops += bool(info["dropped"])
            timeouts += bool(truncated and not terminated)
            returns.append(ep_return)
            ep_return = 0.0
            env.reset(seed=step)
    elapsed = time.time() - start

    print("\n[random policy rollout]")
    print(f"  {n_steps} steps in {elapsed:.1f}s  ->  {n_steps/elapsed:.0f} env-steps/s")
    print(f"  episodes       : {len(returns)}")
    print(f"  success / drop / timeout : {successes} / {drops} / {timeouts}")
    if returns:
        print(f"  mean return    : {np.mean(returns):.2f}   (random baseline)")
    env.close()

    print("\nEnvironment looks healthy.")


if __name__ == "__main__":
    main()