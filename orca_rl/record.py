"""Record a trained (or random) policy to MP4.

    python -m orca_rl.record --model runs/ppo_reorient/final_model.zip
    python -m orca_rl.record --policy random --out random.mp4

Watch this every time the curves move. On this task the numbers are easy to
fool and the video is not: a policy can post a healthy return while doing
something obviously wrong, and ten seconds of footage settles it.

On macOS this runs under plain ``python`` -- offscreen rendering does not need
``mjpython``. Only the interactive viewer does.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from orca_rl.task import CubeReorientContinuous, obs_kwargs_for_model, resolve_stats_path

WIDTH, HEIGHT = 640, 480


def make_camera(env) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.model, cam)
    cam.azimuth = 140.0
    cam.elevation = -25.0
    cam.distance = 0.32
    cam.lookat[:] = env.data.xpos[env._cube_body_id]
    return cam


def build_actor(args):
    if args.policy == "random":
        return (lambda env, obs: env.action_space.sample()), {}
    if args.policy == "zero":
        return (lambda env, obs: np.zeros(env.action_space.shape, dtype=np.float32)), {}

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model = PPO.load(args.model, device="cpu")
    obs_kw = obs_kwargs_for_model(model)
    normalizer = None
    stats = Path(args.vecnormalize) if args.vecnormalize else resolve_stats_path(Path(args.model))
    if stats.exists():
        normalizer = VecNormalize.load(str(stats), DummyVecEnv([lambda: CubeReorientContinuous(**obs_kw)]))
        normalizer.training = False
        normalizer.norm_reward = False

    def act(env, obs):
        x = obs[None, :]
        if normalizer is not None:
            x = normalizer.normalize_obs(x)
        action, _ = model.predict(x, deterministic=True)
        return action[0]

    return act, obs_kw


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None)
    p.add_argument("--vecnormalize", default=None)
    p.add_argument("--policy", default="model", choices=["model", "random", "zero"])
    p.add_argument("--out", default="rollout.mp4")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-episode-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--fps", type=int, default=50)
    args = p.parse_args()

    if args.policy == "model" and args.model is None:
        p.error("--model is required unless --policy is random or zero")

    act, obs_kw = build_actor(args)
    env = CubeReorientContinuous(max_episode_steps=args.max_episode_steps, **obs_kw)
    renderer = mujoco.Renderer(env.model, height=HEIGHT, width=WIDTH)

    frames, solved_total = [], 0
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        cam = make_camera(env)
        for _ in range(args.max_episode_steps):
            obs, _, terminated, truncated, info = env.step(act(env, obs))
            solved_total += int(info["solved_this_step"])
            cam.lookat[:] = env.data.xpos[env._cube_body_id]
            renderer.update_scene(env.data, camera=cam)
            frames.append(renderer.render())
            if terminated or truncated:
                break
        print(f"  episode {ep}: {info['successes']} solves, dropped={info['dropped']}, {info['elapsed_steps']} steps")

    renderer.close()
    env.close()

    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"wrote {args.out}  ({len(frames)} frames, {solved_total} solves total)")


if __name__ == "__main__":
    main()
