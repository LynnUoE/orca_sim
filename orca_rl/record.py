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

from orca_rl.task import CubeReorientContinuous, policy_env_kwargs, resolve_stats_path

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
    obs_kw = policy_env_kwargs(model, args.model, action_scale=args.action_scale)
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


GOAL_RGBA = np.array([0.1, 0.85, 0.2, 1.0], dtype=np.float32)   # where the red face should point
FACE_RGBA = np.array([0.95, 0.15, 0.1, 1.0], dtype=np.float32)   # where it points now
ARROW_LEN = 0.05
# Both arrows share an origin floating above the cube: the hand never hides
# them, and the angle between them *is* the distance to the goal.
GAUGE_OFFSET = np.array([0.0, 0.0, 0.08])


def _add_arrow(scene: mujoco.MjvScene, start: np.ndarray, direction: np.ndarray, rgba: np.ndarray) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                        np.eye(3).flatten(), rgba)
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.004,
                         start, start + ARROW_LEN * np.asarray(direction))
    scene.ngeom += 1


def _font(size: int):
    from PIL import ImageFont
    for name in ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Helvetica.ttc",
                 "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _overlay(pixels: np.ndarray, top: str, bottom: str, border: tuple | None, banner: str | None) -> np.ndarray:
    from PIL import Image, ImageDraw
    img = Image.fromarray(pixels)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    small, big = _font(max(12, h // 30)), _font(max(18, h // 12))
    for text, y in ((top, 6), (bottom, h - h // 30 - 12)):
        draw.rectangle([0, y - 4, w, y + h // 30 + 6], fill=(0, 0, 0))
        draw.text((8, y), text, fill=(255, 255, 255), font=small)
    if border is not None:
        draw.rectangle([0, 0, w - 1, h - 1], outline=border, width=max(4, h // 60))
    if banner:
        box = draw.textbbox((0, 0), banner, font=big)
        bw, bh = box[2] - box[0], box[3] - box[1]
        draw.rectangle([(w - bw) // 2 - 10, h // 2 - bh, (w + bw) // 2 + 10, h // 2 + bh // 2 + 6],
                       fill=(0, 0, 0))
        draw.text(((w - bw) // 2, h // 2 - bh), banner, fill=border or (255, 255, 255), font=big)
    return np.asarray(img)


def rollout_frames(act, env, renderer, *, seeds, max_steps: int, label: str,
                   goal_angle: float | None = None, pad: bool = False) -> tuple[list, dict]:
    """Render `act` on `env` for each seed, with the goal drawn in.

    Two arrows float above the cube from a shared origin -- green: where the
    red face has to point, red: where it points now. A solve flashes the frame green; a drop turns it red and, with
    `pad=True`, freezes the last frame until `max_steps` so several policies
    rolled out on the same seeds stay frame-aligned for a side-by-side.
    """
    frames, totals = [], {"solves": 0, "goals": 0, "drops": 0}
    for ep, seed in enumerate(seeds):
        obs, info = env.reset(seed=int(seed))
        if goal_angle is not None:
            env.goal_angle_deg = float(goal_angle)
        cam = make_camera(env)
        solves, flash, last = 0, 0, None
        for t in range(max_steps):
            obs, _, terminated, truncated, info = env.step(act(env, obs))
            if info["solved_this_step"]:
                solves += 1
                flash = 25
            cam.lookat[:] = env.data.xpos[env._cube_body_id]
            renderer.update_scene(env.data, camera=cam)
            gauge = env.data.xpos[env._cube_body_id] + GAUGE_OFFSET
            _add_arrow(renderer.scene, gauge, env._goal_dir, GOAL_RGBA)
            _add_arrow(renderer.scene, gauge, env._cube_red_face_world_normal(), FACE_RGBA)
            dropped = bool(info["dropped"])
            top = f"{label}   ep {ep + 1}/{len(seeds)}   t={(t + 1) * env.model.opt.timestep * env.frame_skip:4.2f}s"
            bottom = (f"to goal {info['angle_to_goal_deg']:5.1f} deg   solves {solves}"
                      f"   {'held' if info['in_hand'] else 'NOT held'}")
            border = (40, 220, 60) if flash > 0 else ((230, 40, 30) if dropped else None)
            banner = "SOLVED" if flash > 20 else ("DROPPED" if dropped else None)
            last = _overlay(renderer.render(), top, bottom, border, banner)
            frames.append(last)
            flash = max(flash - 1, 0)
            if terminated or truncated:
                break
        totals["solves"] += solves
        totals["goals"] += int(info["episode_goals"]) + 1   # + the attempt still running
        totals["drops"] += int(bool(info["dropped"]))
        if pad:
            frames.extend([last] * (max_steps - (t + 1)))
    return frames, totals


def make_env_for(obs_kw: dict, max_steps: int, goal_angle: float | None) -> CubeReorientContinuous:
    env = CubeReorientContinuous(max_episode_steps=max_steps, **obs_kw)
    if goal_angle is not None:
        env.goal_angle_deg = env.curriculum_min_deg = env.curriculum_max_deg = float(goal_angle)
    return env


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--action-scale", type=float, default=None,
                   help="override the action scale the model was trained with (read from "
                        "the run's env_kwargs.json; 0.15 for runs without one)")
    p.add_argument("--model", default=None)
    p.add_argument("--vecnormalize", default=None)
    p.add_argument("--policy", default="model", choices=["model", "random", "zero"])
    p.add_argument("--out", default="rollout.mp4")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-episode-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--goal-angle", type=float, default=None,
                   help="pin the difficulty, so videos of different runs show the same task")
    p.add_argument("--label", default=None, help="text in the top bar (default: model path)")
    p.add_argument("--fps", type=int, default=50,
                   help="50 = half speed; the sim runs at 100 control steps per second")
    args = p.parse_args()

    if args.policy == "model" and args.model is None:
        p.error("--model is required unless --policy is random or zero")

    act, obs_kw = build_actor(args)
    env = make_env_for(obs_kw, args.max_episode_steps, args.goal_angle)
    renderer = mujoco.Renderer(env.model, height=HEIGHT, width=WIDTH)
    label = args.label or (args.policy if args.policy != "model" else str(Path(args.model).parent))
    seeds = [args.seed + ep for ep in range(args.episodes)]
    frames, totals = rollout_frames(act, env, renderer, seeds=seeds,
                                    max_steps=args.max_episode_steps, label=label,
                                    goal_angle=args.goal_angle)
    renderer.close()
    env.close()

    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"wrote {args.out}  ({len(frames)} frames, {totals['solves']} solves, "
          f"{totals['drops']} drops over {args.episodes} episodes)")


if __name__ == "__main__":
    main()
