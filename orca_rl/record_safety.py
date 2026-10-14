"""Side-by-side video of one policy with the Orca_safety layer off and on.

    python -m orca_rl.record_safety --model runs/run23/final_model.zip --goal-angle 45 \
        --seeds 2 3 --out safety_off_on.mp4

Same policy, same reset seeds, same goals. Left: the joint targets go straight
to the servos. Right: they pass through the CBF-QP filter first. The bar under
each panel is the hand's smallest self-clearance (red: links in contact,
orange: closer than the 5 mm margin, green: clear). See evaluate_safety.py for
the numbers; this is footage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from orca_rl.evaluate_safety import SafeCubeEnv, SafetyLayer, find_urdf
from orca_rl.record import FACE_RGBA, GAUGE_OFFSET, GOAL_RGBA, _add_arrow, make_camera
from orca_rl.task import CubeReorientContinuous, policy_env_kwargs, resolve_stats_path

W, H, BAR = 640, 480, 46


def _font(size: int):
    for name in ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Helvetica.ttc", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def rollout(mode: str, args, model, normalizer, kwargs, urdf: Path) -> list[np.ndarray]:
    env = SafeCubeEnv(**kwargs)
    env.goal_angle_deg = env.curriculum_min_deg = env.curriculum_max_deg = float(args.goal_angle)
    env.safety = SafetyLayer(env, urdf, filtering=mode == "on", config_path=args.safety_config)
    margin = env.safety.config.safe_distance
    renderer = mujoco.Renderer(env.model, height=H, width=W)
    font, small = _font(20), _font(15)
    frames = []
    for seed in args.seeds:
        obs, info = env.reset(seed=int(seed))
        env.goal_angle_deg = float(args.goal_angle)
        cam = make_camera(env)
        solves, flash, contact_steps = 0, 0, 0
        for t in range(args.steps):
            action, _ = model.predict(normalizer.normalize_obs(obs[None, :]), deterministic=True)
            obs, _, terminated, truncated, info = env.step(action[0])
            if info["solved_this_step"]:
                solves, flash = solves + 1, 20
            nearest, collided = env.safety.clearance(env)
            contact_steps += int(collided)
            cam.lookat[:] = env.data.xpos[env._cube_body_id]
            renderer.update_scene(env.data, camera=cam)
            gauge = env.data.xpos[env._cube_body_id] + GAUGE_OFFSET
            _add_arrow(renderer.scene, gauge, env._goal_dir, GOAL_RGBA)
            _add_arrow(renderer.scene, gauge, env._cube_red_face_world_normal(), FACE_RGBA)
            img = Image.new("RGB", (W, H + BAR), (0, 0, 0))
            img.paste(Image.fromarray(renderer.render()), (0, 0))
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, W, 30], fill=(0, 0, 0))
            draw.text((8, 4), f"safety {mode.upper()}", font=font, fill=(255, 255, 255))
            draw.text((W - 150, 7), f"solved {solves}", font=small, fill=(255, 255, 255))
            if flash > 0:
                draw.rectangle([0, 0, W - 1, H - 1], outline=(40, 220, 60), width=6)
                flash -= 1
            mm = 1000 * max(nearest.distance, 0.0)
            color = (225, 60, 50) if collided else ((240, 160, 40) if nearest.distance < margin else (60, 190, 90))
            x0, x1 = 150, W - 12
            draw.text((8, H + 13), f"gap {mm:4.1f} mm", font=small, fill=color)
            draw.rectangle([x0, H + 14, x1, H + 32], outline=(90, 90, 90))
            draw.rectangle([x0, H + 14, x0 + int((x1 - x0) * min(mm, 15.0) / 15.0), H + 32], fill=color)
            tick = x0 + int((x1 - x0) * (1000 * margin) / 15.0)
            draw.line([tick, H + 10, tick, H + 36], fill=(255, 255, 255), width=2)
            frames.append(np.asarray(img))
            if terminated or truncated:
                break
        frames.extend([frames[-1]] * (args.steps - (t + 1) + 15))   # keep the two panels aligned
        print(f"{mode}: seed {seed} solves {solves} dropped {info['dropped']} steps in contact {contact_steps}/{t + 1}")
    renderer.close()
    env.close()
    return frames


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--goal-angle", type=float, default=45.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--fps", type=int, default=50, help="50 = half speed")
    p.add_argument("--urdf", default=None)
    p.add_argument("--safety-config", default=None)
    args = p.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model = PPO.load(args.model, device="cpu")
    kwargs = dict(max_episode_steps=args.steps, hold_steps=10)
    kwargs.update(policy_env_kwargs(model, args.model))
    normalizer = VecNormalize.load(str(resolve_stats_path(Path(args.model))),
                                   DummyVecEnv([lambda: CubeReorientContinuous(**kwargs)]))
    normalizer.training = False
    normalizer.norm_reward = False
    urdf = find_urdf(args.urdf)

    left = rollout("off", args, model, normalizer, kwargs, urdf)
    right = rollout("on", args, model, normalizer, kwargs, urdf)
    gap = np.full((H + BAR, 6, 3), 255, np.uint8)
    video = [np.concatenate([a, gap, b], axis=1) for a, b in zip(left, right)]
    imageio.mimwrite(args.out, video, fps=args.fps, quality=7, macro_block_size=2)
    print(args.out, len(video), "frames")


if __name__ == "__main__":
    main()
