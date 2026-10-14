"""Text-free demo video of a cube-reorientation policy, 1280x720.

    python -m orca_rl.record_clean --model runs/run23/final_model.zip --out cube_v1.mp4 --goal-angle 45

Only the goal arrows (green: where the red face should point, red: where it
points now) and a green border flash on each solve. Scans --scan reset seeds
and keeps the --episodes with the most solves: this picks footage for a demo,
it is not an evaluation -- use orca_rl.evaluate for numbers.
"""
import argparse
from types import SimpleNamespace

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from orca_rl.record import build_actor, make_env_for, make_camera, _add_arrow, GOAL_RGBA, FACE_RGBA, GAUGE_OFFSET

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--out", required=True)
p.add_argument("--goal-angle", type=float, default=45)
p.add_argument("--episodes", type=int, default=3)
p.add_argument("--scan", type=int, default=12)
p.add_argument("--steps", type=int, default=400)
args = p.parse_args()

act, obs_kw = build_actor(SimpleNamespace(policy="model", model=args.model, vecnormalize=None, action_scale=None))
env = make_env_for(obs_kw, args.steps, args.goal_angle)

scores = []
for seed in range(args.scan):
    obs, info = env.reset(seed=seed)
    env.goal_angle_deg = args.goal_angle
    solves = 0
    for t in range(args.steps):
        obs, _, te, tr, info = env.step(act(env, obs))
        solves += int(info["solved_this_step"])
        if te or tr:
            break
    scores.append((solves, not info["dropped"], seed))
    print("seed", seed, "solves", solves, "dropped", info["dropped"])
chosen = [s for _, _, s in sorted(scores, reverse=True)[: args.episodes]]
print("chosen", chosen)

W, H = 1280, 720
env.model.vis.global_.offwidth, env.model.vis.global_.offheight = W, H
renderer = mujoco.Renderer(env.model, height=H, width=W)
frames = []
for seed in chosen:
    obs, info = env.reset(seed=seed)
    env.goal_angle_deg = args.goal_angle
    cam = make_camera(env)
    cam.distance = 0.30
    flash = 0
    for t in range(args.steps):
        obs, _, te, tr, info = env.step(act(env, obs))
        if info["solved_this_step"]:
            flash = 20
        cam.lookat[:] = env.data.xpos[env._cube_body_id]
        renderer.update_scene(env.data, camera=cam)
        gauge = env.data.xpos[env._cube_body_id] + GAUGE_OFFSET
        _add_arrow(renderer.scene, gauge, env._goal_dir, GOAL_RGBA)
        _add_arrow(renderer.scene, gauge, env._cube_red_face_world_normal(), FACE_RGBA)
        px = renderer.render()
        if flash > 0:
            img = Image.fromarray(px)
            ImageDraw.Draw(img).rectangle([0, 0, W - 1, H - 1], outline=(40, 220, 60), width=10)
            px = np.asarray(img)
            flash -= 1
        frames.append(px)
        if te or tr:
            break
    frames.extend([frames[-1]] * 15)
renderer.close()
imageio.mimwrite(args.out, frames, fps=50, quality=8, macro_block_size=1)
print(args.out, len(frames), "frames")
