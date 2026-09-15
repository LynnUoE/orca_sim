"""Render the cube task offscreen: a still PNG and an optional MP4 of a random policy.

The scene ships with no cameras defined, so we drive MuJoCo's free camera
manually and point it at the cube.

    python preview.py                 # writes preview.png
    python preview.py --video         # also writes preview.mp4
"""

import argparse

import imageio.v2 as imageio
import mujoco

from orca_sim import OrcaHandRightCubeOrientation

WIDTH, HEIGHT = 640, 480


def make_camera(env):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.model, cam)
    cam.azimuth = 140.0
    cam.elevation = -25.0
    cam.distance = 0.45
    cam.lookat[:] = env.data.xpos[env._cube_body_id]
    return cam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="store_true", help="also record an MP4")
    parser.add_argument("--steps", type=int, default=200, help="frames to record")
    args = parser.parse_args()

    env = OrcaHandRightCubeOrientation(version="v2")
    env.reset(seed=0)

    renderer = mujoco.Renderer(env.model, height=HEIGHT, width=WIDTH)
    cam = make_camera(env)

    renderer.update_scene(env.data, camera=cam)
    imageio.imwrite("preview.png", renderer.render())
    print("wrote preview.png")

    if args.video:
        frames = []
        for step in range(args.steps):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            cam.lookat[:] = env.data.xpos[env._cube_body_id]
            renderer.update_scene(env.data, camera=cam)
            frames.append(renderer.render())
            if terminated or truncated:
                env.reset(seed=step)
        imageio.mimsave("preview.mp4", frames, fps=30)
        print(f"wrote preview.mp4 ({len(frames)} frames)")

    renderer.close()
    env.close()


if __name__ == "__main__":
    main()