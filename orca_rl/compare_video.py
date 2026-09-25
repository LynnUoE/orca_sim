"""Side-by-side video of several policies on exactly the same goals.

    python -m orca_rl.compare_video --goal-angle 45 --out videos/run10-15.mp4 \\
        --run runs/run10 "run10 baseline s0" --run runs/run11 "run11 reset_cube" ...

Every tile is rolled out on the same seeds at the same pinned difficulty, and
each episode is padded to full length after a drop, so frame N of every tile
is the same moment of the same episode. Each run is also written on its own at
full size next to the grid (`<out stem>_<run>.mp4`).

Green arrow: where the red face has to point. Red arrow: where it points now.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image

from orca_rl.record import HEIGHT, WIDTH, build_actor, make_env_for, rollout_frames


def render_run(run_dir: Path, label: str, seeds: list[int], max_steps: int,
               goal_angle: float) -> tuple[list[np.ndarray], dict]:
    args = SimpleNamespace(policy="model", model=str(run_dir / "final_model.zip"),
                           vecnormalize=None, action_scale=None)
    act, obs_kw = build_actor(args)
    env = make_env_for(obs_kw, max_steps, goal_angle)
    renderer = mujoco.Renderer(env.model, height=HEIGHT, width=WIDTH)
    frames, totals = rollout_frames(act, env, renderer, seeds=seeds, max_steps=max_steps,
                                    label=label, goal_angle=goal_angle, pad=True)
    renderer.close()
    env.close()
    return frames, totals


def tile(frame_sets: list[list[np.ndarray]], cols: int, tile_w: int) -> list[np.ndarray]:
    tile_h = tile_w * HEIGHT // WIDTH
    rows = -(-len(frame_sets) // cols)
    grid = []
    for i in range(len(frame_sets[0])):
        canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), (20, 20, 20))
        for k, frames in enumerate(frame_sets):
            tile_img = Image.fromarray(frames[i]).resize((tile_w, tile_h), Image.BILINEAR)
            canvas.paste(tile_img, ((k % cols) * tile_w, (k // cols) * tile_h))
        grid.append(np.asarray(canvas))
    return grid


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", nargs=2, action="append", required=True, metavar=("RUN_DIR", "LABEL"))
    p.add_argument("--goal-angle", type=float, default=45.0)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-episode-steps", type=int, default=400)
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--tile-width", type=int, default=480)
    p.add_argument("--fps", type=int, default=50, help="50 = half speed")
    p.add_argument("--out", default="videos/compare.mp4")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed + ep for ep in range(args.episodes)]

    frame_sets = []
    for run_dir, label in args.run:
        frames, totals = render_run(Path(run_dir), label, seeds, args.max_episode_steps,
                                    args.goal_angle)
        single = out.with_name(f"{out.stem}_{Path(run_dir).name}.mp4")
        imageio.mimsave(single, frames, fps=args.fps)
        print(f"{label:28s} solves {totals['solves']}  drops {totals['drops']}  -> {single}")
        frame_sets.append(frames)

    imageio.mimsave(out, tile(frame_sets, args.cols, args.tile_width), fps=args.fps)
    print(f"wrote {out}  ({len(frame_sets)} runs, {len(frame_sets[0])} frames, "
          f"goal angle {args.goal_angle:g} deg, seeds {seeds})")


if __name__ == "__main__":
    main()
