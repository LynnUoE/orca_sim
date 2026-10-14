"""Run a cube-reorientation policy through the Orca_safety collision layer.

    python -m orca_rl.evaluate_safety --model runs/run23/final_model.zip --goal-angle 45 \
        --episodes 100 --modes off on --out results/safety-run23.json

The policy is the one PPO trained without any safety layer; nothing is
retrained. Every control step the env turns the action into 17 joint targets
(``q_nominal``). In mode ``on`` those go through ``CBFSafetyFilter`` together
with the measured joint angles, and the filtered ``q_safe`` is what the servos
get. ``q_safe`` also becomes the env's previous target, so the relative action
integrates from what was actually commanded rather than winding up against the
filter, and the policy sees it in its observation.

In both modes the hand's self-clearance is measured after every step with the
same collision model the filter uses (108 link pairs of the v1 right hand), so
``off`` is the passive baseline: what the unfiltered policy does to the hand.

The filter only knows about the hand. Contact with the cube is not a
constraint, and the cube is free to push fingers into each other.

Needs the Orca_safety package with its ``geometry`` and ``safety`` extras and a
checkout of orcahand_description (for the v1 URDF and collision meshes); v1
policies only, since the safety layer supports only the v1 right hand.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import mujoco
import numpy as np

from orca_rl.task import CubeReorientContinuous, policy_env_kwargs, resolve_stats_path

REPO = Path(__file__).resolve().parents[2]          # Orcahand-Safe-Assembly/
# Per-step fields copied from CBFSafetyFilter's info dict into --log (the per-pair distance map is left out: 108 entries a step).
LOG_FILTER_KEYS = ("q_current", "q_nominal", "q_safe", "min_distance_before", "min_distance_after",
                   "rejected_min_distance_after", "closest_pair", "num_active_constraints", "active_pairs",
                   "solver_status", "solver_iterations", "fallback", "error", "command_backtracked",
                   "command_step_scale", "backtracking_steps", "intervention_norm", "distance_time_ms",
                   "gradient_time_ms", "solver_time_ms", "total_safety_time_ms")
URDF_RELATIVE = "orcahand_description/v1/models/urdf/orcahand_right.urdf"


def make_safe_env(env_kwargs: dict, config_path: str | None = None, urdf: str | None = None,
                  fast_gradients: bool = True) -> "SafeCubeEnv":
    """A cube env whose joint targets always pass through the CBF-QP filter (used by train.py --safety)."""
    import logging

    if fast_gradients:
        from orca_rl import safety_fast
        safety_fast.install()
    logging.getLogger("orca_safety_v2.filter").setLevel(logging.ERROR)   # one fallback warning per 100 otherwise
    env = SafeCubeEnv(**env_kwargs)
    env.safety = SafetyLayer(env, find_urdf(urdf), filtering=True, config_path=config_path)
    return env


def find_urdf(explicit: str | None) -> Path:
    candidates = [Path(explicit)] if explicit else [REPO / URDF_RELATIVE, REPO.parent / URDF_RELATIVE]
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        f"v1 URDF not found (looked in {[str(c) for c in candidates]}). Clone "
        "https://github.com/orcahand/orcahand_description next to this repo or pass --urdf.")


class SafetyLayer:
    """The collision model, and optionally the CBF-QP filter, for one env.

    Mirrors ``orca_safety_v2.adapter.build_sim_model`` except for where the
    URDF's ``package://orcahand_description`` meshes are looked up: there it is
    fixed to the directory above Orca_safety, here it follows the URDF.
    """

    def __init__(self, env: CubeReorientContinuous, urdf: Path, filtering: bool, config_path: str | None) -> None:
        from orca_safety_v2.collision import FCLDistanceModel
        from orca_safety_v2.config import load_config
        from orca_safety_v2.filter import CBFSafetyFilter

        if env.version != "v1":
            raise ValueError("the safety layer supports the v1 right hand only")
        model = env.model
        joints = model.actuator_trnid[:, 0].astype(int)
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in joints]
        self.addresses = model.jnt_qposadr[joints].copy()
        self.config, pairs = load_config(config_path)
        package_root = urdf.parents[4]              # .../orcahand_description/v1/models/urdf/x.urdf
        self.collision = FCLDistanceModel(urdf, [package_root], names, model.qpos0[self.addresses], pairs)
        self.collision.broadphase_distance = self.config.activation_distance
        low = np.maximum(model.jnt_range[joints, 0], model.actuator_ctrlrange[:, 0])
        high = np.minimum(model.jnt_range[joints, 1], model.actuator_ctrlrange[:, 1])
        max_step = self.collision.velocity_limits * model.opt.timestep * env.frame_skip
        if self.config.max_step_rad is not None:
            max_step = np.minimum(max_step, self.config.max_step_rad)
        self.filter = CBFSafetyFilter(self.collision, low, high, max_step, self.config) if filtering else None
        self.ready = False

    def reset(self, env: CubeReorientContinuous) -> bool:
        """Arm the filter on the settled reset pose. False if that pose is already inside the margin."""
        self.ready = True
        if self.filter is None:
            return True
        try:
            self.filter.reset(env.data.qpos[self.addresses].copy())
        except ValueError:
            self.ready = False
        return self.ready

    def clearance(self, env: CubeReorientContinuous):
        distances = self.collision.distances(env.data.qpos[self.addresses])
        return min(distances, key=lambda d: d.distance), any(d.collision for d in distances)


class SafeCubeEnv(CubeReorientContinuous):
    """CubeReorientContinuous with the safety layer between the action and the servos."""

    safety: SafetyLayer | None = None
    last_filter_info: dict | None = None

    def reset(self, **kwargs):
        out = super().reset(**kwargs)
        if self.safety is not None:
            self.safety.reset(self)     # arm the filter on the settled pose
        return out

    def _target_from_action(self, action: np.ndarray) -> np.ndarray:
        nominal = super()._target_from_action(action)
        self.last_filter_info = None
        if self.safety is None or self.safety.filter is None or not self.safety.ready:
            return nominal
        safe, info = self.safety.filter.filter(self.data.qpos[self.safety.addresses].copy(), nominal)
        self.last_filter_info = info
        return np.asarray(safe, dtype=np.float64)


def run_chunk(job: dict) -> list[dict]:
    """Evaluate one mode on a list of reset seeds (one worker process)."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    torch.set_num_threads(1)
    model = PPO.load(job["model"], device="cpu")
    kwargs = dict(max_episode_steps=job["steps"], hold_steps=10)
    kwargs.update(policy_env_kwargs(model, job["model"]))
    stats = resolve_stats_path(Path(job["model"]))
    normalizer = VecNormalize.load(str(stats), DummyVecEnv([lambda: CubeReorientContinuous(**kwargs)]))
    normalizer.training = False
    normalizer.norm_reward = False

    env = SafeCubeEnv(**kwargs)
    env.goal_angle_deg = env.curriculum_min_deg = env.curriculum_max_deg = float(job["goal_angle"])
    env.safety = SafetyLayer(env, Path(job["urdf"]), filtering=job["mode"] == "on", config_path=job["config"])
    margin = env.safety.config.safe_distance

    episodes = []
    for seed in job["seeds"]:
        obs, info = env.reset(seed=int(seed))
        env.goal_angle_deg = float(job["goal_angle"])
        armed = env.safety.ready
        row = dict(seed=int(seed), armed=armed, steps=0, collision_steps=0, margin_steps=0,
                   min_clearance=float("inf"), filtered_steps=0, intervened_steps=0, fallback_steps=0,
                   backtracked_steps=0, intervention=[], filter_ms=[], violating_pairs={},
                   margin_mm=1000 * margin, fallback_reasons={}, log=[])
        for _ in range(job["steps"]):
            action, _ = model.predict(normalizer.normalize_obs(obs[None, :]), deterministic=True)
            obs, _, terminated, truncated, info = env.step(action[0])
            row["steps"] += 1
            nearest, collided = env.safety.clearance(env)
            row["min_clearance"] = min(row["min_clearance"], nearest.distance)
            row["collision_steps"] += int(collided)
            if nearest.distance < margin:
                row["margin_steps"] += 1
                row["violating_pairs"][nearest.pair_name] = row["violating_pairs"].get(nearest.pair_name, 0) + 1
            f = env.last_filter_info
            if f is not None:
                row["filtered_steps"] += 1
                row["intervention"].append(float(f["intervention_norm"] or 0.0))
                row["intervened_steps"] += int((f["intervention_norm"] or 0.0) > 1e-3)
                row["fallback_steps"] += int(f["fallback"])
                row["backtracked_steps"] += int(f["command_backtracked"])
                row["filter_ms"].append(float(f["total_safety_time_ms"]))
                if f["fallback"]:
                    reason = str(f.get("error", "unknown"))
                    row["fallback_reasons"][reason] = row["fallback_reasons"].get(reason, 0) + 1
            if job["log"]:
                step = dict(seed=int(seed), step=row["steps"] - 1, min_distance_measured=nearest.distance,
                            closest_pair_measured=nearest.pair_name, measured_collision=bool(collided),
                            q_measured=env.data.qpos[env.safety.addresses].tolist(),
                            in_hand=bool(info["in_hand"]), angle_to_goal_deg=float(info["angle_to_goal_deg"]))
                if f is not None:
                    step.update({k: f.get(k) for k in LOG_FILTER_KEYS})
                else:
                    step.update(q_nominal=np.asarray(env.data.ctrl, dtype=float).tolist())
                row["log"].append(step)
            if terminated or truncated:
                break
        row.update(goals=int(info["episode_goals"]), goal_solves=int(info["episode_goal_solves"]),
                   solves=int(info["successes"]), dropped=bool(info["dropped"]))
        episodes.append(row)
    env.close()
    return episodes


def summarize(episodes: list[dict], margin_mm: float) -> dict:
    steps = sum(e["steps"] for e in episodes)
    goals = sum(e["goals"] for e in episodes)
    intervention = np.concatenate([e["intervention"] for e in episodes]) if any(e["intervention"] for e in episodes) else np.zeros(0)
    filter_ms = np.concatenate([e["filter_ms"] for e in episodes]) if any(e["filter_ms"] for e in episodes) else np.zeros(0)
    filtered = sum(e["filtered_steps"] for e in episodes)
    pairs: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for e in episodes:
        for name, count in e["violating_pairs"].items():
            pairs[name] = pairs.get(name, 0) + count
        for name, count in e["fallback_reasons"].items():
            reasons[name] = reasons.get(name, 0) + count
    out = dict(
        episodes=len(episodes), steps=steps, margin_mm=margin_mm,
        goal_success_rate=sum(e["goal_solves"] for e in episodes) / max(goals, 1),
        solves_per_episode=float(np.mean([e["solves"] for e in episodes])),
        drop_rate=float(np.mean([e["dropped"] for e in episodes])),
        collision_step_rate=sum(e["collision_steps"] for e in episodes) / max(steps, 1),
        margin_violation_step_rate=sum(e["margin_steps"] for e in episodes) / max(steps, 1),
        episodes_with_collision=float(np.mean([e["collision_steps"] > 0 for e in episodes])),
        episodes_with_margin_violation=float(np.mean([e["margin_steps"] > 0 for e in episodes])),
        min_clearance_mm=1000 * min(e["min_clearance"] for e in episodes),
        median_episode_min_clearance_mm=1000 * float(np.median([e["min_clearance"] for e in episodes])),
        episodes_not_armed=sum(not e["armed"] for e in episodes),
        top_violating_pairs=dict(sorted(pairs.items(), key=lambda kv: -kv[1])[:6]),
    )
    if filtered:
        out.update(
            intervened_step_rate=sum(e["intervened_steps"] for e in episodes) / filtered,
            fallback_step_rate=sum(e["fallback_steps"] for e in episodes) / filtered,
            backtracked_step_rate=sum(e["backtracked_steps"] for e in episodes) / filtered,
            mean_intervention_rad=float(intervention.mean()),
            p95_intervention_rad=float(np.percentile(intervention, 95)),
            filter_ms_mean=float(filter_ms.mean()), filter_ms_p95=float(np.percentile(filter_ms, 95)),
            fallback_reasons=dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--goal-angle", type=float, default=45.0)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0, help="first reset seed; both modes use the same seeds")
    p.add_argument("--max-episode-steps", type=int, default=400)
    p.add_argument("--modes", nargs="+", default=["off", "on"], choices=["off", "on"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--urdf", default=None, help="v1 right-hand URDF (default: orcahand_description next to this repo)")
    p.add_argument("--safety-config", default=None, help="Orca_safety YAML (default: its v1_right.yaml)")
    p.add_argument("--out", default=None, help="write the summary and per-episode rows here as JSON")
    p.add_argument("--log", default=None, metavar="PREFIX",
                   help="also write one JSON line per control step to PREFIX.<mode>.jsonl: measured state, "
                        "q_nominal, q_safe, clearances, fallback reason, timings (about 3 kB per step)")
    args = p.parse_args()

    urdf = find_urdf(args.urdf)
    seeds = list(range(args.seed, args.seed + args.episodes))
    chunks = [seeds[i::args.workers] for i in range(args.workers) if seeds[i::args.workers]]
    result = {"model": args.model, "goal_angle": args.goal_angle, "seeds": [seeds[0], seeds[-1]], "modes": {}}
    for mode in args.modes:
        jobs = [dict(model=args.model, mode=mode, seeds=chunk, steps=args.max_episode_steps,
                     goal_angle=args.goal_angle, urdf=str(urdf), config=args.safety_config,
                     log=bool(args.log)) for chunk in chunks]
        started = time.time()
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            episodes = sorted((e for part in pool.map(run_chunk, jobs) for e in part), key=lambda e: e["seed"])
        summary = summarize(episodes, episodes[0]["margin_mm"])
        result["modes"][mode] = {"summary": summary, "episodes": [
            {k: v for k, v in e.items() if k not in ("intervention", "filter_ms", "log")} for e in episodes]}
        if args.log:
            log_path = Path(f"{args.log}.{mode}.jsonl")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w") as handle:
                for e in episodes:
                    for step in e["log"]:
                        handle.write(json.dumps(step) + "\n")
            print(f"wrote {log_path}")
        print(f"\n=== safety {mode.upper()} | {Path(args.model).parent.name} | {args.episodes} episodes, "
              f"goal {args.goal_angle:.0f} deg | {time.time() - started:.0f} s ===")
        print(f"  goal success rate            : {100 * summary['goal_success_rate']:.0f}%")
        print(f"  solves / episode             : {summary['solves_per_episode']:.2f}")
        print(f"  drop rate                    : {100 * summary['drop_rate']:.0f}%")
        print(f"  steps in self-collision      : {100 * summary['collision_step_rate']:.2f}%"
              f"   (episodes with any: {100 * summary['episodes_with_collision']:.0f}%)")
        print(f"  steps under the {summary['margin_mm']:g} mm margin : {100 * summary['margin_violation_step_rate']:.2f}%"
              f"   (episodes with any: {100 * summary['episodes_with_margin_violation']:.0f}%)")
        print(f"  smallest clearance           : {summary['min_clearance_mm']:.2f} mm"
              f"   (median over episodes: {summary['median_episode_min_clearance_mm']:.2f} mm)")
        if summary["top_violating_pairs"]:
            print(f"  pairs under the margin       : {summary['top_violating_pairs']}")
        if "filter_ms_mean" in summary:
            print(f"  steps the filter changed     : {100 * summary['intervened_step_rate']:.1f}%"
                  f"   (mean change {summary['mean_intervention_rad']:.3f} rad, p95 {summary['p95_intervention_rad']:.3f})")
            print(f"  fallback / backtracked steps : {100 * summary['fallback_step_rate']:.2f}% / "
                  f"{100 * summary['backtracked_step_rate']:.2f}%")
            if summary["fallback_reasons"]:
                print(f"  fallback reasons             : {summary['fallback_reasons']}")
            print(f"  filter time per step         : {summary['filter_ms_mean']:.1f} ms mean, "
                  f"{summary['filter_ms_p95']:.1f} ms p95  (control period 10 ms)")
        if summary["episodes_not_armed"]:
            print(f"  episodes where the reset pose was already inside the margin (filter not armed): "
                  f"{summary['episodes_not_armed']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
