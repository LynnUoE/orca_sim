"""Continuous in-hand cube reorientation for the ORCA hand.

This fixes three problems with ``orca_sim``'s stock ``OrcaHandRightCubeOrientation``:

1. **The reward paid the policy to stall.**  The stock reward is
   ``0.5 * (alignment + 1)`` every step *and* the episode terminates on success,
   so hovering just outside the tolerance for 200 steps scores ~196 while
   actually solving in 20 steps scores ~10.  Here the shaping term is
   potential-based (it sums to the total *improvement*, not to elapsed time)
   and success does not end the episode -- it awards a bonus and hands the
   policy a new goal.  Doing the task is now strictly better than not.

2. **The success test did not require holding the cube.**  A random policy hits
   the stock criterion 21% of the time, mostly by flinging the cube and
   catching a lucky frame mid-tumble.  Success here requires the cube to be
   in contact with the hand, above ``in_hand_height``, and aligned for
   ``hold_steps`` consecutive steps.

3. **The action space was raw joint angles with per-joint ranges.**  Actions
   here are always ``[-1, 1]^17``; the env maps them onto the actuator ranges,
   so no ``RescaleAction`` wrapper is needed.  ``action_mode="relative"``
   (the default) treats the action as a delta on the previous target, which
   trains better and produces much smoother trajectories -- which matters if
   these are ever going to run on real tendon-driven hardware.

Optional physics randomization (off by default) is included as the natural
next step toward sim-to-real.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from orca_sim.task_envs import OrcaHandRightCubeOrientation

# Six axis-aligned target directions for the red face, in world coordinates.
AXIS_GOALS = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


class CubeReorientContinuous(OrcaHandRightCubeOrientation):
    """In-hand reorientation that keeps going after each solve.

    Observation (54,):
        hand qpos (17) | cube pos (3) | cube quat (4)
        hand qvel (17) | cube linvel (3) | cube angvel (3)
        red-face world normal (3) | goal direction (3) | alignment (1)

    Action (17,), always in [-1, 1].
    """

    def __init__(
        self,
        render_mode: str | None = None,
        version: str | None = "v2",
        *,
        # --- episode ---
        max_episode_steps: int = 400,
        # --- success definition ---
        success_tolerance_rad: float = np.deg2rad(15.0),
        hold_steps: int = 10,
        in_hand_height: float = 0.15,
        require_contact: bool = True,
        drop_height: float = 0.10,
        # --- goals ---
        goal_mode: str = "curriculum",  # "curriculum" | "axis" | "random"
        resample_goal_on_success: bool = True,
        min_goal_angle_rad: float = np.deg2rad(60.0),
        # --- curriculum (goal_mode="curriculum") ---
        curriculum_start_deg: float = 30.0,
        curriculum_min_deg: float = 25.0,
        curriculum_max_deg: float = 180.0,
        curriculum_step_deg: float = 5.0,
        curriculum_window: int = 20,
        curriculum_up_rate: float = 1.0,
        curriculum_down_rate: float = 0.25,
        # --- actions ---
        action_mode: str = "relative",  # "relative" | "absolute"
        action_scale: float = 0.15,
        # --- reward weights ---
        success_bonus: float = 10.0,
        shaping_coef: float = 1.0,
        drop_penalty: float = 5.0,
        action_rate_penalty: float = 0.002,
        align_bonus: float = 0.0,
        spin_penalty: float = 0.0,
        # --- domain randomization ---
        randomize_physics: bool = False,
        randomization: dict[str, float] | None = None,
        # --- reset ---
        initial_red_face: str = "random",
        cube_pos_xy_jitter: float | tuple[float, float] = 0.005,
        **kwargs: Any,
    ) -> None:
        if goal_mode not in {"axis", "random", "curriculum"}:
            raise ValueError("goal_mode must be 'curriculum', 'axis' or 'random'")
        if action_mode not in {"relative", "absolute"}:
            raise ValueError("action_mode must be 'relative' or 'absolute'")

        # Needed before super().__init__ because it calls _get_obs().
        self._goal_dir = AXIS_GOALS[0].copy()
        self._obs_noise_rad = 0.0

        self.hold_steps = int(hold_steps)
        self.in_hand_height = float(in_hand_height)
        self.require_contact = bool(require_contact)
        self.goal_mode = goal_mode
        self.resample_goal_on_success = bool(resample_goal_on_success)
        self.min_goal_angle_rad = float(min_goal_angle_rad)
        self.curriculum_min_deg = float(curriculum_min_deg)
        self.curriculum_max_deg = float(curriculum_max_deg)
        self.curriculum_step_deg = float(curriculum_step_deg)
        self.curriculum_window = int(curriculum_window)
        self.curriculum_up_rate = float(curriculum_up_rate)
        self.curriculum_down_rate = float(curriculum_down_rate)
        self._solve_rates: deque[float] = deque(maxlen=self.curriculum_window)
        self.goal_angle_deg = float(curriculum_start_deg)
        self.action_mode = action_mode
        self.action_scale = float(action_scale)
        self.success_bonus = float(success_bonus)
        self.shaping_coef = float(shaping_coef)
        self.drop_penalty = float(drop_penalty)
        self.action_rate_penalty = float(action_rate_penalty)
        self.align_bonus = float(align_bonus)
        self.spin_penalty = float(spin_penalty)
        self.randomize_physics = bool(randomize_physics)
        self.randomization = {
            "cube_mass": 0.30,        # +-30% relative
            "cube_friction": 0.30,
            "joint_damping": 0.30,
            "actuator_gain": 0.20,
            "obs_noise_rad": 0.005,   # ~0.3 deg of encoder noise
            **(randomization or {}),
        }

        super().__init__(
            render_mode=render_mode,
            version=version,
            initial_red_face=initial_red_face,
            cube_pos_xy_jitter=cube_pos_xy_jitter,
            max_episode_steps=max_episode_steps,
            success_tolerance_rad=success_tolerance_rad,
            drop_height=drop_height,
            **kwargs,
        )

        # Normalized action space; the env handles the mapping to ctrlrange.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )
        self._ctrl_center = 0.5 * (self.action_high + self.action_low)
        self._ctrl_halfspan = 0.5 * (self.action_high - self.action_low)

        self._cube_geom_ids = {
            self.model.geom("task_cube_body").id,
            self.model.geom("task_cube_red_face").id,
        }

        # Nominal physics, kept so randomization always perturbs around the same base.
        self._nominal = {
            "body_mass": self.model.body_mass.copy(),
            "geom_friction": self.model.geom_friction.copy(),
            "dof_damping": self.model.dof_damping.copy(),
            "actuator_gainprm": self.model.actuator_gainprm.copy(),
            "actuator_biasprm": self.model.actuator_biasprm.copy(),
        }

        self._prev_target = np.zeros(self.model.nu, dtype=np.float64)
        self._prev_alignment = 0.0
        self._hold_counter = 0
        self._successes = 0

        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float64
        )

    # ------------------------------------------------------------------ goals

    def _draw_goal(self) -> np.ndarray:
        if self.goal_mode == "curriculum":
            return self._draw_curriculum_goal()
        if self.goal_mode == "axis":
            return AXIS_GOALS[self.np_random.integers(len(AXIS_GOALS))].copy()
        vec = self.np_random.normal(size=3)
        return vec / (np.linalg.norm(vec) + 1e-12)

    def _draw_curriculum_goal(self) -> np.ndarray:
        """A goal a fixed angular distance from where the red face points now.

        Axis goals are always a 90-180 degree flip, which a fresh policy never
        stumbles into -- 20M steps produced zero solves. Starting at ~30 degrees
        gives the policy something it can reach by nudging, and the angle widens
        on its own once it is solving reliably.
        """
        normal = self._cube_red_face_world_normal()
        lo = max(1.5 * self.success_tolerance_rad, 0.5 * np.deg2rad(self.goal_angle_deg))
        hi = max(lo + 1e-6, np.deg2rad(self.goal_angle_deg))
        theta = float(self.np_random.uniform(lo, hi))

        # A random axis perpendicular to the current normal.
        axis = self.np_random.normal(size=3)
        axis -= np.dot(axis, normal) * normal
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            axis = np.array([1.0, 0.0, 0.0]) - normal[0] * normal
            norm = np.linalg.norm(axis)
        axis /= norm

        # Rodrigues rotation of `normal` about `axis` by `theta`.
        goal = (
            normal * np.cos(theta)
            + np.cross(axis, normal) * np.sin(theta)
            + axis * np.dot(axis, normal) * (1.0 - np.cos(theta))
        )
        return goal / (np.linalg.norm(goal) + 1e-12)

    def _update_curriculum(self) -> None:
        """Widen the goal angle when the policy is solving, narrow it when not.

        Two things this deliberately does NOT do, both learned the hard way on
        a 20M-step run whose difficulty never moved off the floor:

        * It does not compare raw solve *counts*. An episode that ends in a drop
          after 90 steps cannot physically fit two solves, so counting them
          punished the policy for a short episode and ratcheted difficulty down
          every time the drop rate rose. Rates are normalized to a full episode.

        * It does not react to a single episode. Solves per episode are wildly
          overdispersed here (mean 0.84, max 4, ~half of episodes zero), so a
          per-episode rule random-walks: one zero drops 5 degrees, one good
          episode adds them back. Averaged over a window it moves only on
          evidence, and the window is cleared after a change so the next
          decision is measured at the new difficulty.
        """
        if self.goal_mode != "curriculum":
            return

        steps = max(int(self._elapsed_steps), 1)
        rate = self._successes * (self.max_episode_steps / steps)
        self._solve_rates.append(float(rate))

        if len(self._solve_rates) < self.curriculum_window:
            return

        mean_rate = float(np.mean(self._solve_rates))
        if mean_rate >= self.curriculum_up_rate:
            self.goal_angle_deg = min(
                self.goal_angle_deg + self.curriculum_step_deg, self.curriculum_max_deg
            )
            self._solve_rates.clear()
        elif mean_rate <= self.curriculum_down_rate:
            self.goal_angle_deg = max(
                self.goal_angle_deg - self.curriculum_step_deg, self.curriculum_min_deg
            )
            self._solve_rates.clear()

    @property
    def curriculum_solve_rate(self) -> float:
        """Windowed solve rate the curriculum is currently acting on."""
        return float(np.mean(self._solve_rates)) if self._solve_rates else 0.0

    def _sample_goal(self) -> np.ndarray:
        """Draw a goal the cube does not already satisfy.

        Without this, ~1 in 6 axis goals is already met at reset (and the goal
        drawn right after a success is often the one just solved), handing the
        policy free success bonuses it did nothing to earn.
        """
        if self.goal_mode == "curriculum":
            return self._draw_curriculum_goal()
        normal = self._cube_red_face_world_normal()
        threshold = np.cos(self.min_goal_angle_rad)
        for _ in range(32):
            candidate = self._draw_goal()
            if float(np.dot(normal, candidate)) < threshold:
                return candidate
        # Degenerate fall-back: take the direction furthest from the current one.
        candidates = AXIS_GOALS if self.goal_mode == "axis" else np.stack(
            [self._draw_goal() for _ in range(32)]
        )
        return candidates[int(np.argmin(candidates @ normal))].copy()

    def _alignment(self) -> float:
        return float(np.dot(self._cube_red_face_world_normal(), self._goal_dir))

    # ------------------------------------------------------------- in-hand-ness

    def _cube_touching_hand(self) -> bool:
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            cube_side = g1 in self._cube_geom_ids, g2 in self._cube_geom_ids
            if not any(cube_side):
                continue
            other = g2 if cube_side[0] else g1
            # body 0 is the world (floor); anything else here is the hand.
            if int(self.model.geom_bodyid[other]) != 0:
                return True
        return False

    def _cube_in_hand(self) -> bool:
        if float(self.data.xpos[self._cube_body_id, 2]) <= self.in_hand_height:
            return False
        if self.require_contact and not self._cube_touching_hand():
            return False
        return True

    def _aligned(self) -> bool:
        return self._alignment() >= np.cos(self.success_tolerance_rad)

    # ------------------------------------------------------------------ spaces

    def _get_obs(self) -> np.ndarray:
        # BaseOrcaHandEnv._get_obs: qpos + qvel. Skip the parent's own additions.
        base = np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])
        if not hasattr(self, "_cube_qpos_adr"):
            return base
        if self._obs_noise_rad > 0.0:
            base = base.copy()
            n_hand = self._cube_qpos_adr
            base[:n_hand] += self.np_random.normal(
                scale=self._obs_noise_rad, size=n_hand
            )
        return np.concatenate(
            [
                base,
                self._cube_red_face_world_normal(),
                self._goal_dir,
                np.array([self._alignment()], dtype=np.float64),
            ]
        )

    def _get_info(self) -> dict[str, Any]:
        return {
            "goal_dir": self._goal_dir.copy(),
            "alignment": self._alignment(),
            "angle_to_goal_deg": float(
                np.degrees(np.arccos(np.clip(self._alignment(), -1.0, 1.0)))
            ),
            "cube_pos": self._cube_pos(),
            "cube_qvel": self._cube_qvel(),
            "cube_height": float(self.data.xpos[self._cube_body_id, 2]),
            "in_hand": self._cube_in_hand(),
            "aligned": self._aligned(),
            "hold_counter": self._hold_counter,
            "successes": self._successes,
            "dropped": self._cube_dropped(),
            "elapsed_steps": self._elapsed_steps,
            "goal_angle_deg": self.goal_angle_deg,
            "curriculum_solve_rate": self.curriculum_solve_rate,
        }

    # ----------------------------------------------------------- randomization

    def _apply_randomization(self) -> None:
        self.model.body_mass[:] = self._nominal["body_mass"]
        self.model.geom_friction[:] = self._nominal["geom_friction"]
        self.model.dof_damping[:] = self._nominal["dof_damping"]
        self.model.actuator_gainprm[:] = self._nominal["actuator_gainprm"]
        self.model.actuator_biasprm[:] = self._nominal["actuator_biasprm"]
        self._obs_noise_rad = 0.0

        if not self.randomize_physics:
            return

        rng = self.np_random
        rel = self.randomization

        def jitter(scale: float, size: int | tuple[int, ...] = ()) -> np.ndarray:
            return np.asarray(rng.uniform(1.0 - scale, 1.0 + scale, size=size))

        self.model.body_mass[self._cube_body_id] *= float(jitter(rel["cube_mass"]))
        for geom_id in self._cube_geom_ids:
            self.model.geom_friction[geom_id, 0] *= float(jitter(rel["cube_friction"]))
        self.model.dof_damping[:] *= jitter(rel["joint_damping"], self.model.nv)

        # Position actuators: gainprm[:, 0] = kp, biasprm[:, 1] = -kp.
        gain = jitter(rel["actuator_gain"], self.model.nu)
        self.model.actuator_gainprm[:, 0] *= gain
        self.model.actuator_biasprm[:, 1] *= gain

        self._obs_noise_rad = float(rel["obs_noise_rad"])

    # ------------------------------------------------------------------- gym API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # Seed the RNG first: randomization and goal sampling both draw from it.
        gym.Env.reset(self, seed=seed)
        self._apply_randomization()

        obs, _ = super().reset(seed=seed, options=options)

        self._goal_dir = self._sample_goal()
        self._prev_target = np.asarray(self._compose_ctrl_from_qpos(), dtype=np.float64)
        self._prev_alignment = self._alignment()
        self._hold_counter = 0
        self._successes = 0

        return self._get_obs(), self._get_info()

    def _target_from_action(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if self.action_mode == "absolute":
            target = self._ctrl_center + action * self._ctrl_halfspan
        else:
            target = self._prev_target + self.action_scale * action * self._ctrl_halfspan
        return np.clip(target, self.action_low, self.action_high)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        target = self._target_from_action(action)

        rate = (target - self._prev_target) / np.maximum(self._ctrl_halfspan, 1e-8)
        self._prev_target = target

        self.data.ctrl[:] = target
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self._elapsed_steps += 1

        alignment = self._alignment()
        in_hand = self._cube_in_hand()
        dropped = self._cube_dropped()

        # Potential-based shaping: telescopes to total improvement, so waiting
        # around earns nothing. Only counted while the cube is actually held,
        # otherwise a mid-air tumble collects free reward.
        shaping = alignment - self._prev_alignment if in_hand else 0.0

        if self._aligned() and in_hand:
            self._hold_counter += 1
        else:
            self._hold_counter = 0

        solved = self._hold_counter >= self.hold_steps

        reward = self.shaping_coef * shaping
        reward -= self.action_rate_penalty * float(np.sum(rate**2))

        # Paid only while inside the tolerance cone. This cannot be farmed the
        # way the stock reward could: after hold_steps the solve fires and the
        # goal moves, so the most it can ever pay per goal is
        # hold_steps * align_bonus. It exists because a policy that reaches the
        # cone but slides out gets nothing at all otherwise -- the shaping term
        # has already been collected on the way in.
        if self.align_bonus and self._hold_counter > 0:
            reward += self.align_bonus

        # Braking. Measured cause of 85% of failed holds: the cube enters the
        # cone still spinning at ~4 rad/s and sails straight back out. The
        # shaping term is ~0 near the goal, so nothing pays the policy to slow
        # the cube down -- it only ever learned to turn it.
        #
        # Applied only inside the cone, and proportional to the cube's angular
        # speed. Unlike a flat in-cone bonus, freezing the fingers does not
        # collect it: an already-spinning cube keeps spinning unless the
        # fingers actively arrest it. So this pays for a skill, not stillness.
        if self.spin_penalty and self._aligned() and in_hand:
            spin = float(np.linalg.norm(self._cube_qvel()[3:]))
            reward -= self.spin_penalty * spin

        if solved:
            reward += self.success_bonus
            self._successes += 1
            self._hold_counter = 0
            if self.resample_goal_on_success:
                self._goal_dir = self._sample_goal()

        if dropped:
            reward -= self.drop_penalty

        # Recompute after any goal change so the next delta is measured
        # against the new goal rather than jumping.
        self._prev_alignment = self._alignment()

        terminated = bool(dropped)
        truncated = bool(self._elapsed_steps >= self.max_episode_steps)

        info = self._get_info()
        info["solved_this_step"] = bool(solved)
        info["shaping"] = float(shaping)
        if terminated or truncated:
            info["episode_successes"] = self._successes
            self._update_curriculum()

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, info


def make_env(**kwargs: Any) -> CubeReorientContinuous:
    """Factory used by the training and evaluation scripts."""
    return CubeReorientContinuous(**kwargs)
