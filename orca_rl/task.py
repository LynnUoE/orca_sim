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
from pathlib import Path
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
        drop_mode: str = "terminate",  # "terminate" | "reset_cube"
        # --- goals ---
        goal_mode: str = "curriculum",  # "curriculum" | "axis" | "random"
        resample_goal_on_success: bool = True,
        min_goal_angle_rad: float = np.deg2rad(60.0),
        goal_timeout_steps: int = 150,
        # --- curriculum (goal_mode="curriculum") ---
        curriculum_start_deg: float = 30.0,
        curriculum_min_deg: float = 25.0,
        curriculum_max_deg: float = 180.0,
        curriculum_step_deg: float = 5.0,
        curriculum_metric: str = "goal_success",  # "goal_success" | "episode_rate"
        curriculum_window: int = 40,
        curriculum_up_rate: float = 0.55,
        curriculum_down_rate: float = 0.25,
        # --- actions ---
        action_mode: str = "relative",  # "relative" | "absolute"
        action_scale: float = 0.15,
        obs_include_target: bool = True,
        # --- reward weights ---
        success_bonus: float = 10.0,
        shaping_mode: str = "angle",  # "angle" | "cos" | "lookahead"
        lookahead_s: float = 0.15,
        lookahead_mix: float = 0.5,
        freeze_potential_off_hand: bool = False,
        shaping_coef: float = 1.0,
        drop_penalty: float = 5.0,
        action_rate_penalty: float = 0.002,
        # Both off by default: run9 (with them) finished below run10 and run12
        # (two seeds without them) at 30, 45 and 60 degrees. README 复盘六/七.
        align_bonus: float = 0.0,
        spin_penalty: float = 0.0,
        spin_band: float = 1.0,
        # --- domain randomization ---
        randomize_physics: bool = False,
        randomization: dict[str, float] | None = None,
        # --- reset ---
        reset_settle_steps: int = 60,
        initial_red_face: str = "random",
        cube_pos_xy_jitter: float | tuple[float, float] = 0.005,
        **kwargs: Any,
    ) -> None:
        if goal_mode not in {"axis", "random", "curriculum"}:
            raise ValueError("goal_mode must be 'curriculum', 'axis' or 'random'")
        if action_mode not in {"relative", "absolute"}:
            raise ValueError("action_mode must be 'relative' or 'absolute'")
        if drop_mode not in {"terminate", "reset_cube"}:
            raise ValueError("drop_mode must be 'terminate' or 'reset_cube'")
        if shaping_mode not in {"angle", "cos", "lookahead"}:
            raise ValueError("shaping_mode must be 'angle', 'cos' or 'lookahead'")
        if curriculum_metric not in {"goal_success", "episode_rate"}:
            raise ValueError("curriculum_metric must be 'goal_success' or 'episode_rate'")

        # Needed before super().__init__ because it calls _get_obs().
        self._goal_dir = AXIS_GOALS[0].copy()
        self._obs_noise_rad = 0.0
        self.obs_include_target = bool(obs_include_target)

        self.hold_steps = int(hold_steps)
        self.reset_settle_steps = int(reset_settle_steps)
        self.in_hand_height = float(in_hand_height)
        self.require_contact = bool(require_contact)
        self.goal_mode = goal_mode
        self.resample_goal_on_success = bool(resample_goal_on_success)
        self.min_goal_angle_rad = float(min_goal_angle_rad)
        self.goal_timeout_steps = int(goal_timeout_steps)
        self.curriculum_metric = curriculum_metric
        self.curriculum_min_deg = float(curriculum_min_deg)
        self.curriculum_max_deg = float(curriculum_max_deg)
        self.curriculum_step_deg = float(curriculum_step_deg)
        self.curriculum_window = int(curriculum_window)
        self.curriculum_up_rate = float(curriculum_up_rate)
        self.curriculum_down_rate = float(curriculum_down_rate)
        self._solve_rates: deque[float] = deque(maxlen=self.curriculum_window)
        self._goal_outcomes: deque[float] = deque(maxlen=self.curriculum_window)
        self.goal_angle_deg = float(curriculum_start_deg)
        self.action_mode = action_mode
        self.action_scale = float(action_scale)
        self.success_bonus = float(success_bonus)
        self.shaping_mode = shaping_mode
        self.lookahead_s = float(lookahead_s)
        self.lookahead_mix = float(lookahead_mix)
        self.freeze_potential_off_hand = bool(freeze_potential_off_hand)
        self.shaping_coef = float(shaping_coef)
        self.drop_penalty = float(drop_penalty)
        self.drop_mode = drop_mode
        self.action_rate_penalty = float(action_rate_penalty)
        self.align_bonus = float(align_bonus)
        self.spin_penalty = float(spin_penalty)
        self.spin_band = float(spin_band)
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
        self._prev_potential = 0.0
        self._hold_counter = 0
        self._successes = 0
        self._goal_age = 0
        self._align_budget = self.hold_steps
        self._episode_goals = 0
        self._episode_goal_solves = 0
        self._episode_drops = 0
        self._settled_state: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

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

    def _step_curriculum(self, window: deque[float]) -> None:
        """Move the difficulty one notch if the window is decisive, else wait."""
        if len(window) < self.curriculum_window:
            return
        mean = float(np.mean(window))
        if mean >= self.curriculum_up_rate:
            self.goal_angle_deg = min(
                self.goal_angle_deg + self.curriculum_step_deg, self.curriculum_max_deg
            )
            window.clear()
        elif mean <= self.curriculum_down_rate:
            self.goal_angle_deg = max(
                self.goal_angle_deg - self.curriculum_step_deg, self.curriculum_min_deg
            )
            window.clear()

    def _update_curriculum_on_goal(self, solved: bool) -> None:
        """Difficulty follows the fraction of *goal attempts* that succeed.

        The previous rule measured solves per episode and asked for >= 1.0 to
        level up. Measured on run6 that ceiling is unreachable: the policy sat
        at 0.6 solves/episode from 40M to 160M steps and the angle equilibrated
        at ~45 degrees forever. The quantity is also contaminated -- an episode
        that ends in a drop at step 90 cannot fit a solve, so difficulty was
        coupled to the drop rate.

        Per-goal success has neither problem. Each goal attempt is bounded by
        `goal_timeout_steps`, so the denominator is a fixed unit of opportunity,
        and the target band (up at >= 0.55, down at <= 0.25) is the usual
        "keep the student near its own success threshold" rule.

        Attempts still running when the episode ends are not counted: they had
        less than a full budget, which is the same contamination in a new hat.
        """
        if self.goal_mode != "curriculum" or self.curriculum_metric != "goal_success":
            return
        self._goal_outcomes.append(float(solved))
        self._step_curriculum(self._goal_outcomes)

    def _update_curriculum(self) -> None:
        """Legacy episode-rate rule, kept behind `curriculum_metric`."""
        if self.goal_mode != "curriculum" or self.curriculum_metric != "episode_rate":
            return

        steps = max(int(self._elapsed_steps), 1)
        rate = self._successes * (self.max_episode_steps / steps)
        self._solve_rates.append(float(rate))
        self._step_curriculum(self._solve_rates)

    def _retire_goal(self, solved: bool, resample: bool = True) -> None:
        """Close the books on the current goal attempt and hand out the next.

        Before this existed a goal stayed up until it was solved or the episode
        ended. Measured on run6: goals that were never solved burned a median of
        377 steps out of a 400-step episode, so a full episode contained 1.9
        goal attempts and the cube was still being actively moved (1.3 rad/s) --
        the policy was not stuck, it was failing slowly. Capping the attempt
        turns the same wall-clock into 3-5 attempts, and replaying the same
        run6 policy with a 150-step cap raised solves/episode 0.87 -> 1.10
        without any retraining.
        """
        self._episode_goals += 1
        self._episode_goal_solves += int(solved)
        self._update_curriculum_on_goal(solved)
        self._goal_age = 0
        self._align_budget = self.hold_steps
        if resample:
            self._goal_dir = self._sample_goal()

    @property
    def curriculum_solve_rate(self) -> float:
        """Windowed value the curriculum is currently acting on."""
        window = (
            self._goal_outcomes
            if self.curriculum_metric == "goal_success"
            else self._solve_rates
        )
        return float(np.mean(window)) if window else 0.0

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

    def _goal_angle_rad(self) -> float:
        return float(np.arccos(np.clip(self._alignment(), -1.0, 1.0)))

    def _potential(self) -> float:
        """Shaping potential; the step reward is its increase. Higher is closer.

        `cos` was the original choice and its gradient dies exactly where the
        task gets hard: d(cos t)/dt -> 0 as t -> 0, so a degree of progress at
        5 degrees out pays 9% of what a degree pays at 60 degrees out. Measured
        on run6 that is precisely where the policy stalls -- the median goal it
        failed came to rest 13.9 degrees away against a 15-degree tolerance,
        grazing the cone edge with almost no reward left to pull it in.

        Shaping on the angle itself pays the same per degree everywhere, and is
        normalized by pi so a full 180-degree reorientation is worth 1.0.
        """
        if self.shaping_mode == "cos":
            return self._alignment()
        if self.shaping_mode == "lookahead":
            mix = self.lookahead_mix
            return -((1.0 - mix) * self._goal_angle_rad() + mix * self._lookahead_angle_rad()) / np.pi
        return -self._goal_angle_rad() / np.pi

    def _cube_angvel_world(self) -> np.ndarray:
        # A free joint's rotational qvel is in the body frame (checked against a
        # finite difference of the red-face normal); rotate it into the world.
        rot = self.data.xmat[self._cube_body_id].reshape(3, 3)
        return rot @ self._cube_qvel()[3:]

    def _lookahead_angle_rad(self) -> float:
        """Angle to the goal where the red face will point in `lookahead_s`.

        Extrapolates the cube's current spin. This is what lets the potential
        reward braking without taxing speed: far from the goal, turning toward
        it fast brings the predicted angle *down* (paid), while near the goal a
        cube still spinning hard is predicted to sail past, so its predicted
        angle goes back *up* -- and slowing down is what recovers it.

        run9's spin_penalty charged every in-cone step for angular speed, which
        can be dodged by arriving slowly. Since this is a potential, it cannot
        change which policy is optimal -- it only moves the braking signal to
        the moment braking happens.

        Measured (run13, one seed): it did not help. 43/35/18% at 30/45/60 deg
        against 49-53/37-39/25-28% for the baseline's two seeds. README 复盘七.
        """
        normal = self._cube_red_face_world_normal()
        omega = self._cube_angvel_world()
        speed = float(np.linalg.norm(omega))
        turn = min(speed * self.lookahead_s, np.pi)
        if turn < 1e-6:
            return self._goal_angle_rad()
        axis = omega / speed
        predicted = (
            normal * np.cos(turn)
            + np.cross(axis, normal) * np.sin(turn)
            + axis * np.dot(axis, normal) * (1.0 - np.cos(turn))
        )
        predicted /= np.linalg.norm(predicted) + 1e-12
        return float(np.arccos(np.clip(np.dot(predicted, self._goal_dir), -1.0, 1.0)))

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
        parts = [
            base,
            self._cube_red_face_world_normal(),
            self._goal_dir,
            np.array([self._alignment()], dtype=np.float64),
        ]
        # The position-servo target, in the same [-1, 1] units as the action.
        #
        # In relative mode the action is a delta on this target, so it is the
        # integrator state of the controller -- and it was not observable. It
        # differs from qpos exactly when it matters: a finger pressing on the
        # cube sits short of its target, and kp * (target - qpos) *is* the grip
        # force. Measured on run6/run8, targets sat at a ctrlrange rail 77% of
        # the time with a mean 0.17-halfspan gap to the actual joint angle: the
        # policy was driving bang-bang because it could not see where its own
        # targets were. Without this the task is not Markov in the action.
        if self.obs_include_target and hasattr(self, "_ctrl_center"):
            parts.append((self._prev_target - self._ctrl_center) / self._ctrl_halfspan)
        return np.concatenate(parts)

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
            "goal_age": self._goal_age,
            "episode_goals": self._episode_goals,
            "episode_goal_solves": self._episode_goal_solves,
            "episode_drops": self._episode_drops,
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
        self._settle_cube()
        # Where drop_mode="reset_cube" puts the hand and cube back after a drop.
        self._settled_state = (
            self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy()
        )

        self._goal_dir = self._sample_goal()
        # Continue from the targets the hand actually held while the cube
        # settled. Re-deriving them from qpos here would snap every servo to
        # its cube-deflected angle on the first step -- the fingers let go and
        # the cube rolls ~10 degrees and slides ~18 mm before the policy acts.
        self._prev_target = np.asarray(self.data.ctrl, dtype=np.float64).copy()
        self._prev_potential = self._potential()
        self._hold_counter = 0
        self._successes = 0
        self._goal_age = 0
        self._align_budget = self.hold_steps
        self._episode_goals = 0
        self._episode_goal_solves = 0
        self._episode_drops = 0

        return self._get_obs(), self._get_info()

    def _replace_cube(self) -> None:
        """Put the hand and cube back to this episode's settled start, new goal.

        Used by drop_mode="reset_cube". A drop that ends the episode costs far
        more than `drop_penalty`: every remaining step of shaping and every
        solve the episode could still have produced goes with it. The long-1e8
        notes estimate that at ~-4.7 on top of the explicit -5, and run9/run10
        answered it the way you would expect -- drop rates of 1-4% while ~60%
        of goals were never even reached. Continuing after a drop makes the
        price exactly `drop_penalty`, and lets the policy see what happens
        after a regrasp instead of the episode simply ending.

        The interrupted goal attempt is not scored for the curriculum -- it did
        not get its full budget, same rule as an attempt cut off by the episode
        end.
        """
        qpos, qvel, ctrl = self._settled_state
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = ctrl
        mujoco.mj_forward(self.model, self.data)
        self._prev_target = ctrl.astype(np.float64).copy()
        self._hold_counter = 0
        self._goal_age = 0
        self._align_budget = self.hold_steps
        self._goal_dir = self._sample_goal()

    def _settle_cube(self) -> None:
        """Let the cube land before the goal is drawn from its orientation.

        The cube spawns at 0.19 m and comes to rest on the palm at ~0.174 m,
        tipping ~21 degrees on the way down and taking 35-50 control steps to
        stop. The goal used to be drawn from the pre-drop orientation, 22.5-30
        degrees away -- so a quarter of the time the drop alone carried the red
        face into the 15-degree cone. Measured: the zero policy "solved" 13 of
        50 episodes at step ~13, a free +10 that also inflated the curriculum's
        success rate at exactly the low angles run8 was stuck at.

        The hand holds its reset pose (ctrl pinned to the initial targets) while
        the cube settles; the parent's own `settle_steps` instead re-targets the
        servos to the current joint angles every substep, which lets the hand
        sag under the cube.
        """
        if self.reset_settle_steps <= 0:
            return
        self.data.ctrl[:] = self._compose_ctrl_from_qpos()
        mujoco.mj_step(self.model, self.data, nstep=self.reset_settle_steps * self.frame_skip)

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
        goal_before = self._goal_dir
        target = self._target_from_action(action)

        rate = (target - self._prev_target) / np.maximum(self._ctrl_halfspan, 1e-8)
        self._prev_target = target

        self.data.ctrl[:] = target
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self._elapsed_steps += 1
        self._goal_age += 1

        in_hand = self._cube_in_hand()
        dropped = self._cube_dropped()

        # Potential-based shaping: telescopes to total improvement, so waiting
        # around earns nothing. Only counted while the cube is actually held,
        # otherwise a mid-air tumble collects free reward.
        shaping = self._potential() - self._prev_potential if in_hand else 0.0

        if self._aligned() and in_hand:
            self._hold_counter += 1
        else:
            self._hold_counter = 0

        solved = self._hold_counter >= self.hold_steps

        reward = self.shaping_coef * shaping
        reward -= self.action_rate_penalty * float(np.sum(rate**2))

        # Paid only while inside the tolerance cone, and only `hold_steps` times
        # per goal -- the budget is what makes that true. Without it the counter
        # resets on every exit, so a policy that oscillates across the cone edge
        # collects the bonus forever without ever solving: the stock task's
        # stalling exploit, rebuilt. It exists because a policy that reaches the
        # cone but slides out gets nothing at all otherwise -- the shaping term
        # has already been collected on the way in.
        if self.align_bonus and self._hold_counter > 0 and self._align_budget > 0:
            reward += self.align_bonus
            self._align_budget -= 1

        # Braking. Measured cause of 59% of failed holds on run6: the cube
        # enters the cone still spinning (median 1.5 rad/s, p90 7.3) and sails
        # straight back out, ending the run a median of 9 steps long against a
        # 10-step bar. The shaping term is ~0 near the goal, so nothing pays the
        # policy to slow the cube down -- it only ever learned to turn it.
        #
        # Applied inside a band `spin_band` tolerances wide and proportional to
        # the cube's angular speed. Unlike a flat in-cone bonus, freezing the
        # fingers does not collect it: an already-spinning cube keeps spinning
        # unless the fingers actively arrest it. So this pays for a skill, not
        # for stillness.
        #
        # Keep the band at 1.0. At 2.0 it reaches 30 degrees out, which at the
        # 30-degree end of the curriculum covers every goal the env hands out:
        # a do-nothing policy then scores -2.6 per episode purely for the cube
        # settling on the palm, and the cheapest response to that is to clamp
        # the cube and stop moving. That is exactly the failure this repo
        # already hit once with drop_penalty > success_bonus.
        if self.spin_penalty and in_hand:
            band = self.spin_band * self.success_tolerance_rad
            if self._goal_angle_rad() <= band:
                spin = float(np.linalg.norm(self._cube_qvel()[3:]))
                reward -= self.spin_penalty * spin

        if solved:
            reward += self.success_bonus
            self._successes += 1
            self._hold_counter = 0

        timed_out = (
            not solved
            and self.goal_timeout_steps > 0
            and self._goal_age >= self.goal_timeout_steps
        )
        if solved or timed_out:
            self._retire_goal(
                solved, resample=timed_out or self.resample_goal_on_success
            )

        if dropped:
            reward -= self.drop_penalty
            self._episode_drops += 1
            if self.drop_mode == "reset_cube":
                self._replace_cube()

        # Recompute after any goal change so the next delta is measured
        # against the new goal rather than jumping.
        #
        # With freeze_potential_off_hand the potential is NOT tracked while the
        # cube is out of the hand: the step that re-establishes the grasp is
        # charged for everything that changed in between. Otherwise shaping only
        # telescopes over in-hand stretches, and whatever the potential loses
        # while the cube is loose is simply never billed -- flick it toward the
        # goal (paid), let it hop off the fingers and fall back (unbilled),
        # repeat. checks.py demonstrates that loop with a scripted policy.
        goal_changed = self._goal_dir is not goal_before
        if not self.freeze_potential_off_hand or in_hand or goal_changed:
            self._prev_potential = self._potential()

        terminated = bool(dropped) and self.drop_mode == "terminate"
        truncated = bool(self._elapsed_steps >= self.max_episode_steps)

        info = self._get_info()
        info["dropped_this_step"] = bool(dropped)
        info["solved_this_step"] = bool(solved)
        info["goal_timed_out"] = bool(timed_out)
        info["shaping"] = float(shaping)
        if terminated or truncated:
            info["episode_successes"] = self._successes
            self._update_curriculum()

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, info


LEGACY_OBS_DIM = 54  # runs 1-8: no controller target in the observation


def obs_kwargs_for_model(model: Any) -> dict[str, Any]:
    """Env kwargs that reproduce the observation a saved policy was trained on."""
    return {"obs_include_target": int(model.observation_space.shape[0]) != LEGACY_OBS_DIM}


def resolve_stats_path(model_path: Path) -> Path:
    """The VecNormalize stats that belong to a saved model.

    `final_model.zip` sits next to `vecnormalize.pkl`, but a checkpoint
    `checkpoints/ppo_<N>_steps.zip` sits next to `ppo_vecnormalize_<N>_steps.pkl`.
    Only looking for the former meant resuming from a checkpoint -- the normal
    way to recover a crashed run -- silently restarted the observation
    statistics from zero.
    """
    step_suffix = model_path.stem.removeprefix("ppo_")
    checkpoint_stats = model_path.parent / f"ppo_vecnormalize_{step_suffix}.pkl"
    if model_path.stem.startswith("ppo_") and checkpoint_stats.exists():
        return checkpoint_stats
    return model_path.parent / "vecnormalize.pkl"


def make_env(**kwargs: Any) -> CubeReorientContinuous:
    """Factory used by the training and evaluation scripts."""
    return CubeReorientContinuous(**kwargs)
