"""SO-101 with absolute Cartesian targets ``(x, y, z, gripper)``, so the LLM agent gets ``move_to``.

Wraps inspect-robots-so101's joint-space ``SOArmEmbodiment``. Each target is solved
to joint angles by weighted least squares over the URDF chain: exact position, gripper
as close to straight down as the wrist limit allows, wrist roll held. Each joint moves
at most ``MAX_JOINT_STEP_DEG`` per step, then the command goes through the inner
embodiment, which keeps its joint clamp, pacing, settling and operator episode end.
``reset()`` refuses a start pose that isn't gripper-down, so set the inner
``home_pose`` to one.

The CLI's ``-E`` passes scalars only. Tuples (workspace, offsets, ``home_pose``) and
LeRobot camera configs go in a Python factory returning ``SOArmCartesian(...)``;
point the ``so_arm_cartesian`` entry point at that factory for hardware runs.
"""

import dataclasses
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from ikpy.chain import Chain
from scipy.optimize import least_squares
from inspect_robots.approver import DeltaLimitApprover, GuardrailContribution
from inspect_robots.errors import EmbodimentFault
from inspect_robots.scene import Scene
from inspect_robots.spaces import ActionSemantics, Box, StateField, StateSpec
from inspect_robots.types import Action, Observation, StepResult
from inspect_robots_so101 import STATE_KEY as JOINT_KEY
from inspect_robots_so101 import SOArmEmbodiment

URDF = Path(__file__).with_name("so101_new_calib.urdf")  # TheRobotStudio/SO-ARM100, Apache-2.0
STATE_KEY = "eef_pos"
DOWN = np.array([0.0, 0.0, -1.0])
# Chain links: base, pan, lift, elbow, wrist_flex, wrist_roll, gripper_frame. Roll is held.
ACTIVE = [False, True, True, True, True, False, False]
SOLVED = np.flatnonzero(ACTIVE)
# Wrist flex tops out at 95 deg, so straight-down only fits below z ~0.08 m. Above
# that the gripper tilts; 0.01 keeps the position within ~0.5 mm of exact.
TILT_WEIGHT = 0.01
IK_TOL_M = 0.005  # ponytail: targets the IK misses by more than this are held, not chased
MAX_STEP = (0.004, 0.004, 0.004, 4.0)  # per control step: 12 cm/s at 30 Hz, gripper 4 units
# Joint-space speed limit: the step is scaled uniformly (a straight joint-space line)
# so no joint moves more than this. A 4 mm step needs up to ~18 deg near the far
# +-y corners at table height; those waypoints just take two steps.
MAX_JOINT_STEP_DEG = 10.0

DOCS = """\
Cartesian control of one SO-101 arm. Frame: robot base, meters. +x points forward
away from the base, +y to the robot's left, +z up; z=0 is roughly the table
surface. x, y, z place the point between the fingertips. The gripper points
straight down up to about z=0.08 and tilts outward above that. gripper:
0 = closed, 100 = fully open. `eef_pos` reports the commanded pose and gripper,
`joint_pos` the measured joints and gripper, so a closed command whose measured
gripper stops well above 0 means an object is held. A target the arm cannot reach
is not executed and `eef_pos` stays put, so check `eef_pos` after each move."""


class HoldRewind:
    """Approver applying an embodiment's holds before its step, as inspect-robots asks of
    a guardrail that substitutes a target: the CLI's delta limiter measures each action
    from the last approved one, so a hold it never saw would drag the next move toward
    the refused target."""

    def __init__(self, ran: Callable[[np.ndarray], np.ndarray]) -> None:
        self._ran = ran

    def review(self, action: Action, store: dict[str, Any]) -> Action:
        """Return the action, or the held target that will run instead."""
        ran = self._ran(np.asarray(action.data, dtype=np.float64))
        if np.allclose(ran, action.data, rtol=0, atol=1e-9):
            return action
        DeltaLimitApprover.rewind_reference(store, ran)
        return dataclasses.replace(action, data=ran)


class SOArmCartesian:
    """``so_arm`` behind a 4-D ``eef_abs_pose`` action space."""

    def __init__(
        self,
        inner: SOArmEmbodiment | None = None,
        *,
        # Fully reachable per the self-check. z_low is the table guard: raise it if
        # your table sits above the base's z=0.
        workspace_low: tuple[float, float, float] = (0.16, -0.20, 0.01),
        workspace_high: tuple[float, float, float] = (0.30, 0.20, 0.20),
        joint_offset_deg: tuple[float, ...] = (0.0,) * 5,  # motor reading minus URDF angle
        **flat: Any,  # SOArmConfig fields, e.g. the CLI's -E port=...
    ) -> None:
        if inner is not None and flat:
            raise TypeError(f"give {sorted(flat)} to the inner SOArmEmbodiment instead")
        self._inner = inner if inner is not None else SOArmEmbodiment(**flat)
        self._cfg = cfg = self._inner._cfg  # ponytail: private, but the dependency is pinned
        if not cfg.use_degrees or cfg.joints_are_delta:
            raise ValueError("SOArmCartesian needs use_degrees=True and joints_are_delta=False")
        obs_space = self._inner.info.observation_space
        with warnings.catch_warnings():  # the URDF's fixed joints carry an ignored axis
            warnings.simplefilter("ignore", UserWarning)
            self._chain = Chain.from_urdf_file(
                str(URDF), base_elements=["base_link"], active_links_mask=ACTIVE
            )
        self._offset = np.asarray(joint_offset_deg, dtype=np.float64)
        # Solve inside both the URDF limits and the inner joint clamp, so the guards below
        # judge the joints that actually run.
        urdf = np.array([self._chain.links[i].bounds for i in SOLVED]).T
        clamp = np.radians(np.array([cfg.low, cfg.high])[:, :4] - self._offset[:4])
        self._bounds = np.clip(urdf, clamp[0], clamp[1])
        if np.any(self._bounds[0] >= self._bounds[1]):
            raise ValueError("calibrated joint limits minus joint_offset_deg miss the URDF's")
        self._low = np.asarray(workspace_low, dtype=np.float64)
        self._high = np.asarray(workspace_high, dtype=np.float64)
        self.cmd = np.zeros(6)  # last commanded joints + gripper; holding these doesn't sag
        self.info = dataclasses.replace(
            self._inner.info,
            name="so_arm_cartesian",
            action_space=Box(
                shape=(4,),
                low=np.append(self._low, 0.0),
                high=np.append(self._high, 100.0),
                semantics=ActionSemantics(
                    control_mode="eef_abs_pose",
                    gripper="continuous",
                    dim_labels=("x", "y", "z", "gripper"),
                    max_step=MAX_STEP,
                ),
            ),
            observation_space=dataclasses.replace(
                obs_space,
                state_keys=obs_space.state_keys | {STATE_KEY},
                state=StateSpec(
                    fields=(*obs_space.state.fields, StateField(STATE_KEY, (4,), "m+normalized"))
                ),
            ),
            docs=DOCS,
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Reset the inner arm; refuse to start from a pose the IK would snap away from."""
        obs = self._inner.reset(scene, seed=seed)
        measured = np.asarray(obs.state[JOINT_KEY], dtype=np.float64)
        # Clipped like the inner clamp will clip it, so we judge the pose that runs.
        self.cmd = np.clip(measured, self._cfg.low, self._cfg.high)
        # preferred is inside the clamp, so this also bounds the clamp's snap (step 1 can
        # add one capped step on top).
        preferred = self.ik(self.fk(self.cmd)[:3, 3], self.cmd)[0]
        if np.max(np.abs(preferred - measured[:5])) > MAX_JOINT_STEP_DEG:
            raise EmbodimentFault(
                f"start pose {measured[:5].round(1)} deg is not gripper-down or is outside the "
                f"joint limits; set SOArmConfig.home_pose to a gripper-down pose, e.g. "
                f"{preferred.round(1)} plus a gripper value"
            )
        return self._augment(obs)

    def step(self, action: Action) -> StepResult:
        """Plan the target and send it."""
        return self.execute(*self.plan(action.data), meta=action.meta)

    def plan(self, data: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Return the next 6-D command toward ``data`` and its IK info, without sending it.

        The command is the IK solution for the workspace-clipped target, joint- and
        gripper-capped. An IK miss, or a step ending below the table guard (or, if the
        arm already sits under it, lower than it is now), holds the arm; the gripper
        still follows.
        """
        target = np.asarray(data, dtype=np.float64).reshape(4)
        if not np.all(np.isfinite(target)):  # the rollout rejects these; guard direct callers
            raise ValueError(f"non-finite target {target}")
        here = self.fk(self.cmd)[:3, 3]
        xyz = np.clip(target[:3], self._low, self._high)
        # Already there (an unnamed arm, a gripper-only move): re-solving would shift it by
        # the IK's tilt trade-off, a little more with every move. ponytail: a held seed stuck
        # in a joint-limit corner stays stuck; retry from a gripper-down seed if that shows up.
        there = np.allclose(xyz, np.clip(here, self._low, self._high), rtol=0, atol=1e-6)
        joints, err = (self.cmd[:5], 0.0) if there else self.ik(xyz, self.cmd)
        delta = joints - self.cmd[:5]
        arm = self.cmd[:5] + delta / max(1.0, np.abs(delta).max() / MAX_JOINT_STEP_DEG)
        floor = min(self._low[2] - IK_TOL_M, here[2])
        held = err > IK_TOL_M or self.fk(arm)[2, 3] < floor
        # Capped like the joints, so a target that ran ahead during a hold can't snap the jaw.
        grip = self.cmd[5] + np.clip(target[3] - self.cmd[5], -MAX_STEP[3], MAX_STEP[3])
        cmd = np.append(self.cmd[:5] if held else arm, grip)
        return cmd, {"ik_error_m": err, "ik_held": held}

    def execute(
        self, cmd: np.ndarray, info: dict[str, Any], *, meta: Mapping[str, Any]
    ) -> StepResult:
        """Send a command from ``plan`` (or ``self.cmd`` with a new gripper, to hold)."""
        self.cmd = cmd
        result = self._inner.step(Action(data=cmd, meta=meta))
        return dataclasses.replace(
            result, observation=self._augment(result.observation), info={**result.info, **info}
        )

    def ran(self, data: np.ndarray) -> np.ndarray:
        """The target a step toward ``data`` actually runs: a held arm stays where it is."""
        cmd, _ = self.plan(data)
        held = np.array_equal(cmd[:5], self.cmd[:5])
        return np.append(self.fk(self.cmd)[:3, 3] if held else data[:3], cmd[5])

    def contribute_guardrails(self, action_space: Box) -> GuardrailContribution:
        """CLI hook: apply holds in the approver chain (see ``HoldRewind``)."""
        return GuardrailContribution(approvers=(("holds", HoldRewind(self.ran)),))

    def close(self) -> None:
        """Release the inner arm."""
        self._inner.close()

    def ik(self, xyz: np.ndarray, seed_deg: np.ndarray) -> tuple[np.ndarray, float]:
        """Return arm joints (deg) reaching ``xyz``, gripper as close to down as possible, and the
        position miss (m)."""
        q = self._q(seed_deg)
        low, high = self._bounds

        def residual(solved: np.ndarray) -> np.ndarray:
            q[SOLVED] = solved
            pose = self._chain.forward_kinematics(q)
            return np.concatenate([pose[:3, 3] - xyz, TILT_WEIGHT * (pose[:3, 2] - DOWN)])

        q[SOLVED] = least_squares(residual, np.clip(q[SOLVED], low, high), bounds=(low, high)).x
        err = float(np.linalg.norm(self._chain.forward_kinematics(q)[:3, 3] - xyz))
        return np.degrees(q[1:6]) + self._offset, err

    def fk(self, joints_deg: np.ndarray) -> np.ndarray:
        """Return the 4x4 grasp-point pose for five arm joints (deg)."""
        return self._chain.forward_kinematics(self._q(joints_deg))

    def urdf_angles(self, joints_deg: np.ndarray) -> np.ndarray:
        """Return the five arm joints as URDF (and MJCF) angles (rad)."""
        return np.radians(joints_deg[:5] - self._offset)

    def _q(self, joints_deg: np.ndarray) -> np.ndarray:
        q = np.zeros(len(self._chain.links))
        q[1:6] = self.urdf_angles(joints_deg)
        return q

    def _augment(self, obs: Observation) -> Observation:
        # Commanded, not measured: move_to fills unnamed dims from here, and measured values
        # would let a sagging arm sink and a stalled-open gripper loosen with every move.
        eef = np.append(self.fk(self.cmd)[:3, 3], self.cmd[5])
        return dataclasses.replace(obs, state={**obs.state, STATE_KEY: eef})
