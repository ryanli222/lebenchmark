"""Two SO-101 arms as one embodiment: ``(x, y, z, gripper)`` per arm in one table frame.

Each arm is an ``SOArmCartesian``. This wrapper maps the 8-D action into each arm's
base frame, holds the arms when a step would bring them closer than ``min_gap_m``,
and merges the two observations.

The gap is measured on the MuJoCo collision model of both arms, so the bases must be
measured on the real table. Each step sends the left arm, then the right, and checks
every pose that can bring about: each arm at its command (jaw there, or stalled open
where measured) or still where measured (blocked by an object). ``settle_tolerance``
is required so an arm gets there before the other moves (on hardware ~1-2 deg, for
backlash; raise ``settle_timeout_budget`` too, or a blocked arm soon switches settling
off for the trial). ponytail: ``min_gap_m`` must also cover what isn't checked: the
path between checked poses (one capped step), a held object, the bases (no collision
shapes: keep each out of the other arm's reach), and homing.

Wiring a real rig: only the left arm's ``SOArmEmbodiment`` gets the operator prompt and
the end-key poll; give the right one ``operator=SilentOperator()`` and
``poll_end=lambda: False``. Camera names must differ between the arms.
"""

import dataclasses
import itertools
import re
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from inspect_robots.approver import GuardrailContribution
from inspect_robots.scene import Scene
from inspect_robots.spaces import ActionSemantics, Box, StateField, StateSpec
from inspect_robots.types import Action, Observation, StepResult
from inspect_robots_so101 import STATE_KEY as JOINT_KEY
from inspect_robots_so101 import MOTORS, OperatorIO

from so101_cartesian import STATE_KEY, HoldRewind, SOArmCartesian

ARM_XML = Path(__file__).with_name("so101_mjcf") / "so101.xml"  # MuJoCo Menagerie, Apache-2.0
SIDES = ("left", "right")
# Table frame: origin between the bases on the table, +x forward. The default is side
# by side, 30 cm apart, facing +x; their reach overlaps around y=0.
LEFT_BASE = (0.0, 0.15, 0.0)  # x m, y m, yaw deg
RIGHT_BASE = (0.0, -0.15, 0.0)
# The arm file's solver settings, tuned for grasping; attach() would drop them.
OPTIONS = ("timestep", "integrator", "cone", "iterations", "ls_iterations", "impratio")

DOCS = """\
Two SO-101 arms, left and right, in ONE shared table frame (this replaces the
per-arm base frames the move tool mentions). Meters; +z up, z=0 is the table
surface. {bases} (0 deg faces +x, 90 deg faces +y). Each arm takes its own
<side>_x, <side>_y, <side>_z, <side>_gripper; name only one arm's dimensions to
move it alone. x, y, z place the point between that arm's fingertips; the gripper
points straight down up to about z=0.08 and tilts outward above that. gripper:
0 = closed, 100 = fully open. `eef_pos` reports the commanded poses and grippers,
`joint_pos` (per arm: 5 joints in degrees, then the gripper) the measured ones, so
a closed command whose measured gripper stops well above 0 means an object is
held. A target an arm cannot reach is not executed, and a move that would bring
the arms within {gap:g} cm of each other is held; check `eef_pos` after each move."""


class SilentOperator(OperatorIO):
    """No readiness prompt: for the right arm of a real pair, and for the sim."""

    def wait_ready(self, prompt: str = "") -> None:
        pass


def bases_text(bases: np.ndarray) -> str:
    """The bases as DOCS states them, to the mm and 0.1 deg."""
    return "; ".join(
        f"{side} base at x={x:.3f}, y={y:.3f}, facing {yaw:.1f} deg"
        for side, (x, y, yaw) in zip(SIDES, bases)
    )


def parse_bases(docs: str) -> np.ndarray:
    """``bases_text`` read back from an embodiment's docs: (2, 3) ``(x, y, yaw_deg)``.

    The docs are the only place ``EmbodimentInfo`` carries the bases, so a policy that
    needs them (teleop's FK) reads them here instead of keeping a second copy.
    """
    pattern = r"(left|right) base at x=(-?[\d.]+), y=(-?[\d.]+), facing (-?[\d.]+) deg"
    found = {side: xyz for side, *xyz in re.findall(pattern, docs)}
    if set(found) != set(SIDES):
        raise ValueError("these docs state no left and right base: not an SOArmPair")
    return np.array([found[side] for side in SIDES], dtype=np.float64)


def two_arm_spec(bases: np.ndarray) -> mujoco.MjSpec:
    """Both arms on their ``(x, y, yaw_deg)`` bases, names prefixed ``left_``/``right_``."""
    spec = mujoco.MjSpec()
    for side, (x, y, yaw) in zip(SIDES, bases):
        arm = mujoco.MjSpec.from_file(str(ARM_XML))
        for key in OPTIONS:
            setattr(spec.option, key, getattr(arm.option, key))
        frame = spec.worldbody.add_frame(pos=[x, y, 0.0], euler=[0, 0, yaw])  # MjSpec: degrees
        spec.attach(arm, prefix=f"{side}_", frame=frame)
    return spec


def gripper_angle(model: mujoco.MjModel, side: str, norm: float) -> float:
    """MJCF jaw angle (rad) for a LeRobot gripper value (0..100)."""
    # ponytail: 0..100 spans the MJCF jaw range linearly; LeRobot's calibration sets
    # the real mapping, so the guard's jaw opening is approximate on hardware.
    return np.interp(norm, (0.0, 100.0), model.joint(f"{side}_gripper").range)


class SOArmPair:
    """Two ``SOArmCartesian`` arms behind one 8-D ``eef_abs_pose`` action space."""

    def __init__(
        self,
        left: SOArmCartesian,
        right: SOArmCartesian,
        *,
        left_base: tuple[float, float, float] = LEFT_BASE,  # x m, y m, yaw deg; table frame
        right_base: tuple[float, float, float] = RIGHT_BASE,
        min_gap_m: float = 0.02,  # closest allowed distance between the arms' collision shapes
    ) -> None:
        self._arms = (left, right)
        if any(arm._cfg.settle_tolerance is None for arm in self._arms):
            raise ValueError("both arms need SOArmConfig.settle_tolerance (see so101_pair)")
        self._bases = np.array([left_base, right_base], dtype=np.float64)
        self._rot = Rotation.from_euler("z", self._bases[:, 2:], degrees=True)
        self._origin = np.c_[self._bases[:, :2], np.zeros(2)]
        self._min_gap = min_gap_m
        self._model = two_arm_spec(self._bases).compile()
        self._data = mujoco.MjData(self._model)
        self._qadr = [
            [self._model.joint(f"{side}_{m}").qposadr[0] for m in MOTORS] for side in SIDES
        ]
        root = self._model.body_rootid[self._model.geom_bodyid]
        solid = (self._model.geom_contype | self._model.geom_conaffinity) > 0
        shapes = [
            np.flatnonzero(solid & (root == self._model.body(f"{side}_base").id)) for side in SIDES
        ]
        self._ga, self._gb = np.array([*itertools.product(*shapes)]).T
        rbound = self._model.geom_rbound
        self._reach = rbound[self._ga] + rbound[self._gb] + min_gap_m

        infos = [arm.info for arm in self._arms]
        if infos[0].control_hz != infos[1].control_hz:
            raise ValueError("both arms need the same control_hz")
        cameras = (*infos[0].observation_space.cameras, *infos[1].observation_space.cameras)
        if len({c.name for c in cameras}) < len(cameras):
            raise ValueError("camera names must differ between the arms")
        low, high = [], []
        for i, info in enumerate(infos):
            box = info.action_space
            # ponytail: a non-right-angle yaw makes this world box larger than the arm's
            # own box, and the arm clips the difference; mount at multiples of 90 deg.
            corners = self._to_world(
                i, np.array([*itertools.product(*zip(box.low[:3], box.high[:3]))])
            )
            low += [*np.min(corners, axis=0), box.low[3]]
            high += [*np.max(corners, axis=0), box.high[3]]
        self.info = dataclasses.replace(
            infos[0],
            name="so_arm_pair",
            action_space=Box(
                shape=(8,),
                low=np.array(low),
                high=np.array(high),
                semantics=ActionSemantics(
                    control_mode="eef_abs_pose",
                    gripper="continuous",
                    frame="world",
                    dim_labels=tuple(
                        f"{side}_{d}" for side in SIDES for d in ("x", "y", "z", "gripper")
                    ),
                    max_step=infos[0].action_space.semantics.max_step * 2,
                ),
            ),
            observation_space=dataclasses.replace(
                infos[0].observation_space,
                cameras=cameras,
                state=StateSpec(
                    fields=(
                        StateField(JOINT_KEY, (12,), "deg+normalized"),
                        StateField(STATE_KEY, (8,), "m+normalized"),
                    )
                ),
            ),
            docs=DOCS.format(bases=bases_text(self._bases), gap=self._min_gap * 100),
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Reset the right arm, then the left, whose reset holds the operator prompt; then
        hold the right arm one step so its first frame comes after the prompt too."""
        left, right = self._arms
        right.reset(scene, seed=seed)
        obs = [left.reset(scene, seed=seed), right.execute(right.cmd, {}, meta={}).observation]
        self._measured = [o.state[JOINT_KEY] for o in obs]
        self._floor = self._gap([self._measured])
        return self._merge(obs)

    def step(self, action: Action) -> StepResult:
        """Plan both arms; if that closes the gap below ``min_gap_m``, hold the arms
        (grippers still follow), and if even that does, hold the grippers too."""
        cmds, infos, gap, level = self._plan(action.data)
        results = [
            arm.execute(cmd, info, meta=action.meta)
            for arm, cmd, info in zip(self._arms, cmds, infos)
        ]
        self._measured = [r.observation.state[JOINT_KEY] for r in results]
        # min_gap_m, or for arms that start closer, as far apart as they've been since.
        self._floor = max(self._floor, self._gap([self._measured]))
        ended = [r for r in results if r.terminated]
        return StepResult(
            observation=self._merge([r.observation for r in results]),
            terminated=bool(ended),
            termination_reason=ended[0].termination_reason if ended else None,
            info={
                **{f"{side}_{k}": v for side, r in zip(SIDES, results) for k, v in r.info.items()},
                "arm_gap_m": gap,
                "pair_held": level,  # 0 none, 1 arms held, 2 arms and grippers held
            },
        )

    def ran(self, data: np.ndarray) -> np.ndarray:
        """The target a step toward ``data`` actually runs: held arms stay where they are."""
        cmds = self._plan(data)[0]
        ran = np.asarray(data, dtype=np.float64).reshape(2, 4).copy()
        for i, (arm, cmd) in enumerate(zip(self._arms, cmds)):
            if np.array_equal(cmd[:5], arm.cmd[:5]):
                ran[i, :3] = self._to_world(i, arm.fk(cmd)[:3, 3])
            ran[i, 3] = cmd[5]
        return ran.reshape(-1)

    def contribute_guardrails(self, action_space: Box) -> GuardrailContribution:
        """CLI hook: apply holds in the approver chain (see ``HoldRewind``)."""
        return GuardrailContribution(approvers=(("holds", HoldRewind(self.ran)),))

    def close(self) -> None:
        """Release both arms."""
        try:
            self._arms[0].close()
        finally:
            self._arms[1].close()

    def _plan(self, data: np.ndarray) -> tuple[list[np.ndarray], list[dict], float, int]:
        """Commands for a step toward ``data``, the arms' plan infos, the gap and hold level."""
        targets = np.asarray(data, dtype=np.float64).reshape(2, 4)
        plans = [
            arm.plan(np.append(self._to_base(i, t[:3]), t[3]))
            for i, (arm, t) in enumerate(zip(self._arms, targets))
        ]
        planned = [cmd for cmd, _ in plans]
        arms_held = [np.append(arm.cmd[:5], cmd[5]) for arm, cmd in zip(self._arms, planned)]
        for level, cmds in enumerate((planned, arms_held, [arm.cmd for arm in self._arms])):
            gap = self._gap(self._new_poses(cmds))
            # Moving apart is always allowed, so arms that start too close can leave.
            if gap >= self._floor:
                break
        return cmds, [info for _, info in plans], gap, level

    def _new_poses(self, cmds: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        """(left, right) poses a step to ``cmds`` can bring about that the arms aren't in
        already: the left moves first, then the right, each to its command with the jaw
        there or stalled where measured, or staying where measured (blocked)."""
        (left, right), (m_left, m_right) = self._arms, self._measured

        def at(cmd: np.ndarray, measured: np.ndarray) -> list[np.ndarray]:
            return [cmd, np.append(cmd[:5], measured[5])]

        poses = []
        if not np.array_equal(cmds[0], left.cmd):
            poses += itertools.product(at(cmds[0], m_left), [*at(right.cmd, m_right), m_right])
        if not np.array_equal(cmds[1], right.cmd):
            poses += itertools.product([*at(cmds[0], m_left), m_left], at(cmds[1], m_right))
        return poses

    def _gap(self, poses: list) -> float:
        """Closest distance (m) between the arms over these (left, right) poses (5 joints in
        deg, then the gripper), capped at min_gap_m."""
        gap = self._min_gap
        for pair in poses:
            for side, adr, arm, pose in zip(SIDES, self._qadr, self._arms, pair):
                opening = gripper_angle(self._model, side, pose[5])
                self._data.qpos[adr] = [*arm.urdf_angles(pose), opening]
            mujoco.mj_kinematics(self._model, self._data)
            # Only shapes whose bounding spheres come within min_gap_m can be that close.
            xpos = self._data.geom_xpos
            near = np.linalg.norm(xpos[self._ga] - xpos[self._gb], axis=1) < self._reach
            for a, b in zip(self._ga[near], self._gb[near]):
                gap = min(gap, mujoco.mj_geomDistance(self._model, self._data, a, b, gap, None))
        return gap

    def _to_base(self, i: int, xyz: np.ndarray) -> np.ndarray:
        return self._rot[i].apply(xyz - self._origin[i], inverse=True)

    def _to_world(self, i: int, xyz: np.ndarray) -> np.ndarray:
        return self._rot[i].apply(xyz) + self._origin[i]

    def _merge(self, obs: list[Observation]) -> Observation:
        eef = [
            np.append(self._to_world(i, o.state[STATE_KEY][:3]), o.state[STATE_KEY][3])
            for i, o in enumerate(obs)
        ]
        return Observation(
            images={**obs[0].images, **obs[1].images},
            state={
                JOINT_KEY: np.concatenate([o.state[JOINT_KEY] for o in obs]),
                STATE_KEY: np.concatenate(eef),
            },
            instruction=obs[0].instruction,
            image_times={**obs[0].image_times, **obs[1].image_times},
            state_time=max(o.state_time for o in obs),
        )
