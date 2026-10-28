"""Self-check for so101_cartesian, no hardware: `python test_so101_cartesian.py`."""

import itertools

import numpy as np
from inspect_robots.conformance import assert_embodiment_conformant
from inspect_robots.errors import EmbodimentFault
from inspect_robots.scene import Scene
from inspect_robots.types import Action
from inspect_robots_agent._tools import build_toolset
from inspect_robots_so101 import MOTORS, SOArmConfig, SOArmEmbodiment

from so101_cartesian import DOWN, IK_TOL_M, MAX_JOINT_STEP_DEG, SOArmCartesian
from so101_pair import SilentOperator  # OperatorIO's stdin drain uses select(): no Windows

# Folded, gripper down, ~2.7 cm outside the workspace: the first clipped step must not snap.
START = (0.0, -60.0, 60.0, 90.0, 0.0, 0.0)
NOT_DOWN = (20.1, -39.9, 93.1, -73.6, 0.0, 0.0)  # inside the workspace, gripper not down


class FakeDriver:
    """Servos that reach every command instantly, except a gripper blocked open at ``stall``."""

    def __init__(self, start: tuple[float, ...], stall: float) -> None:
        self.pos = {f"{m}.pos": v for m, v in zip(MOTORS, start)}  # last commanded
        self.stall = stall

    def get_observation(self) -> dict:
        gripper = max(self.pos["gripper.pos"], self.stall)
        return {**self.pos, "gripper.pos": gripper, "front": np.zeros((480, 640, 3), np.uint8)}

    def send_action(self, action: dict) -> dict:
        self.pos.update(action)
        return dict(action)

    def disconnect(self) -> None:
        pass


def make(
    start: tuple[float, ...] = START,
    stall: float = 0.0,
    joint_low: tuple[float, ...] = SOArmConfig.joint_low,
    **kw: tuple[float, ...],  # SOArmCartesian knobs
) -> SOArmCartesian:
    inner = SOArmEmbodiment(
        joint_low=joint_low,
        driver_factory=lambda _cfg: FakeDriver(start, stall),
        operator=SilentOperator(),
        poll_end=lambda: False,
        sleep_fn=lambda _s: None,
    )
    return SOArmCartesian(inner, **kw)


def main() -> None:
    emb = make()
    assert_embodiment_conformant(emb.info)
    tools = build_toolset(emb.info.action_space, emb.info.observation_space, emb.info.control_hz)
    assert tools.schemas()[0]["function"]["name"] == "move_to"
    for bad in ({"use_degrees": False}, {"joints_are_delta": True}):  # units the IK can't drive
        try:
            SOArmCartesian(SOArmEmbodiment(**bad))
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass

    # The whole default workspace is reachable, and grasp height is gripper-down.
    seed = np.array([0.0, -60.0, 60.0, 90.0, 0.0])
    for xyz in itertools.product(*np.linspace(emb._low, emb._high, 5).T):
        assert emb.ik(np.array(xyz), seed)[1] <= IK_TOL_M, xyz
    joints, err = emb.ik(np.array([0.22, 0.0, 0.02]), seed)
    tilt = np.degrees(np.arccos(np.clip(emb.fk(joints)[:3, 2] @ DOWN, -1, 1)))
    assert err <= IK_TOL_M and tilt < 5, (err, tilt)

    # Agent-style straight-line move: no joint snaps, and eef_pos lands on the target.
    obs = emb.reset(Scene(id="s", instruction="test"))
    start, goal = obs.state["eef_pos"], np.array([0.25, 0.10, 0.05, 30.0])
    joints = obs.state["joint_pos"]
    for t in np.linspace(0, 1, 40)[1:]:
        result = emb.step(Action(data=start + t * (goal - start)))
        assert not result.info["ik_held"], result.info
        moved = np.abs(result.observation.state["joint_pos"][:5] - joints[:5])
        assert moved.max() <= MAX_JOINT_STEP_DEG + 1e-9, moved
        joints = result.observation.state["joint_pos"]
    assert np.allclose(result.observation.state["eef_pos"][:3], goal[:3], atol=IK_TOL_M), (
        result.observation.state["eef_pos"]
    )
    assert result.observation.state["joint_pos"][5] == goal[3]

    # A NaN target raises instead of reaching the motors.
    try:
        emb.step(Action(data=np.array([0.2, 0.0, 0.1, np.nan])))
        raise AssertionError("NaN target was accepted")
    except ValueError:
        pass

    # A start pose that isn't gripper-down is refused before anything moves.
    try:
        make(NOT_DOWN).reset(Scene(id="s", instruction="test"))
        raise AssertionError("non-gripper-down start was accepted")
    except EmbodimentFault:
        pass

    # Mid-trial off-down pose: the capped joint catch-up never dips below the table guard.
    emb.cmd[:5] = (-36.0, 29.0, 53.0, -68.0, 0.0)
    above = emb.fk(emb.cmd)[:3, 3]
    for _ in range(10):
        result = emb.step(Action(data=np.append(above[:2], [emb._low[2], 50.0])))
        assert emb.fk(result.observation.state["joint_pos"])[2, 3] >= emb._low[2] - IK_TOL_M

    # A reached target the IK left a hair outside the workspace (at a corner) stays put.
    for corner in ((0.16, -0.20, 0.20), (0.30, -0.20, 0.01)):
        emb.cmd[:5] = emb.ik(np.array(corner), emb.cmd)[0]
        here = emb.step(Action(data=np.append(corner, 50.0))).observation.state["eef_pos"]
        for _ in range(3):
            assert np.array_equal(emb.step(Action(data=here)).observation.state["eef_pos"], here)

    # A big step near a far corner at table height must make progress, not hold forever.
    emb.cmd[:5] = (41.3, 51.1, -50.7, 89.6, 0.0)
    corner = np.array([0.2424, -0.1814, 0.01, 50.0])
    for _ in range(10):
        result = emb.step(Action(data=corner))
        assert not result.info["ik_held"], result.info
    assert np.allclose(result.observation.state["eef_pos"][:3], corner[:3], atol=IK_TOL_M)

    # Calibrated limits tighter than the URDF: the inner clamp never alters a command, so
    # ik_held and the table guard judge the joints that actually run.
    tight = make(
        joint_low=(-40.0, -180.0, -63.8, -180.0, -180.0, 0.0),
        joint_offset_deg=(-5.0, 0.0, 0.0, 0.0, 0.0),  # nonzero, so a flipped sign shows up
    )
    tight.reset(Scene(id="s", instruction="test"))
    for xyz in itertools.product(*np.linspace(tight._low, tight._high, 5).T):
        for _ in range(5):
            ran = tight.step(Action(data=np.append(xyz, 50.0))).observation.state["joint_pos"]
            assert np.allclose(ran[:5], tight.cmd[:5]), (xyz, ran, tight.cmd)
            assert tight.fk(ran)[2, 3] >= tight._low[2] - IK_TOL_M, xyz
    xyz = np.array([0.22, 0.05, 0.05])  # ik and fk agree on the offset sign
    assert np.allclose(tight.fk(tight.ik(xyz, tight.cmd)[0])[:3, 3], xyz, atol=IK_TOL_M)

    # A start outside the calibrated limits is refused, not snapped into range at full speed.
    try:
        outside = (-90.0, *START[1:])
        make(outside, joint_low=(-40.0, *SOArmConfig.joint_low[1:])).reset(
            Scene(id="s", instruction="test")
        )
        raise AssertionError("out-of-limits start was accepted")
    except EmbodimentFault:
        pass

    # Starting under a raised table guard (high table): climbing out is allowed, not held.
    high = make(workspace_low=(0.16, -0.20, 0.06))
    obs = high.reset(Scene(id="s", instruction="test"))
    assert obs.state["eef_pos"][2] < 0.06 - IK_TOL_M  # START really is under the guard
    for _ in range(5):
        result = high.step(Action(data=obs.state["eef_pos"]))  # a gripper-only move_to
        assert not result.info["ik_held"], result.info
        obs = result.observation

    # Holding an object (gripper commanded 0, blocked at 15): a lift that leaves the
    # gripper out, so move_to fills it from eef_pos, must keep squeezing.
    grip = make(stall=15.0)
    obs = grip.reset(Scene(id="s", instruction="test"))
    for _ in range(4):  # 15 -> 0 at the 4-unit gripper cap
        obs = grip.step(Action(data=np.append(obs.state["eef_pos"][:3], 0.0))).observation
    assert obs.state["joint_pos"][5] == 15.0
    lift = obs.state["eef_pos"] + [0.0, 0.0, 0.004, 0.0]
    grip.step(Action(data=lift))
    assert grip._inner._driver.pos["gripper.pos"] == 0.0

    # An out-of-reach target (allowed by a too-big workspace) holds the arm; the gripper follows.
    far = make(workspace_high=(0.50, 0.20, 0.30))
    before = far.reset(Scene(id="s", instruction="test")).state["joint_pos"].copy()
    for _ in range(20):
        result = far.step(Action(data=np.array([0.50, 0.0, 0.30, 80.0])))
        assert result.info["ik_held"]
    assert np.allclose(result.observation.state["joint_pos"][:5], before[:5])
    assert result.observation.state["joint_pos"][5] == 80.0
    # The CLI hook swaps a held target for the pose that will run.
    [(_, holds)] = far.contribute_guardrails(far.info.action_space).approvers
    ran = holds.review(Action(data=np.array([0.50, 0.0, 0.30, 80.0])), {}).data
    assert np.allclose(ran, np.append(far.fk(far.cmd)[:3, 3], 80.0)), ran

    # The gripper is rate-capped like the joints: a jump target moves it 4 units per step.
    jump = far.step(Action(data=np.array([0.50, 0.0, 0.30, 0.0])))
    assert jump.observation.state["joint_pos"][5] == 76.0
    print("ok")


if __name__ == "__main__":
    main()
