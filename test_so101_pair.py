"""Self-check for so101_pair and so101_sim, no hardware: `python test_so101_pair.py`."""

import json
import tempfile
from functools import partial
from types import SimpleNamespace

import mujoco
import numpy as np
from inspect_robots import eval as run_eval
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots.conformance import assert_embodiment_conformant
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.types import ActionChunk, Observation
from inspect_robots_agent._tools import Toolset, build_toolset

from so101_cartesian import IK_TOL_M
from so101_pair import SIDES
from so101_sim import SETUPS, SOArmPairSim
from so101_tasks import pick_cube

SCENE = Scene(id="s", instruction="test")


def toolset(emb: SOArmPairSim) -> Toolset:
    return build_toolset(emb.info.action_space, emb.info.observation_space, emb.info.control_hz)


def rails(emb: SOArmPairSim) -> ChainApprover:
    """The CLI's guardrails: bounds clamp, delta limit, then the embodiment's own."""
    space = emb.info.action_space
    extra = (approver for _, approver in emb.contribute_guardrails(space).approvers)
    return ChainApprover(ClampApprover(space), DeltaLimitApprover(space), *extra)


def move(tools: Toolset, obs: Observation, targets: dict[str, float]) -> ActionChunk:
    """The chunk the agent's move_to tool plays out for these targets."""
    args = json.dumps({"targets": targets, "note": "test"})
    result = tools.execute(SimpleNamespace(name="move_to", arguments=args), obs)
    assert result.error is None, result.error
    return result.chunk


def xyz(side: str, *values: float) -> dict[str, float]:
    return {f"{side}_{d}": v for d, v in zip("xyz", values)}


def play(emb: SOArmPairSim, chunk: ActionChunk, review=lambda action: action) -> list:
    return [emb.step(review(action)) for action in chunk.actions]


def world_gap(emb: SOArmPairSim, cap: float = 0.005) -> float:
    """The arms' closest distance in the sim world itself, capped at ``cap``: every left x
    right collision shape, no prefilter, so it checks ``SOArmPair._gap`` independently."""
    m, d = emb.world.model, emb.world.data
    root = m.body_rootid[m.geom_bodyid]
    solid = (m.geom_contype | m.geom_conaffinity) > 0
    left, right = (np.flatnonzero(solid & (root == m.body(f"{s}_base").id)) for s in SIDES)
    return min(mujoco.mj_geomDistance(m, d, a, b, cap, None) for a in left for b in right)


def play_guarded(emb: SOArmPairSim, chunk: ActionChunk) -> list:
    """``play``, asserting after every step: the arms at least 5 mm apart in the world
    (none of the stress tests came within 11), the guard's floor kept, and the left arm's
    new pose clear of the right's old one (the left moves first)."""
    results = []
    for action in chunk.actions:
        right_was = emb._arms[1].cmd  # step() replaces it, so this stays the old one
        results.append(emb.step(action))
        info = results[-1].info
        assert world_gap(emb) >= 0.005, info
        assert emb._floor >= emb._min_gap, info
        assert emb._gap([(emb._arms[0].cmd, right_was)]) >= emb._min_gap - 1e-3, info
    return results


def sites_match(emb: SOArmPairSim, eef: np.ndarray) -> bool:
    """eef_pos (URDF IK frame, base-to-table transform) sits on MuJoCo's grasp sites, up
    to the sag under the commanded pose it reports."""
    sites = [emb.world.data.site(f"{side}_gripperframe").xpos for side in SIDES]
    return np.allclose(eef[[0, 1, 2, 4, 5, 6]], np.concatenate(sites), atol=1e-3)


class Oracle:
    """Picks the cube with the agent's move_to tool, reading its pose from the sim."""

    def __init__(self, emb: SOArmPairSim) -> None:
        self._emb, self._tools = emb, toolset(emb)
        self.info = PolicyInfo(
            name="oracle",
            action_space=emb.info.action_space,
            observation_space=emb.info.observation_space,
        )
        self.config = PolicyConfig()

    def reset(self, scene: Scene) -> None:
        self._stage = 0

    def act(self, observation: Observation) -> ActionChunk:
        x, y, _ = self._emb.world.data.body("cube").xpos
        stages = ({"x": x, "y": y, "z": 0.08, "gripper": 100}, {"z": 0.015}, {"gripper": 0})
        side = "left" if y >= 0 else "right"
        stage = stages[self._stage] if self._stage < len(stages) else {"z": 0.12}
        self._stage += 1
        return move(self._tools, observation, {f"{side}_{k}": v for k, v in stage.items()})


def main() -> None:
    emb = SOArmPairSim()
    assert_embodiment_conformant(emb.info)
    assert "right_gripper" in toolset(emb).schemas()[0]["function"]["parameters"]["properties"][
        "targets"
    ]["description"]
    obs = emb.reset(SCENE)
    assert sites_match(emb, obs.state["eef_pos"])
    # The left arm's reset holds the operator prompt; the right arm's frame comes after it.
    assert obs.image_times["right_wrist_cam"] > obs.image_times["top"], obs.image_times

    # The task runs through eval() with the CLI's guardrails, and the sim scores it.
    with tempfile.TemporaryDirectory() as log_dir:
        task = pick_cube(num_scenes=3, max_seconds=20)
        [log] = run_eval(task, Oracle(emb), emb, log_dir=log_dir, approver=rails(emb))
    assert log.status == "success", log.error
    assert log.results.metrics["success_at_end"] == 1.0, log.results.metrics

    # Both arms driven at the same spot, then one chasing the other: the guard holds them
    # before they touch, and pulling apart afterwards is never held.
    space, rng = emb.info.action_space, np.random.default_rng(0)
    tools, engaged = toolset(emb), 0
    for _ in range(6):
        obs = emb.reset(SCENE)
        for _ in range(3):
            p = rng.uniform((0.18, -0.05, 0.02), (0.28, 0.05, 0.15))
            q = np.minimum(p + rng.uniform((-0.03, -0.03, -0.01), 0.03), (0.30, 0.05, 0.20))
            values = (*p, rng.uniform(0, 100), *q, rng.uniform(0, 100))
            targets = dict(zip(space.semantics.dim_labels, values))
            results = play_guarded(emb, move(tools, obs, targets))
            engaged += sum(r.info["pair_held"] > 0 for r in results)
            obs = results[-1].observation
        # The left arm chases the right into the space it leaves.
        right = obs.state["eef_pos"][4:7]
        x, y, z = np.clip(right, space.low[:3], space.high[:3])
        chase = {"left_x": x, "left_y": y, "left_z": z, "right_y": right[1] - 0.05}
        obs = play_guarded(emb, move(tools, obs, chase))[-1].observation
        for result in play(emb, move(tools, obs, {"left_y": 0.2, "right_y": -0.2})):
            assert result.info["pair_held"] == 0, result.info
    assert engaged, "the guard never engaged; the test isn't exercising it"

    # An arm blocked by an object is guarded where it is, not where it was sent: it retreats
    # into a low wall on its own side, stalls mid-table, and the other arm comes at it.
    for blocked, other, sign in (("left", "right", 1), ("right", "left", -1)):
        SETUPS["wall"] = lambda spec, rng, s=sign: spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.01, 0.03], pos=[0.25, s * 0.04, 0.03]
        )
        obs = emb.reset(Scene(id="w", instruction="test", setup="wall"))
        for target in ((0.25, 0.0, 0.1), (0.25, 0.0, 0.03), (0.25, sign * 0.12, 0.03)):
            obs = play(emb, move(tools, obs, xyz(blocked, *target)))[-1].observation
        stalled = emb.world.data.site(f"{blocked}_gripperframe").xpos
        play_guarded(emb, move(tools, obs, xyz(other, *stalled)))

    # Through the CLI's guardrails: arms that start closer than min_gap_m may move apart
    # but not closer; the held move doesn't drag the next one back; once apart, min_gap_m
    # holds again.
    near = SOArmPairSim(min_gap_m=1.0)  # gaps are capped at min_gap_m: this reads the real one
    start, tools, review = near.reset(SCENE), toolset(near), partial(rails(near).review, store={})
    near._min_gap = near._floor + 0.005
    obs = play(near, move(tools, start, {"left_y": 0, "right_y": 0}), review)[-1].observation
    assert np.array_equal(obs.state["eef_pos"], start.state["eef_pos"]), obs.state["eef_pos"]
    obs = play(near, move(tools, obs, {"left_y": 0.25, "right_y": -0.25}), review)[-1].observation
    assert np.allclose(obs.state["eef_pos"][[1, 5]], (0.25, -0.25), atol=1e-3), obs.state
    for _ in play(near, move(tools, obs, {"left_y": 0, "right_y": 0}), review):
        assert near._gap([near._measured]) >= near._min_gap - 1e-3
    near.close()

    # Facing each other: a yawed base still puts eef_pos on the grasp site, and the arm on
    # the target. The left arm, not named, stays exactly where it was commanded.
    facing = SOArmPairSim(right_base=(0.45, 0.0, 180.0))
    obs = facing.reset(SCENE)
    assert sites_match(facing, obs.state["eef_pos"])
    goal = (0.22, -0.05, 0.05)
    targets = dict(zip(("right_x", "right_y", "right_z"), goal))
    after = play(facing, move(toolset(facing), obs, targets))[-1].observation
    assert np.array_equal(after.state["eef_pos"][:4], obs.state["eef_pos"][:4])
    site = facing.world.data.site("right_gripperframe").xpos
    assert np.allclose(site, goal, atol=IK_TOL_M), site
    facing.close()
    emb.close()
    print("ok")


if __name__ == "__main__":
    main()
