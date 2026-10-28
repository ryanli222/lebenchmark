"""Self-check for so101_demo, the demo tasks and the sim's demo support, no hardware or
API key: `python test_so101_demo.py`. Builds one SOArmPairSim at a time (RAM)."""

import base64
import contextlib
import io
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import mujoco
import numpy as np
from inspect_robots import eval as run_eval
from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.errors import ConfigError, EmbodimentFault
from inspect_robots.rollout import StepRecord, TrialRecord
from inspect_robots.scene import Scene
from inspect_robots.types import Action, ActionChunk, Observation, StepResult
from inspect_robots_agent import encode_png
from inspect_robots_so101 import SOArmConfig, SOArmEmbodiment

from so101_cartesian import SOArmCartesian
from so101_demo import MAX_IMAGES, ContextAgent, FakeLeader, TeleopPolicy, keyframes, open_leader
from so101_pair import DOCS, SIDES, SOArmPair, bases_text, parse_bases
from so101_sim import START, SOArmPairSim, at_rest
from so101_tasks import imitate, match_goal, pick_cube
from test_so101_pair import SCENE, Oracle, move, rails

# The pair's spaces and docs at the default bases, without a sim: hardware arms, never
# connected.
cfg = dict(cam_height=48, cam_width=64, settle_tolerance=0.5)
arms = (SOArmCartesian(SOArmEmbodiment(SOArmConfig(cameras=c, **cfg))) for c in (("top",), ()))
INFO = SOArmPair(*arms).info


def fake_obs(eef: np.ndarray, roll=0.0, frame: int = 0) -> Observation:
    """An observation of the stub pair: these eef_pos, the followers' wrist roll at ``roll``
    (both, or a (left, right) pair)."""
    joints = np.zeros(12)
    joints[[4, 10]] = roll
    return Observation(
        images={"top": np.full((48, 64, 3), frame, np.uint8)},
        state={"joint_pos": joints, "eef_pos": np.asarray(eef, dtype=np.float64)},
    )


def teleop(info: EmbodimentInfo, *leaders, **kw) -> TeleopPolicy:
    policy = TeleopPolicy(leaders=leaders, **kw)
    policy.bind(info)
    policy.reset(SCENE)
    return policy


def act(policy, obs: Observation) -> np.ndarray:
    return np.asarray(policy.act(obs).actions[0].data)


def check_context_agent(tmp: Path, wire: str) -> None:
    """The pinned demo image survives the agent's image horizon, on the chat and messages
    wires, after the goal and the context's text and label; the agent logs as context_agent."""
    png = tmp / "pin.png"
    png.write_bytes(encode_png(np.full((48, 64, 3), 200, np.uint8)))
    pin = base64.b64encode(png.read_bytes()).decode()
    context = [{"text": "A person did this once."}, {"label": "Demo frame", "png": str(png)}]
    scene = Scene(id="c", instruction="Pick up the red cube.", metadata={"context": context})
    sent = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request.content.decode())
        args = {"targets": {"left_z": 0.1}, "note": "mock"}
        if wire == "messages":
            use = {"type": "tool_use", "id": f"t{len(sent)}", "name": "move_to", "input": args}
            return httpx.Response(200, json={"stop_reason": "tool_use", "content": [use]})
        call = {"id": f"c{len(sent)}", "type": "function"}
        call["function"] = {"name": "move_to", "arguments": json.dumps(args)}
        message = {"role": "assistant", "content": None, "tool_calls": [call]}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    agent = ContextAgent(
        model="test/model",
        base_url="http://mock",
        wire=wire,
        env={},
        transport=httpx.MockTransport(handle),
    )
    agent.bind(INFO)
    agent.reset(scene)
    eef = np.array([0.2, 0.15, 0.08, 50.0, 0.2, -0.15, 0.08, 50.0])
    observations = [fake_obs(eef, frame=10 * i) for i in range(3)]
    for obs in observations:
        agent.act(obs)
    frames = [base64.b64encode(encode_png(o.images["top"])).decode() for o in observations]
    assert len(sent) == 3 and pin in sent[2], len(sent)
    assert [f in sent[2] for f in frames] == [False, True, True]  # the horizon still works
    assert "Goal: Pick up the red cube." in sent[2] and "A person did this once." in sent[2]
    assert "Demo frame:" in sent[2]  # the image's label part
    assert agent.info.name == "context_agent", agent.info.name
    if wire == "messages":  # call 2 stubs the pinned turn: its restored copy keeps the anchor
        assert [s.count("cache_control") for s in sent] == [2, 3, 3], sent


def check_open_leader() -> None:
    """open_leader on a stub of LeRobot's leader: keyword config, never LeRobot's
    interactive calibration; no lerobot, a bus error or no calibration halts the eval;
    teleop opens each leader once, on its port under its id."""
    calls = []

    class Leader:
        calibration_fpath, is_calibrated = "leader.json", False

        def __init__(self, config: dict) -> None:
            calls.append(config)

        def connect(self, calibrate: bool = True) -> None:
            calls.append(calibrate)

        def disconnect(self) -> None:
            calls.append("off")

    class Flaky(Leader):  # is_calibrated reads the motors' EEPROM
        is_calibrated = property(lambda self: 1 / 0)

    name = "lerobot.teleoperators.so_leader"
    for leader in (None, Leader, Flaky):  # None: lerobot isn't installed
        sys.modules[name] = leader and SimpleNamespace(SOLeader=leader, SOLeaderTeleopConfig=dict)
        try:
            open_leader("/dev/ttyACM0", 1)
            raise AssertionError("open_leader accepted it")
        except EmbodimentFault:
            pass
    config = {"port": "/dev/ttyACM0", "id": "1", "use_degrees": True}
    assert calls == [config, False, "off", config, False], calls
    # teleop opens each leader once, on its own port under its own id.
    Leader.is_calibrated, calls[:] = True, []
    sys.modules[name] = SimpleNamespace(SOLeader=Leader, SOLeaderTeleopConfig=dict)
    pair = TeleopPolicy(left_port="L", right_port="R")
    pair.reset(SCENE)
    pair.reset(SCENE)
    del sys.modules[name]
    opened = [(c["port"], c["id"]) for c in calls if type(c) is dict]
    assert opened == [("L", "leader_left"), ("R", "leader_right")], calls


def check_teleop_pure(tmp: Path) -> None:
    """Bases round trip; FK, roll, offsets and catch gating on fake observations; the
    operator's stderr lines; a demo of a pinned scene with store_frames off, and one from
    a rig without cameras."""
    bases = np.array([[0.1, 0.2, 0.0], [0.45, -0.123, -180.0]])
    assert np.array_equal(parse_bases(DOCS.format(bases=bases_text(bases), gap=2)), bases)
    assert keyframes(np.array([[0.0], [2.0], [1.0]])) == [0, 1, 2]  # a turn back is kept
    # Leaders at START: step 0 holds whatever eef_pos says, step 1 gives the leader point.
    start = teleop(INFO, FakeLeader([START]), FakeLeader([START]), catch_m=1.0)
    still = fake_obs(np.zeros(8))
    assert np.array_equal(act(start, still), np.zeros(8))
    here = act(start, still)
    # A leader that stops answering mid-run halts the eval, not just this trial.
    dead = teleop(INFO, SimpleNamespace(get_action=lambda: 1 / 0), FakeLeader([START]))
    try:
        act(dead, still)
        raise AssertionError("a dead leader didn't halt the eval")
    except EmbodimentFault:
        pass
    # A leader roll of 90 deg doesn't move the point; a follower roll of 90 does (FK uses it).
    rolled = (*START[:4], 90.0, START[5])
    twisted = teleop(INFO, FakeLeader([rolled]), FakeLeader([rolled]), catch_m=1.0)
    act(twisted, still)
    moved = act(twisted, fake_obs(np.zeros(8), roll=(0.0, 90.0)))  # each arm its own follower's
    assert np.allclose(moved[:4], here[:4]) and not np.allclose(moved[4:], here[4:]), moved
    # The next trial starts uncaught: its step 0 holds again.
    start.reset(SCENE)
    assert np.array_equal(act(start, still), np.zeros(8))
    # A leader reading START + its offset gives START's point: URDF = reading - offset;
    # the 5th (roll) offset is the follower's, taken off the follower's roll.
    lead = FakeLeader([np.add(START, (10, -5, 3, 7, 0, 0))])
    offsets = {"left_offset": "10,-5,3,7,20", "right_offset": "0,0,0,0,20"}
    off = teleop(INFO, lead, FakeLeader([START]), catch_m=1.0, **offsets)
    act(off, fake_obs(np.zeros(8), roll=20.0))
    assert np.allclose(act(off, fake_obs(np.zeros(8), roll=20.0)), here, rtol=0, atol=1e-9)
    # A mistyped offset (here the roll left off) names the flag, not a numpy traceback.
    try:
        TeleopPolicy(leaders=(), left_offset="10,-5,3,0")
        raise AssertionError("a 4-value offset was accepted")
    except ConfigError:
        pass
    # The right arm 10 cm from its leader: step 0 holds both, then only the left follows;
    # the right follows once within catch_m (decided a step later), then stays caught.
    far = here + [0, 0, 0, 0, 0.1, 0, 0, 0]
    with contextlib.redirect_stderr(io.StringIO()) as err:
        gate = teleop(INFO, FakeLeader([START]), FakeLeader([START]))
        assert np.array_equal(act(gate, fake_obs(far)), far)
        gate._nag = 0.0  # a second later: the unchanged status isn't repeated
        assert np.array_equal(act(gate, fake_obs(far)), [*here[:4], *far[4:]])
    shown = err.getvalue()  # the operator sees the instruction, and each status once
    assert "teleop: test" in shown and shown.count("right 10 cm") == 1, shown
    near = here + [0, 0, 0, 0, 0.02, 0, 0, 0]
    assert np.array_equal(act(gate, fake_obs(near)), [*here[:4], *near[4:]])
    assert np.array_equal(act(gate, fake_obs(far)), here)
    assert np.array_equal(act(gate, fake_obs(far)), here)  # latched: lagging 10 cm still follows
    # A demo of a pinned scene with store_frames off (images in the record): it keeps the
    # pinned world, drops a 3-unit gripper wobble (1 unit = 1 mm, under the 5 mm tolerance),
    # times waypoints in seconds, shows the last frame, and its metadata path is POSIX.
    gate.reset(Scene(id="p", instruction="t", metadata={"world_seed": 5}))
    before, after = fake_obs(np.zeros(8), frame=1), fake_obs(np.ones(8), frame=2)
    wobble = fake_obs(np.r_[0.5, 0.5, 0.5, 3.5, 0.5, 0.5, 0.5, 3.5])
    steps = [StepRecord(0, before, Action(data=np.zeros(8)), StepResult(wobble))]
    steps += [StepRecord(1, wobble, Action(data=np.zeros(8)), StepResult(after))]
    record = TrialRecord(scene_id="p", epoch=0, seed=9, steps=steps, operator_judgement="yes")
    gate.on_trial_end(record, str(tmp), "run")
    assert record.metadata["demo"] == "demos/run/p-e0.json", record.metadata
    demo = json.loads((tmp / record.metadata["demo"]).read_text(encoding="utf-8"))
    assert demo["world_seed"] == 5, demo["world_seed"]
    assert [w["t_s"] for w in demo["waypoints"]] == [0.0, 0.067], demo["waypoints"]
    assert (tmp / "demos/run/p-e0-2.png").read_bytes() == encode_png(after.images["top"])
    # A rig without cameras still gets its demo, with no frames; match-goal refuses it.
    blind = (SOArmCartesian(SOArmEmbodiment(SOArmConfig(cameras=(), **cfg))) for _ in SIDES)
    dark = Observation(images={}, state=still.state)
    steps = [StepRecord(0, dark, Action(data=np.zeros(8)), StepResult(dark))]
    record = TrialRecord(scene_id="b", epoch=0, seed=1, steps=steps, operator_judgement="no")
    teleop(SOArmPair(*blind).info, FakeLeader([START]), FakeLeader([START])).on_trial_end(
        record, str(tmp), "run"
    )
    try:
        match_goal(str(tmp / "demos/run/b-e0.json"))
        raise AssertionError("match-goal accepted a demo without a goal image")
    except ConfigError:
        pass


class PickLeader:
    """A fake leader that picks the cube, if it's on this arm's side, from the follower's
    START pose; ``place``: then puts it down 5 cm outward, lets go and backs off. Planned
    by IK at the first read, when the trial's world exists; it records the cube there."""

    def __init__(self, emb: SOArmPairSim, side: str, place: bool = False) -> None:
        self._emb, self._i, self._place, self._leader = emb, SIDES.index(side), place, None

    def get_action(self) -> dict[str, float]:
        if self._leader is None:
            self.cube = self._emb.world.data.body("cube").xpos.copy()
            self._leader = FakeLeader(self._plan())
        return self._leader.get_action()

    def _plan(self) -> list[np.ndarray]:
        emb, i = self._emb, self._i
        (x, y, _), arm = self.cube, emb._arms[i]
        if (y >= 0) != (i == 0):
            return [START]
        stops = [((x, y, 0.08), 100), ((x, y, 0.015), 100), ((x, y, 0.015), 0)]
        stops += [((x, y, 0.12), 0)]
        if self._place:
            y2 = y + (0.05 if i == 0 else -0.05)
            stops[3:] = [((x, y, 0.06), 0), ((x, y2, 0.06), 0), ((x, y2, 0.017), 0)]
            stops += [((x, y2, 0.017), 100), ((x, y2, 0.08), 100)]
        q, xyz = np.array(START, dtype=np.float64), emb._to_world(i, arm.fk(START)[:3, 3])
        poses = [q]
        for goal, grip in stops:
            joints = arm.ik(emb._to_base(i, np.array(goal)), q)[0]
            nxt = np.append(joints, grip)
            # 3 mm and 3 gripper units per step: under the 4 mm cap, so the arm keeps up.
            n = max(1, math.ceil(max(math.dist(goal, xyz) / 0.003, abs(grip - q[5]) / 3)))
            poses += list(np.linspace(q, nxt, n + 1)[1:])
            q, xyz = nxt, goal
        return poses


class Replay(Oracle):
    """Plays move_to targets in order, then holds the last."""

    def __init__(self, emb: SOArmPairSim, moves: list[dict[str, float]]) -> None:
        super().__init__(emb)
        self._moves = moves

    def act(self, observation: Observation) -> ActionChunk:
        targets = self._moves[min(self._stage, len(self._moves) - 1)]
        self._stage += 1
        return move(self._tools, observation, targets)


def run(task, policy, emb: SOArmPairSim, log_dir: Path, seed: int = 0):
    """eval() with the CLI's guardrails, frames streamed to disk (RAM: ~2.8 MB per step)."""
    [log] = run_eval(
        task, policy, emb, log_dir=str(log_dir), approver=rails(emb), store_frames=True, seed=seed
    )
    assert log.status == "success", log.error
    return log


def check_sim(tmp: Path) -> None:
    emb = SOArmPairSim()

    # Teleop picks the cube through eval() and the CLI's guardrails; the sim scores it,
    # and the trial is written as a demo.
    pickers = [PickLeader(emb, side) for side in SIDES]
    task = pick_cube(num_scenes=1, max_seconds=20)
    log = run(task, TeleopPolicy(leaders=pickers), emb, tmp / "pick")
    assert log.results.metrics["success_at_end"] == 1.0, log.results.metrics
    [pick] = (tmp / "pick").glob("demos/*/*.json")
    demo = json.loads(pick.read_text(encoding="utf-8"))
    images = [w["image"] for w in demo["waypoints"] if "image" in w]
    final = sorted((tmp / "pick").glob("frames/*/*_top_*.npy"))[-1]  # after the last step
    assert pick.with_name(images[-1]).read_bytes() == encode_png(np.load(final))
    assert min(min(w["eef"][3], w["eef"][7]) for w in demo["waypoints"]) == 0.0  # closed

    # imitate, given the run's log dir (it skips the eval log there), pins the demo's world
    # at another eval seed, and its waypoints, read from the context as move_to targets,
    # redo the pick. Its frames are context images by absolute path.
    task = imitate(str(tmp / "pick"))
    [scene] = task.scenes
    fresh = imitate(str(tmp), same_world=False).scenes  # the pure check's demo, then pick's
    assert "world_seed" not in fresh[1].metadata and fresh[0].init_seed != fresh[1].init_seed
    assert "another layout" in fresh[1].metadata["context"][0]["text"]
    # A judged sim goal demo is no demo either (nothing the sim scores): left for the count.
    goal = {**demo, "termination_reason": "operator_end", "operator_judgement": "yes"}
    goal["scene"] = {**demo["scene"], "target": None}
    (tmp / "goal.json").write_text(json.dumps(goal), encoding="utf-8")
    pngs = [c["png"] for c in scene.metadata["context"] if "png" in c]
    assert len(pngs) == len(images) and all(Path(p).is_file() for p in pngs), pngs
    emb.reset(scene, seed=12345)
    assert np.array_equal(emb.world.data.body("cube").xpos, pickers[0].cube)
    lines = scene.metadata["context"][1]["text"].splitlines()
    moves = [json.loads(line[line.index("{") :]) for line in lines]
    log = run(task, Replay(emb, moves), emb, tmp / "imitate", seed=12345)
    assert log.results.metrics["success_at_end"] == 1.0, log.results.metrics

    # A free demo puts the cube down elsewhere and backs off (more waypoints than it keeps
    # frames). match-goal from it isn't met at the start, and replaying its waypoints meets
    # it; it refuses the pick demo, which ends mid-lift, and one that moved nothing.
    placers = [PickLeader(emb, side, place=True) for side in SIDES]
    task = pick_cube(num_scenes=1, max_seconds=10, free=True)
    run(task, TeleopPolicy(leaders=placers), emb, tmp / "free")
    [free] = (tmp / "free").glob("demos/*/*.json")
    (tmp / "twin").mkdir()  # another run's cube-0-e0: trial ids repeat across runs
    (tmp / "twin" / pick.name).write_bytes(pick.read_bytes())
    # Three: the pure check's demo (its yes), pick's and its twin; not the camera-less one
    # (its no), free's (no verdict) or the judged sim goal demo (nothing the sim scores).
    assert len(imitate(str(tmp)).scenes) == 3
    goal = match_goal(str(free), tol_m=0.03).scenes[0].target
    assert goal.spec["tol_m"] == 0.03  # -T tol_m reaches the scorer
    task = match_goal(str(free))
    [scene] = task.scenes
    emb.reset(scene, seed=777)
    assert not at_rest(emb.world, **scene.target.spec)
    # At rest where it sits; sliding through there at 0.5 m/s, it isn't.
    world = emb.world
    here = {"cube": world.objects()["cube"]}
    assert at_rest(world, here)
    world.data.qvel[world.model.body("cube").dofadr[0]] = 0.5
    mujoco.mj_forward(world.model, world.data)
    assert not at_rest(world, here)
    demo = json.loads(free.read_text(encoding="utf-8"))
    assert Path(scene.metadata["rubric"]).name == demo["waypoints"][-1]["image"]  # the end
    assert {"label": "Goal image", "png": scene.metadata["rubric"]} in scene.metadata["context"]
    assert sum("image" in w for w in demo["waypoints"]) == MAX_IMAGES < len(demo["waypoints"])
    moves = [dict(zip(demo["labels"], w["eef"])) for w in demo["waypoints"]]
    log = run(task, Replay(emb, moves), emb, tmp / "goal", seed=777)
    assert log.results.metrics["success_at_end"] == 1.0, log.results.metrics
    demo["objects"]["final"] = demo["objects"]["initial"]  # a demo that moved nothing
    still = free.with_name("still.json")
    still.write_text(json.dumps(demo), encoding="utf-8")
    for bad in (pick, still, tmp / "imitate", tmp / "typo.json"):  # an empty dir, a typo
        try:
            match_goal(str(bad))
            raise AssertionError(f"match-goal accepted {bad.name}")
        except ConfigError:
            pass
    # With a viewer open, sim time may not outrun the wall clock; a render-sized lag is
    # caught up, a stall re-anchors; and the viewer is shown each step once. A stand-in
    # window, so no GL.
    obs, synced = emb.reset(SCENE), []
    world._window = SimpleNamespace(sync=lambda: synced.append(1), close=lambda: None)
    world._t0 = time.perf_counter() - world.data.time
    wall = time.perf_counter()
    world.sleep(0.1)
    assert time.perf_counter() - wall >= 0.09
    for lag, anchors in ((0.05, False), (0.2, True)):
        world._t0 = t0 = time.perf_counter() - world.data.time - lag
        world.sleep(0.005)
        assert (world._t0 != t0) == anchors, lag
    emb.step(Action(data=obs.state["eef_pos"]))
    assert synced == [1], synced
    emb.close()

    # Facing each other: leaders at START give the targets eef_pos reports there (catch_m:
    # caught at once, so a wrong FK can't hide behind an uncaught arm's hold).
    facing = SOArmPairSim(right_base=(0.45, 0.0, 180.0))
    obs = facing.reset(SCENE)
    policy = teleop(facing.info, FakeLeader([START]), FakeLeader([START]), catch_m=1.0)
    act(policy, obs)
    assert np.allclose(act(policy, obs), obs.state["eef_pos"], rtol=0, atol=1e-3)
    facing.close()


def main() -> None:
    check_open_leader()
    with tempfile.TemporaryDirectory() as tmp:
        for wire in ("chat", "messages"):
            check_context_agent(Path(tmp), wire)
        check_teleop_pure(Path(tmp))
        check_sim(Path(tmp))
    print("ok")


if __name__ == "__main__":
    main()
