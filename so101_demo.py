"""Teleop and demos: a person drives the pair, and each trial becomes a demo for the LLM.

``teleop`` reads two SO-101 leader arms (LeRobot, on the rig; ``FakeLeader`` in tests)
and sends their fingertip points as the same 8-D table-frame targets as the LLM's
``move_to``, through the same CLI guardrails, so a teleop run scores like an LLM run: a
human baseline. Each trial is also written as a demo: the scene, the commanded path
simplified to waypoints, a few frames of the first camera, and in the sim where the
objects started and ended. ``so101_tasks`` builds tasks from demos (``imitate`` shows
one as text and images, ``match-goal`` asks for its end state), and ``context_agent``
is the LLM agent with that demo pinned into its first message. The demo's writer and
reader live here together because they share its format.

An arm follows its leader only once the leader's point comes within ``catch_m`` of the
arm's, so a leader left elsewhere doesn't drag its arm across the table. A trial ends on
the sim's target or ``max_seconds``; also on the Linux sim console's Esc, and on the
real rig on Enter at the left arm. Windows has no end key (Ctrl-C cancels the trial).
"""

import atexit
import base64
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from inspect_robots import Scene, policy
from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.errors import ConfigError, EmbodimentFault
from inspect_robots.frames import _safe  # ponytail: private, but the dependency is pinned
from inspect_robots.policy import PolicyBase, PolicyInfo
from inspect_robots.rollout import StepRecord, TrialRecord
from inspect_robots.spaces import Box
from inspect_robots.types import Action, ActionChunk, Observation
from inspect_robots_agent import LLMAgentPolicy, encode_png
from inspect_robots_so101 import STATE_KEY as JOINT_KEY
from inspect_robots_so101 import MOTORS, from_obs_dict, to_action_dict

from so101_cartesian import STATE_KEY
from so101_pair import SIDES, parse_bases, two_arm_spec

GRIP_M = 0.001  # simplifying the path, 1 gripper unit counts as 1 mm
TOL_M = 0.005  # the waypoints stay this close to the commanded path
MAX_IMAGES = 8  # frames per demo: every one is resent on every LLM call


def open_leader(port: str, leader_id: str) -> Any:
    """Connect one calibrated SO-101 leader arm (LeRobot, on the rig).

    Raises ``EmbodimentFault`` (halts the eval, like a follower that fails to open) rather
    than failing every trial in turn.
    """
    # Lazy: lerobot lives only on the rig (the project's `hardware` extra), and importing it
    # pulls in torch. Its file: $HF_LEROBOT_CALIBRATION/teleoperators/so_leader/<id>.json.
    try:
        from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

        # Keywords only: positionally the 2nd field is use_degrees. str(): -P makes "1" an int.
        leader = SOLeader(SOLeaderTeleopConfig(port=port, id=str(leader_id), use_degrees=True))
        leader.connect(calibrate=False)  # never LeRobot's interactive calibration mid-eval
        # Compares the file's ranges with the motors' EEPROM, so swapped ports fail here too.
        calibrated = leader.is_calibrated
    except Exception as exc:  # no lerobot (or LEROBOT_HOME set), wrong port, no servo power
        cause = f" ({exc.__cause__})" if exc.__cause__ else ""  # lerobot hides the errno there
        message = f"leader {leader_id!r} on {port}: {str(exc).strip()}{cause}"
        raise EmbodimentFault(message) from exc
    if not calibrated:
        leader.disconnect()
        raise EmbodimentFault(
            f"leader {leader_id!r} on {port} is not calibrated: no file matching its motors at "
            f"{leader.calibration_fpath}. First check that {port} is this leader, not the other "
            "one or a follower (ttyACM numbers reshuffle on replug; /dev/serial/by-id/ names "
            "don't). If it is, run `lerobot-calibrate --teleop.type=so101_leader "
            f"--teleop.port={port} --teleop.id={leader_id}` and type c at its first prompt "
            "(Enter would copy the file onto these motors)."
        )
    return leader


class FakeLeader:
    """Replays 6-D poses (5 joints in degrees, gripper 0..100) through LeRobot's leader
    interface; holds the last one."""

    def __init__(self, poses: list) -> None:
        self._poses, self._i = poses, 0

    def get_action(self) -> dict[str, float]:
        pose = self._poses[min(self._i, len(self._poses) - 1)]
        self._i += 1
        return to_action_dict(pose)

    def disconnect(self) -> None:
        pass


def keyframes(path: np.ndarray, tol: float = TOL_M) -> list[int]:
    """Indices Ramer-Douglas-Peucker keeps: the fewest points the path stays within
    ``tol`` of. ponytail: recursive, depth ~ the number of kept points; a jittery
    3000-step path at a tiny tol could hit the recursion limit, make it a stack then."""
    a, b = path[0], path[-1]
    ab = b - a
    t = np.clip((path - a) @ ab / max(ab @ ab, 1e-12), 0.0, 1.0)
    d = np.linalg.norm(path - a - t[:, None] * ab, axis=1)
    i = int(np.argmax(d))
    if d[i] <= tol:
        return [0, len(path) - 1]
    return keyframes(path[: i + 1], tol)[:-1] + [i + j for j in keyframes(path[i:], tol)]


def _frame(steps: list[StepRecord], j: int) -> np.ndarray:
    """The first camera's frame at path index ``j`` (0: before the first step, else after
    step ``j - 1``): streamed to disk with store_frames (the CLI default here), else kept
    in the record."""
    step, after = (steps[0], False) if j == 0 else (steps[j - 1], True)
    refs = step.result_image_refs if after else step.image_refs
    if refs:
        return next(iter(refs.values())).load()
    return next(iter((step.result.observation if after else step.observation).images.values()))


@policy("teleop")
class TeleopPolicy(PolicyBase):
    """Two leader arms drive the pair through the LLM's 8-D targets; each trial is a demo."""

    # Turns on the CLI's live console where there is one (Linux sim: Esc ends the trial);
    # the messages themselves are ignored.
    accepts_operator_messages = True

    def __init__(
        self,
        left_port: str | None = None,
        right_port: str | None = None,
        left_id: str = "leader_left",
        right_id: str = "leader_right",
        # Per leader, 5 joints: reading minus URDF angle (deg), as SOArmCartesian's
        # joint_offset_deg, so start at the follower's; keep the 5th (roll) at the
        # follower's: FK takes the follower's roll. Comma strings: -P passes scalars only.
        left_offset: str = "0,0,0,0,0",
        right_offset: str = "0,0,0,0,0",
        catch_m: float = 0.03,  # an arm follows once its leader's point is this close
        leaders: tuple[Any, Any] | None = None,  # Python only: two leader objects (tests)
    ) -> None:
        if leaders is None and None in (left_port, right_port):
            raise ConfigError("teleop needs the leaders' ports: -P left_port=... -P right_port=...")
        self._ports = ((left_port, left_id), (right_port, right_id))
        offsets = [str(o).split(",") for o in (left_offset, right_offset)]
        try:
            self._offset = np.array(offsets, dtype=np.float64).reshape(2, 5)
        except ValueError as exc:  # wrong count, separator or a non-number
            raise ConfigError(
                "-P left_offset/right_offset take 5 comma-separated degrees, e.g. 10,-5,3,0,0; "
                f"got {left_offset!r}, {right_offset!r}"
            ) from exc
        self._catch_m, self._leaders = catch_m, leaders
        self.info = PolicyInfo(name="teleop", action_space=Box(shape=(1,)))  # until bind()

    def bind(self, embodiment_info: EmbodimentInfo) -> None:
        """Adopt the pair's spaces, as the agent does, and its bases for the leaders' FK."""
        bases = parse_bases(embodiment_info.docs or "")  # ValueError: not the pair
        # The pair's own MJCF on those bases: its grasp sites are eef_pos's table frame.
        self._model = two_arm_spec(bases).compile()
        self._data = mujoco.MjData(self._model)
        self._qadr = [[self._model.joint(f"{s}_{m}").qposadr[0] for m in MOTORS[:5]] for s in SIDES]
        self._simulated, self._hz = embodiment_info.is_simulated, embodiment_info.control_hz
        self.info = PolicyInfo(
            name="teleop",
            action_space=embodiment_info.action_space,
            observation_space=embodiment_info.observation_space,
            control_hz=embodiment_info.control_hz,
        )

    def reset(self, scene: Scene) -> None:
        """Open the leaders on first use; start the trial with neither arm caught, and show
        the operator the scene's instruction and rubric. Runs before the embodiment resets,
        so there is no observation yet."""
        if self._leaders is None:
            self._leaders = [open_leader(port, leader_id) for port, leader_id in self._ports]
            atexit.register(lambda: [leader.disconnect() for leader in self._leaders])
        self._scene, self._caught, self._nag, self._last = scene, [False, False], 0.0, ""
        print(f"teleop: {scene.instruction}", file=sys.stderr)
        if scene.metadata.get("rubric"):  # match-goal's: the goal image's path
            print(f"  rubric: {scene.metadata['rubric']}", file=sys.stderr)
        print(
            "teleop: bring each leader to its arm's pose; the arm follows once its leader "
            f"is within {self._catch_m * 100:g} cm",
            file=sys.stderr,
        )

    def act(self, observation: Observation) -> ActionChunk:
        """Each caught arm's leader point and gripper; an uncaught arm holds where it is."""
        eef = np.asarray(observation.state[STATE_KEY], dtype=np.float64)
        rolls = np.asarray(observation.state[JOINT_KEY])[[4, 10]]
        grips = []
        for side, leader, adr, offset, roll in zip(
            SIDES, self._leaders, self._qadr, self._offset, rolls
        ):
            try:
                q = from_obs_dict(leader.get_action())
            except Exception as exc:  # unplugged or unpowered mid-run: halt, as open_leader does
                message = f"teleop: the {side} leader stopped answering: {exc}"
                raise EmbodimentFault(message) from exc
            # ponytail: the followers hold wrist roll, so FK takes the follower's measured
            # roll (minus the 5th offset, the follower's); the leader handle's twist has an
            # arbitrary zero and moves the point ~17 mm.
            self._data.qpos[adr] = np.radians(np.r_[q[:4], roll] - offset)
            grips.append(q[5])
        mujoco.mj_kinematics(self._model, self._data)
        target = np.concatenate(
            [[*self._data.site(f"{s}_gripperframe").xpos, g] for s, g in zip(SIDES, grips)]
        )
        # From last step's catches, so step 0 always holds: the delta limiter lets a
        # trial's first action through unlimited.
        action = np.where(np.repeat(self._caught, 4), target, eef)
        dist = [float(np.linalg.norm(target[i : i + 3] - eef[i : i + 3])) for i in (0, 4)]
        for i, side in enumerate(SIDES):
            if not self._caught[i] and dist[i] < self._catch_m:
                self._caught[i] = True
                print(f"teleop: {side} arm caught", file=sys.stderr)
        now = time.monotonic()
        if not all(self._caught) and now >= self._nag:
            self._nag = now + 1.0  # ponytail: a 1 Hz line; a status footer if it scrolls
            status = (
                f"{side} caught" if caught else f"{side} {d * 100:.0f} cm"
                for side, caught, d in zip(SIDES, self._caught, dist)
            )
            line = "teleop: " + ", ".join(status)
            if line != self._last:  # a parked leader (one-arm tasks) would repeat it every second
                print(line, file=sys.stderr)
                self._last = line
        return ActionChunk(actions=[Action(data=action)])

    def on_trial_end(self, record: TrialRecord, log_dir: str, run_stamp: str) -> None:
        """Write the trial as ``demos/<run_stamp>/<trial>.json``, its frames as PNGs beside
        it, and the JSON's path (relative to log_dir) in ``record.metadata["demo"]``."""
        steps = record.steps
        if not steps:  # errored before the first step
            return
        trial = f"{_safe(record.scene_id)}-e{record.epoch}"
        rel = Path("demos") / run_stamp / f"{trial}.json"
        out = Path(log_dir) / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        # The commanded path (eef_pos: what ran, after the guardrails), start included.
        path = np.array(
            [steps[0].observation.state[STATE_KEY]]
            + [s.result.observation.state[STATE_KEY] for s in steps]
        )
        kept = keyframes(path * ((1, 1, 1, GRIP_M) * 2))
        picks = np.linspace(0, len(kept) - 1, min(len(kept), MAX_IMAGES)).round().astype(int)
        # Evenly spaced, always the first and the last; none on a rig without cameras.
        shown = {kept[k] for k in picks} if self.info.observation_space.cameras else set()
        waypoints = []
        for j in kept:
            waypoint = {"t_s": round(j / self._hz, 3), "eef": path[j].round(4).tolist()}
            if j in shown:
                waypoint["image"] = f"{trial}-{j}.png"
                out.with_name(waypoint["image"]).write_bytes(encode_png(_frame(steps, j)))
            waypoints.append(waypoint)
        objects = None
        if "objects" in steps[-1].result.info:  # the sim's
            prev, last = steps[-2 if len(steps) > 1 else -1].result, steps[-1].result
            dt = max(last.observation.state_time - prev.observation.state_time, 1e-9)
            final = last.info["objects"]
            objects = {
                "initial": steps[0].result.info["objects"],
                "final": final,
                "final_speed": {
                    b: float(np.linalg.norm(np.subtract(xyz, prev.info["objects"][b]))) / dt
                    for b, xyz in final.items()
                },
            }
        scene, target = self._scene, self._scene.target
        demo = {
            "scene": {
                "id": scene.id,
                "instruction": scene.instruction,
                "setup": scene.setup,
                "target": target and {"kind": target.kind, "spec": dict(target.spec)},
            },
            # A demo recorded on a pinned scene (imitate, match-goal) keeps that world.
            "world_seed": scene.metadata.get("world_seed", record.seed),
            "simulated": self._simulated,
            "control_hz": self._hz,
            "status": record.status,
            "termination_reason": record.termination_reason,
            "operator_judgement": record.operator_judgement,
            "labels": list(self.info.action_space.semantics.dim_labels),
            "waypoints": waypoints,
            "objects": objects,
        }
        out.write_text(
            json.dumps(demo, indent=1, allow_nan=False), encoding="utf-8", newline="\n"
        )
        record.metadata["demo"] = rel.as_posix()  # like core's "actions", agent's "transcript"


def load_demos(demo: str) -> list[tuple[Path, dict[str, Any]]]:
    """The demo file ``demo``, or every demo ``*.json`` under that directory (so a run's
    log dir works: its eval log is skipped), by absolute path. Refuses finding none."""
    root = Path(demo).resolve()
    files = sorted(root.rglob("*.json")) if root.is_dir() else [root] if root.is_file() else []
    loaded = [(f, json.loads(f.read_text(encoding="utf-8"))) for f in files]
    demos = [(f, d) for f, d in loaded if "waypoints" in d]
    if not demos:
        raise ConfigError(f"no teleop demos at {root}: pass a demo .json or logs/demos/<run>")
    return demos


def demo_context(path: Path, demo: dict[str, Any], intro: str) -> list[dict[str, str]]:
    """``ContextAgent`` items for a demo: ``intro``, its waypoints as move_to targets in
    the table frame, then its frames, labeled."""
    waypoints = list(enumerate(demo["waypoints"], 1))
    moves = "\n".join(
        f"{n}. at {w['t_s']:.1f} s: move_to "
        + json.dumps({k: round(v, 3) for k, v in zip(demo["labels"], w["eef"])})
        for n, w in waypoints
    )
    # Labels must not start with "camera '": the agent's on_demand mode parses that prefix.
    images = [
        {"label": f"Demo frame at move {n}", "png": str(path.with_name(w["image"]))}
        for n, w in waypoints
        if "image" in w
    ]
    return [{"text": intro}, {"text": moves}, *images]


@policy("context_agent")
class ContextAgent(LLMAgentPolicy):
    """The ``agent`` policy with ``scene.metadata["context"]`` pinned after the goal in the
    first user message of every LLM call: ``{"text": ...}`` items as text, ``{"label": ...,
    "png": path}`` items as a label line and the image. Without context it is ``agent``.

    ponytail: rides on inspect-robots-agent 0.28 internals (``_client``, ``_messages``, and
    image_horizon stubbing the oldest image message first); re-check them on upgrade. The
    pinned images are resent in full on every call.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        complete = self._client.complete

        def pinned(messages: list[dict[str, Any]], *args: Any, **kw: Any) -> Any:
            # image_horizon stubs the oldest image messages, the pinned one first: put its
            # content back (keeping a cache_anchor). Stateful wires get the list itself.
            if messages[1] is not self._messages[1]:
                restored = {**messages[1], "content": self._messages[1]["content"]}
                messages = [messages[0], restored, *messages[2:]]
            return complete(messages, *args, **kw)

        self._client.complete = pinned

    def bind(self, embodiment_info: EmbodimentInfo) -> None:
        """Bind as the agent does, but log as ``context_agent`` (eval logs info.name)."""
        super().bind(embodiment_info)
        self.info = dataclasses.replace(self.info, name="context_agent")

    def reset(self, scene: Scene) -> None:
        """Start the trial as the agent does, then pin the scene's context after the goal."""
        super().reset(scene)
        context = scene.metadata.get("context")
        if not context:
            return
        parts = [{"type": "text", "text": self._messages[1]["content"]}]
        for item in context:
            if "png" in item:
                data = base64.b64encode(Path(item["png"]).read_bytes()).decode("ascii")
                parts += [
                    {"type": "text", "text": f"{item['label']}:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
                ]
            else:
                parts.append({"type": "text", "text": item["text"]})
        self._messages[1] = {"role": "user", "content": parts}
