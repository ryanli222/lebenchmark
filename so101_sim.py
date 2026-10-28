"""MuJoCo sim of the two-arm rig: the real ``SOArmPair`` stack with simulated drivers.

Only the bottom layer differs from hardware. Each arm's ``SOArmEmbodiment`` gets a
``SimArm`` driver (LeRobot's motor dict: degrees, gripper 0..100) plus the world's
clock and sleep, so pacing runs on sim time. A scene's ``setup`` names a function in
``SETUPS`` (adds objects, seeded) and its ``target`` a check in ``TARGETS``; add your
own there. A met target ends the trial with reason "success" (``success_at_end``).
Object poses never reach the observation: as on the real rig, the policy sees only
cameras and joints. Step ``info["objects"]`` carries them for the trial record, where
only the demo writer reads them. A scene's ``metadata["world_seed"]`` pins its world (a
demo's) whatever the eval seed. ``viewer=True`` (CLI ``-E viewer=true``) opens MuJoCo's
viewer for a human teleoperating the sim, and paces the sim to the wall clock.
"""

import dataclasses
import itertools
import time
from collections.abc import Callable
from typing import Any

import mujoco
import numpy as np
from inspect_robots.embodiment import PRIVILEGED_SUCCESS, RESETTABLE, SEEDABLE
from inspect_robots.scene import Scene
from inspect_robots.types import Action, Observation, StepResult
from inspect_robots_so101 import MOTORS, SOArmConfig, SOArmEmbodiment, from_obs_dict, to_action_dict

from so101_cartesian import SOArmCartesian
from so101_pair import SIDES, SilentOperator, SOArmPair, gripper_angle, two_arm_spec

# Grasp point at (0.17, 0, 0.08) in the arm's frame, gripper down and half open: just
# inside the workspace, so the first move doesn't jump, and above any cube.
START = (0.0, -37.0, 32.4, 94.5, 0.0, 50.0)
CAMERAS = (("top", "left_wrist_cam"), ("right_wrist_cam",))  # per arm; MJCF camera names
# Matches World.reset's `top` camera: 0.55 m up, fovy 60 deg, 480 px tall -> 1.32 mm/px.
CAMERA_DOCS = """
Camera `top` looks straight down at the table from 0.55 m: image up is +x, image left
is +y, the image centre is the table point x=0.14, y=0, and one pixel spans about
1.3 mm of table (things above the table look slightly bigger and further from the
centre). `left_wrist_cam` and `right_wrist_cam` ride on the grippers."""


RED, BLUE, GREEN = [0.85, 0.1, 0.1, 1], [0.1, 0.25, 0.85, 1], [0.2, 0.7, 0.3, 1]
PAD_HALF = 0.04  # pads are 8 cm squares: room for two cubes
# Across the table (y, table frame) the left arm reaches only above -0.05 and the right
# only below 0.05. A cube at y >= 0.12 is the left's alone and a pad whose near edge is at
# -0.11 the right's alone, each with a cube's width or more to spare, so a nudge or a slide
# can't finish a handover with one arm. One arm's side (ONE_ARM) holds a whole scene for
# that arm. Every spot below is inside x 0.16-0.30, both workspaces.
LEFT_ONLY, RIGHT_ONLY = (0.12, 0.22), (-0.24, -0.15)
ONE_ARM = ((-0.03, 0.15), (-0.15, 0.03))  # left's side, right's side


def add_cube(spec: mujoco.MjSpec, name: str, xy, rgba: list[float]) -> None:
    """A 3 cm cube resting on the table at ``xy``."""
    body = spec.worldbody.add_body(name=name, pos=[*xy, 0.015])
    body.add_freejoint()
    body.add_geom(  # contact settings from Menagerie's scene_box.xml
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.015] * 3,
        rgba=rgba,
        friction=[1, 0.03, 0.003],
        solref=[0.01, 1],
    )


def add_pad(spec: mujoco.MjSpec, name: str, xy, rgba: list[float]) -> None:
    """A flat colored square on the table; nothing collides with it, it only marks a spot."""
    spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[*xy, 0.0005],
        size=[PAD_HALF, PAD_HALF, 0.0005],
        rgba=rgba,
        contype=0,
        conaffinity=0,
    )


def spots(rng: np.random.Generator, n: int, gap: float, x=(0.20, 0.27), y=(-0.15, 0.15)):
    """``n`` random table points in the box, at least ``gap`` m apart."""
    # ponytail: rejection sampling; sort's 4 cubes take ~10 tries, the rest one or two
    while True:
        pts = rng.uniform((x[0], y[0]), (x[1], y[1]), size=(n, 2))
        if all(np.linalg.norm(a - b) >= gap for a, b in itertools.combinations(pts, 2)):
            return pts


def cube(spec: mujoco.MjSpec, rng: np.random.Generator) -> None:
    """A 3 cm red cube on the table where at least one arm reaches it."""
    # Two scalar draws, as before: recorded demos pin their world by seed.
    add_cube(spec, "cube", (rng.uniform(0.20, 0.27), rng.uniform(-0.15, 0.15)), RED)


def place(spec: mujoco.MjSpec, rng: np.random.Generator) -> None:
    """A red cube and a green pad at least 10 cm apart, both on one arm's side."""
    c, p = spots(rng, 2, 0.10, y=ONE_ARM[rng.integers(2)])
    add_cube(spec, "cube", c, RED)
    add_pad(spec, "pad", p, GREEN)


def handover(spec: mujoco.MjSpec, rng: np.random.Generator) -> None:
    """A red cube only the left arm reaches and a green pad only the right arm reaches."""
    add_cube(spec, "cube", rng.uniform((0.20, LEFT_ONLY[0]), (0.27, LEFT_ONLY[1])), RED)
    add_pad(spec, "pad", rng.uniform((0.20, RIGHT_ONLY[0]), (0.27, RIGHT_ONLY[1])), GREEN)


def stack(spec: mujoco.MjSpec, rng: np.random.Generator) -> None:
    """A red and a blue cube at least 6 cm apart, both on one arm's side."""
    r, b = spots(rng, 2, 0.06, y=ONE_ARM[rng.integers(2)])
    add_cube(spec, "red", r, RED)
    add_cube(spec, "blue", b, BLUE)


def sort(spec: mujoco.MjSpec, rng: np.random.Generator) -> None:
    """Two red and two blue cubes in the middle, a red and a blue pad at the sides. Which
    side is which is seeded, so a policy can't memorize it."""
    sides = [(0.24, 0.17), (0.24, -0.17)]
    rng.shuffle(sides)
    add_pad(spec, "red_pad", sides[0], [0.95, 0.6, 0.6, 1])
    add_pad(spec, "blue_pad", sides[1], [0.6, 0.7, 0.95, 1])
    pts = spots(rng, 4, 0.05, x=(0.19, 0.29), y=(-0.09, 0.09))
    for name, rgba, xy in zip(("red_1", "red_2", "blue_1", "blue_2"), (RED, RED, BLUE, BLUE), pts):
        add_cube(spec, name, xy, rgba)


def speed(world: "World", body: str) -> float:
    """The body's linear speed, m/s."""
    # ponytail: mj_objectVelocity after mj_step is one timestep (5 ms) old; fine for "at rest"
    vel = np.zeros(6)  # (angular, linear), world-aligned
    b = world.data.body(body).id
    mujoco.mj_objectVelocity(world.model, world.data, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
    return float(np.linalg.norm(vel[3:]))


def lifted(world: "World", body: str = "cube", height: float = 0.05) -> bool:
    """The body's center is at least ``height`` m above the table."""
    return bool(world.data.body(body).xpos[2] >= height)


def at_rest(
    world: "World", goal: dict[str, list[float]], tol_m: float = 0.02, max_speed: float = 0.01
) -> bool:
    """Every ``goal`` body is within ``tol_m`` of its ``[x, y, z]`` and slower than
    ``max_speed`` m/s, so one falling or sliding through its goal doesn't count."""
    return all(
        np.linalg.norm(world.data.body(b).xpos - xyz) <= tol_m and speed(world, b) < max_speed
        for b, xyz in goal.items()
    )


def on_pad(world: "World", placed: dict[str, str], max_speed: float = 0.01) -> bool:
    """Every ``placed`` body rests on the table with its center over its pad: not held
    above it, not sliding across it."""
    return all(
        np.abs(world.data.body(b).xpos[:2] - world.data.geom(pad).xpos[:2]).max() <= PAD_HALF
        and world.data.body(b).xpos[2] <= 0.02  # a resting cube's center is at 0.015
        and speed(world, b) < max_speed
        for b, pad in placed.items()
    )


def stacked(world: "World", top: str, bottom: str, tol_m: float = 0.015) -> bool:
    """``top`` rests on ``bottom``: centers within ``tol_m`` across and one cube (3 cm)
    apart up, ``bottom`` on the table, both still."""
    t, b = world.data.body(top).xpos, world.data.body(bottom).xpos
    return bool(
        np.linalg.norm(t[:2] - b[:2]) <= tol_m
        and abs(t[2] - b[2] - 0.03) <= 0.005
        and b[2] <= 0.02
        and speed(world, top) < 0.01
        and speed(world, bottom) < 0.01
    )


SETUPS: dict[str, Callable[[mujoco.MjSpec, np.random.Generator], None]] = {
    "cube": cube,
    "place": place,
    "handover": handover,
    "stack": stack,
    "sort": sort,
}
TARGETS: dict[str, Callable[..., bool]] = {
    "lifted": lifted,
    "at_rest": at_rest,
    "on_pad": on_pad,
    "stacked": stacked,
}


class World:
    """The MuJoCo scene both drivers share; rebuilt each reset, advanced by ``sleep``."""

    def __init__(self, bases: np.ndarray, viewer: bool = False) -> None:
        self._bases, self._viewer = bases, viewer
        self._renderer: mujoco.Renderer | None = None
        self._window = None  # the viewer's handle while it is open

    def reset(self, setup: str | None, seed: int | None) -> None:
        """Rebuild the scene with ``setup``'s objects and both arms at ``START``."""
        spec = two_arm_spec(self._bases)
        spec.worldbody.add_geom(  # the table top, z=0
            type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], rgba=[0.55, 0.5, 0.45, 1]
        )
        spec.worldbody.add_light(
            pos=[0, 0, 1.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        )
        # Straight down over the workspace; image up is +x, image left is +y.
        spec.worldbody.add_camera(name="top", pos=[0.14, 0, 0.55], quat=[1, 0, 0, -1], fovy=60)
        if setup is not None:
            SETUPS[setup](spec, np.random.default_rng(seed))
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.qadr = {s: [self.model.joint(f"{s}_{m}").qposadr[0] for m in MOTORS] for s in SIDES}
        self.act = {s: [self.model.actuator(f"{s}_{m}").id for m in MOTORS] for s in SIDES}
        for side in SIDES:
            q = [*np.radians(START[:5]), gripper_angle(self.model, side, START[5])]
            self.data.qpos[self.qadr[side]] = self.data.ctrl[self.act[side]] = q
        mujoco.mj_forward(self.model, self.data)
        self.close()
        self._renderer = mujoco.Renderer(self.model, SOArmConfig.cam_height, SOArmConfig.cam_width)
        self._debt = 0.0
        # ponytail: a new window per reset (the model changes with the setup), ~350 MB; its
        # mouse perturbations can reach the sim, so a viewer run is a practice-grade baseline.
        if self._viewer:
            from mujoco import viewer  # lazy; `import mujoco.viewer` here would shadow mujoco

            self._window = viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False
            )
            self._t0 = time.perf_counter()

    def objects(self) -> dict[str, list[float]]:
        """Each free body's ``[x, y, z]`` (table frame), as plain floats."""
        free = self.model.jnt_bodyid[self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE]
        return {self.model.body(b).name: self.data.xpos[b].tolist() for b in free}

    def render(self, camera: str) -> np.ndarray:
        """Return one camera's RGB frame."""
        assert self._renderer is not None
        self._renderer.update_scene(self.data, camera)
        return self._renderer.render()

    def sleep(self, seconds: float) -> None:
        """Advance physics by ``seconds``; the remainder under one timestep carries over."""
        dt = self.model.opt.timestep
        self._debt += seconds
        n = int(self._debt / dt)
        self._debt -= n * dt
        mujoco.mj_step(self.model, self.data, n)
        if self._window is not None:  # a human is watching: sim time may not outrun wall time
            ahead = self.data.time - (time.perf_counter() - self._t0)
            # ponytail: a stall (an LLM call, a slow step) re-anchors; smaller lags (renders,
            # viewer sync) are caught up, so a burst stays under 0.1 s.
            if ahead > 0:
                time.sleep(ahead)
            elif ahead < -0.1:
                self._t0 = time.perf_counter() - self.data.time

    def sync(self) -> None:
        """Show the current state in the viewer, if open. Once per step, not per ``sleep``:
        each call waits for the viewer's frame (~7 ms)."""
        if self._window is not None:
            self._window.sync()

    def close(self) -> None:
        """Release the renderer's GL context and close the viewer."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._window is not None:
            self._window.close()
            self._window = None


class SimArm:
    """One arm of the ``World`` behind LeRobot's driver interface."""

    def __init__(self, world: World, side: str, cameras: tuple[str, ...]) -> None:
        self._world, self._side, self._cameras = world, side, cameras

    def get_observation(self) -> dict[str, Any]:
        """Return motor positions (deg, gripper 0..100) and camera frames."""
        w = self._world
        q = w.data.qpos[w.qadr[self._side]]
        jaw = np.interp(q[5], w.model.joint(f"{self._side}_gripper").range, (0.0, 100.0))
        frames = {cam: w.render(cam) for cam in self._cameras}
        return {**to_action_dict([*np.degrees(q[:5]), jaw]), **frames}

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Set the servo targets; returns the action, as LeRobot does."""
        cmd, w = from_obs_dict(action), self._world
        jaw = gripper_angle(w.model, self._side, cmd[5])
        w.data.ctrl[w.act[self._side]] = [*np.radians(cmd[:5]), jaw]
        return dict(action)

    def disconnect(self) -> None:
        pass


class SOArmPairSim(SOArmPair):
    """``SOArmPair`` on a MuJoCo ``World``: seeded setups, success from the scene target."""

    def __init__(self, *, viewer: bool = False, **kw: Any) -> None:  # kw: SOArmPair's bases, gap
        arms = [
            SOArmCartesian(
                SOArmEmbodiment(
                    # A blocked arm times out every step: keep that short, and keep settling.
                    SOArmConfig(
                        cameras=cameras,
                        settle_tolerance=0.5,
                        settle_timeout_s=0.25,
                        settle_timeout_budget=10**9,
                    ),
                    driver_factory=lambda _cfg, s=side, c=cameras: SimArm(self.world, s, c),
                    operator=SilentOperator(),
                    poll_end=lambda: False,
                    sleep_fn=lambda s: self.world.sleep(s),
                    clock=lambda: self.world.data.time,
                )
            )
            for side, cameras in zip(SIDES, CAMERAS)
        ]
        super().__init__(*arms, **kw)
        self.world = World(self._bases, viewer)
        self.info = dataclasses.replace(
            self.info,
            name="so_arm_pair_sim",
            docs=self.info.docs + CAMERA_DOCS,
            is_simulated=True,
            capabilities=frozenset({SEEDABLE, RESETTABLE, PRIVILEGED_SUCCESS}),
            supported_setups=frozenset(SETUPS),
            supported_target_kinds=frozenset(TARGETS),
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Rebuild the world for the scene, then reset both arms in it."""
        seed = scene.metadata.get("world_seed", scene.init_seed if seed is None else seed)
        self.world.reset(scene.setup, seed)
        self._target = scene.target
        return super().reset(scene, seed=seed)

    def step(self, action: Action) -> StepResult:
        """Step both arms; end with "success" once the scene's target holds."""
        result = super().step(action)
        self.world.sync()
        result = dataclasses.replace(result, info={**result.info, "objects": self.world.objects()})
        if self._target and TARGETS[self._target.kind](self.world, **self._target.spec):
            return dataclasses.replace(result, terminated=True, termination_reason="success")
        return result

    def close(self) -> None:
        """Release both arms, the renderer and the viewer."""
        try:
            super().close()
        finally:
            self.world.close()
