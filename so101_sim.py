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


def cube(spec: mujoco.MjSpec, rng: np.random.Generator) -> None:
    """A 3 cm red cube on the table where at least one arm reaches it."""
    body = spec.worldbody.add_body(
        name="cube", pos=[rng.uniform(0.20, 0.27), rng.uniform(-0.15, 0.15), 0.015]
    )
    body.add_freejoint()
    body.add_geom(  # contact settings from Menagerie's scene_box.xml
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.015] * 3,
        rgba=[0.85, 0.1, 0.1, 1],
        friction=[1, 0.03, 0.003],
        solref=[0.01, 1],
    )


def lifted(world: "World", body: str = "cube", height: float = 0.05) -> bool:
    """The body's center is at least ``height`` m above the table."""
    return bool(world.data.body(body).xpos[2] >= height)


def at_rest(
    world: "World", goal: dict[str, list[float]], tol_m: float = 0.02, max_speed: float = 0.01
) -> bool:
    """Every ``goal`` body is within ``tol_m`` of its ``[x, y, z]`` and slower than
    ``max_speed`` m/s, so one falling or sliding through its goal doesn't count."""
    # ponytail: mj_objectVelocity after mj_step is one timestep (5 ms) old; fine for "at rest"
    vel = np.zeros(6)  # (angular, linear), world-aligned
    for body, xyz in goal.items():
        b = world.data.body(body)
        mujoco.mj_objectVelocity(world.model, world.data, mujoco.mjtObj.mjOBJ_BODY, b.id, vel, 0)
        if np.linalg.norm(b.xpos - xyz) > tol_m or np.linalg.norm(vel[3:]) >= max_speed:
            return False
    return True


SETUPS: dict[str, Callable[[mujoco.MjSpec, np.random.Generator], None]] = {"cube": cube}
TARGETS: dict[str, Callable[..., bool]] = {"lifted": lifted, "at_rest": at_rest}


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
