"""Your tasks. Add one with ``@task``; importing this module registers them all.

A scene's ``setup`` and ``target`` name entries in ``so101_sim.SETUPS`` and ``TARGETS``,
which the sim realizes and checks; on the real rig you set the scene up and judge
success. Each task lists both scorers so it runs on both: ``success_at_end`` scores
sim runs, ``operator`` scores real ones; read the one that matches the run.

``imitate`` and ``match-goal`` are built from demos recorded with ``--policy teleop``
(``so101_demo``, whose import also registers ``teleop`` and ``context_agent``): pass
``-T demo=`` a demo's JSON or a directory searched for them, and run ``context_agent``.
"""

import math

from inspect_robots import Scene, Target, Task, is_affirmative_verdict, task
from inspect_robots.errors import ConfigError

from so101_demo import demo_context, load_demos

SCORERS = ["success_at_end", "operator"]


@task("pick-cube")
def pick_cube(num_scenes: int = 10, max_seconds: float = 120.0, free: bool = False) -> Task:
    """Pick up a cube placed where at least one arm reaches it. ``free``: no target, the
    operator ends the trial (to record a goal demo for ``match-goal``)."""
    return Task(
        name="pick-cube",
        scenes=[
            Scene(
                id=f"cube-{i}",
                instruction="Pick up the red cube and hold it above the table.",
                init_seed=i,
                setup="cube",
                target=None if free else Target("lifted", {"body": "cube", "height": 0.05}),
            )
            for i in range(num_scenes)
        ],
        scorer=SCORERS,
        max_seconds=max_seconds,
    )


def seeded(name: str, setup: str, instruction: str, target: Target, n: int, secs: float) -> Task:
    """``n`` scenes of one setup, each its own seeded layout."""
    return Task(
        name=name,
        scenes=[
            Scene(
                id=f"{setup}-{i}", instruction=instruction, init_seed=i, setup=setup, target=target
            )
            for i in range(n)
        ],
        scorer=SCORERS,
        max_seconds=secs,
    )


@task("place")
def place(num_scenes: int = 10, max_seconds: float = 120.0) -> Task:
    """Put a cube on a pad."""
    instruction = "Put the red cube on the green square and let go."
    target = Target("on_pad", {"placed": {"cube": "pad"}})
    return seeded("place", "place", instruction, target, num_scenes, max_seconds)


@task("handover")
def handover(num_scenes: int = 10, max_seconds: float = 180.0) -> Task:
    """Only the left arm reaches the cube and only the right arm reaches the pad, so the
    arms must pass the cube between them (hand to hand, or set down where both reach)."""
    instruction = (
        "Put the red cube on the green square and let go. Only the left arm can reach the "
        "cube and only the right arm can reach the square."
    )
    target = Target("on_pad", {"placed": {"cube": "pad"}})
    return seeded("handover", "handover", instruction, target, num_scenes, max_seconds)


@task("stack")
def stack(num_scenes: int = 10, max_seconds: float = 120.0) -> Task:
    """Stack one cube on another."""
    instruction = "Stack the red cube on top of the blue cube and let go."
    target = Target("stacked", {"top": "red", "bottom": "blue"})
    return seeded("stack", "stack", instruction, target, num_scenes, max_seconds)


@task("sort")
def sort(num_scenes: int = 10, max_seconds: float = 300.0) -> Task:
    """Sort four cubes onto the pad of their color."""
    instruction = (
        "Put each red cube on the pink square and each blue cube on the light blue square, "
        "side by side (not stacked), and let go."
    )
    placed = {"red_1": "red_pad", "red_2": "red_pad", "blue_1": "blue_pad", "blue_2": "blue_pad"}
    target = Target("on_pad", {"placed": placed})
    return seeded("sort", "sort", instruction, target, num_scenes, max_seconds)


@task("imitate")
def imitate(demo: str, same_world: bool = True, max_seconds: float = 120.0) -> Task:
    """Each successful demo's task, shown the demo's moves and frames.

    ponytail: ``same_world`` (the demo's own sim world) is a replay ceiling: the waypoints
    (to the mm) stop on the step the target first held, so a copy must carry the last one
    through; ``same_world=False`` (a fresh world per epoch) is the generalization test.
    """
    where = "in this same layout" if same_world else "in another layout (objects elsewhere)"
    intro = (
        f"A person did this task once, {where}. Their trial ended the moment the task was "
        "met, so their last move stops right at that point: finish it before calling done. "
        "Their moves, as move_to targets in the table frame, then what the first camera saw "
        "along the way:"
    )
    scenes = []
    for path, d in load_demos(demo):
        s = d["scene"]
        if not (
            d["termination_reason"] == "success" or is_affirmative_verdict(d["operator_judgement"])
        ):
            continue  # a try that missed its target is no demonstration
        if d["simulated"] and not s["target"]:
            continue  # a goal demo (pick-cube -T free=true): nothing the sim can score
        pin = {"world_seed": d["world_seed"]} if same_world and d["simulated"] else {}
        scenes.append(
            Scene(
                id=f"{path.parent.name}-{path.stem}",  # <run_stamp>-<trial>: trials repeat
                instruction=s["instruction"],
                init_seed=d["world_seed"],  # fresh worlds differ per demo, never its own
                setup=s["setup"],
                target=s["target"] and Target(s["target"]["kind"], s["target"]["spec"]),
                metadata={"context": demo_context(path, d, intro), **pin},
            )
        )
    if not scenes:
        raise ConfigError(f"no demo at {demo} met its target: imitate shows only successes")
    return Task(name="imitate", scenes=scenes, scorer=SCORERS, max_seconds=max_seconds)


@task("match-goal")
def match_goal(demo: str, tol_m: float = 0.02, max_seconds: float = 120.0) -> Task:
    """From each demo's start, make the scene look like the demo's end: in the sim, every
    object at rest within ``tol_m`` of where the demo left it."""
    scenes = []
    for path, d in load_demos(demo):
        if "image" not in d["waypoints"][-1]:
            raise ConfigError(
                f"{path}: no goal image (recorded without a camera); record goal demos with one"
            )
        goal_png = str(path.with_name(d["waypoints"][-1]["image"]))
        target, pin = None, {}
        if d["simulated"]:
            objects = d["objects"]
            moving = [b for b, v in objects["final_speed"].items() if v > 0.01]
            if moving:
                raise ConfigError(
                    f"{path}: {moving} still moving at the end (a pick that ends on success "
                    "is mid-lift, out of view); record goal demos with pick-cube -T "
                    "free=true: let go, move the arm away, then end the trial"
                )
            initial, final = objects["initial"], objects["final"]
            if all(math.dist(final[b], initial[b]) < tol_m for b in final):
                raise ConfigError(
                    f"{path}: no object moved {tol_m * 100:g} cm or more, so the goal holds "
                    "from the start; record a demo that moves one"
                )
            target = Target("at_rest", {"goal": final, "tol_m": tol_m})
            pin = {"world_seed": d["world_seed"]}
        context = [
            {"text": "The goal image shows how the scene should look at the end."},
            {"label": "Goal image", "png": goal_png},
        ]
        scenes.append(
            Scene(
                id=f"{path.parent.name}-{path.stem}",
                instruction="Move the objects so the scene matches the goal image.",
                setup=d["scene"]["setup"],
                target=target,
                # The operator's verdict prompt shows the rubric: the goal image's path.
                metadata={"context": context, "rubric": goal_png, **pin},
            )
        )
    return Task(name="match-goal", scenes=scenes, scorer=SCORERS, max_seconds=max_seconds)
