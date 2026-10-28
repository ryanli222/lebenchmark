"""Self-check for the scene setups and targets: run ``python test_so101_scenes.py``.

Every task's scenes build, start unsolved, and score once their objects are put where
the target wants them (moved by hand in the sim, no arms involved).
"""

import mujoco
import numpy as np

from so101_pair import LEFT_BASE, RIGHT_BASE
from so101_sim import LEFT_ONLY, PAD_HALF, TARGETS, World
from so101_tasks import handover, pick_cube, place, sort, stack


def put(world: World, body: str, xyz) -> None:
    """Teleport a free body to ``xyz``, upright and still."""
    b = world.model.body(body)
    adr = world.model.joint(b.jntadr[0]).qposadr[0]
    world.data.qpos[adr : adr + 7] = [*xyz, 1, 0, 0, 0]
    world.data.qvel[b.dofadr[0] : b.dofadr[0] + 6] = 0
    mujoco.mj_forward(world.model, world.data)  # xpos follows qpos only after this


def met(world: World, target) -> bool:
    return TARGETS[target.kind](world, **target.spec)


def main() -> None:
    world = World(np.array([LEFT_BASE, RIGHT_BASE]))
    try:
        for task in (pick_cube(5), place(5), handover(5), stack(5), sort(5)):
            for scene in task.scenes:
                world.reset(scene.setup, scene.init_seed)
                world.sleep(1.0)
                assert not met(world, scene.target), (task.name, scene.id)
                objs = world.objects()

                if scene.setup == "handover":  # neither arm can do it alone
                    # Each arm's workspace spans y +-0.20 around its base (y = +-0.15).
                    assert objs["cube"][1] >= LEFT_ONLY[0] > RIGHT_BASE[1] + 0.20, objs
                    pad_y = world.data.geom("pad").xpos[1]
                    assert pad_y + PAD_HALF < LEFT_BASE[1] - 0.20, pad_y

                if scene.setup in ("place", "stack"):  # one arm reaches everything
                    ys = [p[1] for p in objs.values()]
                    if scene.setup == "place":
                        ys.append(world.data.geom("pad").xpos[1])
                    left, right = LEFT_BASE[1] - 0.20, RIGHT_BASE[1] + 0.20
                    assert min(ys) >= left or max(ys) <= right, (scene.id, ys)

                if scene.target.kind == "on_pad":
                    placed = scene.target.spec["placed"]
                    for i, (body, pad) in enumerate(placed.items()):
                        x, y, _ = world.data.geom(pad).xpos
                        put(world, body, (x, y + (0.02 if i % 2 else -0.02), 0.06))
                    assert not met(world, scene.target), "held above the pad"
                    world.sleep(1.0)
                    assert met(world, scene.target), (task.name, scene.id, world.objects())
                    first = world.model.body(next(iter(placed)))
                    world.data.qvel[first.dofadr[0]] = 0.5  # sliding across the pad
                    mujoco.mj_forward(world.model, world.data)
                    assert not met(world, scene.target), "sliding is not placed"
                elif scene.target.kind == "stacked":
                    bx, by, _ = objs["blue"]
                    put(world, "red", (bx, by, 0.08))  # still, but held above, not resting
                    assert not met(world, scene.target), "hovering is not stacked"
                    put(world, "red", (bx, by, 0.05))
                    world.sleep(1.0)
                    assert met(world, scene.target), (task.name, scene.id, world.objects())
                    put(world, "red", (bx + 0.05, by, 0.015))
                    world.sleep(1.0)
                    assert not met(world, scene.target), "side by side is not stacked"
                else:  # lifted
                    x, y, _ = objs["cube"]
                    put(world, "cube", (x, y, 0.1))
                    assert met(world, scene.target)

        # The sort pads swap sides between seeds, so no layout can be memorized.
        sides = set()
        for scene in sort(10).scenes:
            world.reset(scene.setup, scene.init_seed)
            sides.add(world.data.geom("red_pad").xpos[1] > 0)
        assert sides == {True, False}, sides

        # pick-cube's worlds are unchanged by the refactor: recorded demos pin them by seed.
        world.reset("cube", 0)
        rng = np.random.default_rng(0)
        xy = rng.uniform(0.20, 0.27), rng.uniform(-0.15, 0.15)
        assert np.allclose(world.objects()["cube"][:2], xy)
    finally:
        world.close()
    print("ok")


if __name__ == "__main__":
    main()
