# lebenchmark

A benchmark for AI models controlling **two SO-101 robot arms**, scored against
human teleoperation on the same tasks. Built on
[inspect-robots](https://pypi.org/project/inspect-robots/) and
[inspect-robots-so101](https://github.com/robocurve/inspect-robots-so101).

**Status: in progress.** Simulation works end to end, and an LLM agent has completed
`pick-cube` in sim. The real two-arm rig is next.

## How it works

- **One control interface for humans and models.** Both arms take absolute
  `(x, y, z, gripper)` targets in a shared table frame (8-D for the pair). Each target
  is solved to joint angles with inverse kinematics (ikpy + least squares), with the
  gripper held as close to straight down as the wrist allows.
- **Safety guard.** Before each step, the arms' MuJoCo collision models are checked,
  and a step that would bring them closer than `min_gap_m` is held.
- **Simulation.** A MuJoCo scene of the real rig (MuJoCo Menagerie SO-101 model), with
  only the motor drivers swapped for simulated ones, so the sim runs the same code
  path as hardware.
- **Teleoperation and demos.** A person drives the arms with two leader arms through
  the same interface and guardrails, so a teleop run is a human baseline. Each trial
  is saved as a demo (waypoints, camera frames, object start and end poses).

## Tasks

| Task | What it asks |
|---|---|
| `pick-cube` | Pick up a cube placed where at least one arm reaches it |
| `place` | Put a cube on a green pad (both on one arm's side) |
| `handover` | Put a cube on a pad, where only the left arm reaches the cube and only the right arm reaches the pad |
| `stack` | Stack the red cube on the blue cube |
| `sort` | Put two red and two blue cubes on the pad of their color (pad sides change per seed) |
| `imitate` | Repeat a recorded demo, shown its moves and frames (same or new layout) |
| `match-goal` | Make the scene match the end of a recorded demo, from a goal image |

## Files

| File | Purpose |
|---|---|
| `so101_cartesian.py` | One arm with Cartesian targets (IK, joint step limits) |
| `so101_pair.py` | Two arms as one embodiment, with the arm-vs-arm collision guard |
| `so101_sim.py` | MuJoCo sim of the pair, scene setups and success checks |
| `so101_demo.py` | Leader-arm teleop policy, demo recording, demo-in-context agent |
| `so101_tasks.py` | Task definitions |
| `test_so101_*.py` | Self-checks (`python test_so101_scenes.py`, one at a time) |

## Run

```bash
pip install -e .                 # add [hardware] on Linux for the real arms
python test_so101_cartesian.py   # self-checks
inspect-robots run --task pick-cube --policy agent -P model=<model> \
  --embodiment so_arm_pair_sim --grader none
inspect-robots run --task pick-cube --policy teleop --embodiment so_arm_pair_sim \
  -E viewer=true -T free=true    # record a demo in sim
```

Model API keys go in `.env` (gitignored).
