"""End-to-end MuJoCo smokes: plan → DomainBridge → PickPlaceExecutor (Tier 1–3)."""

from pathlib import Path

import pytest

from workbenchmark_pddl.bridge import execute_plan_on_bridge, make_lego_bridge
from workbenchmark_pddl.frames import TableFrame
from workbenchmark_pddl.planner import plan_task
from workbenchmark_pddl.scene import build_lego_env, place_initial_blocks
from workbenchmark_pddl.task import load_task

# Run the full plan in the sim.
# Should solve the plan and execute it on the environment.
def _run_full_plan_in_sim(task_path: Path, tmp_path: Path) -> None:

    # Import the mujoco and tampanda modules.
    pytest.importorskip("mujoco")
    pytest.importorskip("tampanda")

    from tampanda.planners.grasp_planner import GraspPlanner
    from tampanda.planners.pick_place import PickPlaceExecutor
    from tampanda.planners.rrt_star import RRTStar

    # Load the task.
    task = load_task(task_path)

    # Create the table frame.
    frame = TableFrame()

    # Build the environment.
    env = build_lego_env(task, template_dir=tmp_path)

    # Place the initial bricks.
    place_initial_blocks(env, task, frame=frame)

    # Plan the task.
    plan = plan_task(task)

    # Assert the plan is solved.
    assert plan.solved, f"planner failed: {plan.status}"

    # Create the planner.
    planner = RRTStar(env)

    # Create the grasp planner.
    grasp = GraspPlanner(table_z=frame.surface_z)

    # Create the executor.
    executor = PickPlaceExecutor(
        env, planner, grasp, use_attachment=True, max_plan_iters=5000
    )

    # Create the bridge.
    bridge = make_lego_bridge(env, task, executor=executor, frame=frame)

    # Execute the plan on the bridge.
    assert execute_plan_on_bridge(bridge, plan, return_home=True)

# Test the full plan in the sim for each tier.
# Should solve the plan and execute it on the environment.
@pytest.mark.slow
@pytest.mark.parametrize(
    "task_fixture",
    ["tier1_task_001", "tier2_task_001", "tier3_task_001"],
)
def test_full_plan_in_sim(task_fixture: str, request: pytest.FixtureRequest, tmp_path: Path):

    # Get the task path.
    task_path = request.getfixturevalue(task_fixture)

    # Create the run directory.
    run_dir = tmp_path / task_fixture

    # Create the run directory.
    run_dir.mkdir()

    # Run the full plan in the sim.
    _run_full_plan_in_sim(task_path, run_dir)
