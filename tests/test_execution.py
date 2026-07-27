"""Tests for plan → YAML pose grounding (no MuJoCo)."""

from pathlib import Path

import pytest

from workbenchmark_pddl.execution import (
    GroundedAction,
    brick_for_pick,
    brick_for_place,
    ground_plan,
    ground_plan_action,
    half_size_for,
    yaml_brick_name,
)
from workbenchmark_pddl.planner import PlanAction, PlanResult, plan_task
from workbenchmark_pddl.task import load_task

# Test the yaml_brick_name function.
# Should strip the "b_" prefix from the PDDL object name.
# Should raise an error if the PDDL object name does not start with "b_".
def test_yaml_brick_name_strips_prefix():

    # Test the yaml_brick_name function with a PDDL object name that starts with "b_".
    assert yaml_brick_name("b_4x2_brick_1") == "4x2_brick_1"

    # Test the yaml_brick_name function with a PDDL object name that does not start with "b_".
    with pytest.raises(ValueError):
        yaml_brick_name("4x2_brick_1")


# Test the brick_for_pick and brick_for_place functions.
# Should return the brick with the given name from the initial_blocks and goal_blocks respectively.
def test_pose_lookup_pick_vs_place(tier1_task_001: Path):

    # Load the task.
    task = load_task(tier1_task_001)

    # Look up the brick in the initial_blocks.
    pick = brick_for_pick(task, "4x2_brick_1")

    # Look up the brick in the goal blocks.
    place = brick_for_place(task, "4x2_brick_1")

    # Same brick, different regions: pick-area vs assembly goal.
    assert pick.pos != place.pos
    assert pick.pos == task.initial_blocks[0].pos
    assert place.pos == task.goal_blocks[0].pos


# Test the half_size_for function.
# Should return the half-size of the brick in meters.
def test_half_size_4x2():
    from workbenchmark_pddl.task import BrickSpec

    # Create a brick.
    brick = BrickSpec(
        name="x",
        type="brick_4x2",
        color="red",
        pos=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
    )

    # Calculate the half-size of the brick.
    hx, hy, hz = half_size_for(brick)

    # Assert the half-size of the brick is approximately 0.032 meters.
    assert hx == pytest.approx(0.032)

    # Assert the half-size of the brick is approximately 0.016 meters.
    assert hy == pytest.approx(0.016)

    # Assert the half-size of the brick is approximately 0.0191 meters.
    assert hz == pytest.approx(0.0191 / 2)


# Test the ground_plan_action function.
# Should return the grounded action for the given plan action.
def test_ground_pick_and_stack_actions(tier1_task_001: Path):

    # Load the task.
    task = load_task(tier1_task_001)

    # Ground the pick action.
    pick = ground_plan_action(
        task,
        PlanAction(
            name="pick",
            params=(
                "gripper",
                "b_4x2_brick_1",
                "stud_0_0",
                "layer_0",
                "rot_0",
                "pick_area",
                "type_4x2",
            ),
        ),
    )
    assert pick.kind == "pick"
    assert pick.brick == "4x2_brick_1"
    assert pick.pos == task.initial_blocks[0].pos

    # Ground the stack action.
    stack = ground_plan_action(
        task,
        PlanAction(
            name="stack",
            params=(
                "gripper",
                "b_4x2_brick_2",
                "b_4x2_brick_1",
                "stud_0_0",
                "stud_0_0",
                "layer_1",
                "layer_0",
                "rot_0",
                "rot_0",
                "assembly_area",
                "type_4x2",
                "type_4x2",
            ),
        ),
    )

    # Assert the stack action is a stack action.
    assert stack.kind == "stack"

    # Assert the stack action is the correct brick.
    assert stack.brick == "4x2_brick_2"

    # Assert the stack action is the correct support brick.
    assert stack.support_brick == "4x2_brick_1"

    # Assert the stack action is the correct position.
    assert stack.pos == task.goal_blocks[1].pos


# Test the ground_plan function.
# Should return the grounded actions for the given plan.
def test_ground_full_tier1_plan(tier1_task_001: Path):

    # Load the task.
    task = load_task(tier1_task_001)

    # Plan the task.
    plan = plan_task(task)

    # Assert the plan is solved.
    assert plan.solved

    # Ground the plan.
    grounded = ground_plan(task, plan)

    # Assert the number of grounded actions is the same as the number of plan actions.
    assert len(grounded) == len(plan.actions)
    assert all(isinstance(step, GroundedAction) for step in grounded)

    # First action picks from initial pose; last stacks at goal pose.
    assert grounded[0].kind == "pick"
    assert grounded[0].pos == brick_for_pick(task, grounded[0].brick).pos
    assert grounded[-1].kind == "stack"
    assert grounded[-1].pos == brick_for_place(task, grounded[-1].brick).pos


# Test the ground_plan function.
# Should raise an error if the plan is not solved.
def test_ground_unsolved_plan_raises(tier1_task_001: Path):

    # Load the task.
    task = load_task(tier1_task_001)

    # Create a failed plan.
    failed = PlanResult(solved=False, status="UNSOLVABLE", actions=())

    # Assert the ground_plan function raises an error if the plan is not solved.
    with pytest.raises(ValueError, match="unsolved"):
        ground_plan(task, failed)
