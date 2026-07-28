"""Tests for GroundedAction → PickPlaceExecutor / DomainBridge wiring (no MuJoCo)."""

from pathlib import Path

import numpy as np
import pytest

from workbenchmark_pddl.bridge import (
    euler_deg_to_quat_wxyz,
    execute_grounded_plan,
    execute_plan,
    execute_plan_on_bridge,
    make_lego_bridge,
)
from workbenchmark_pddl.execution import GroundedAction, brick_for_pick, brick_for_place
from workbenchmark_pddl.planner import plan_task
from workbenchmark_pddl.task import load_task

# A stand-in for PickPlaceExecutor that records calls.
# This is used to test the bridge wiring without MuJoCo.
class RecordingExecutor:

    # Initialize the executor.
    def __init__(self) -> None:
        # List of calls to the executor.
        self.calls: list[tuple] = []
        # Set of blocks to fail on.
        self.fail_on: set[str] = set()

    # Pick a block from the environment.
    def pick(self, block_name, block_pos, half_size, block_quat, candidates=None):
        # Record the call.
        self.calls.append(
            (
                "pick",
                block_name,
                np.asarray(block_pos).copy(),
                np.asarray(half_size).copy(),
                np.asarray(block_quat).copy(),
            )
        )
        return block_name not in self.fail_on

    # Place a block in the environment.
    def place(
        self,
        block_name,
        place_block_center,
        ee_quat=None,
        target_block_name=None,
        approach_height=0.15,
        place_clearance=0.003,
    ):
        # Record the call.
        self.calls.append(
            (
                "place",
                block_name,
                np.asarray(place_block_center).copy(),
                target_block_name,
            )
        )
        return block_name not in self.fail_on

# Test the identity Euler is a unit quaternion.
# Should be (1.0, 0.0, 0.0, 0.0).
def test_identity_euler_is_unit_quat():
    q = euler_deg_to_quat_wxyz((0.0, 0.0, 0.0))
    assert q == pytest.approx(np.array([1.0, 0.0, 0.0, 0.0]))

# Test the yaw 90° quaternion.
# Should be (sqrt(0.5), 0.0, 0.0, sqrt(0.5)).
def test_yaw_90_quat():
    q = euler_deg_to_quat_wxyz((0.0, 0.0, 90.0))
    # wxyz for +90° about Z
    assert q == pytest.approx(
        np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]),
        abs=1e-9,
    )

# Test executing a grounded pick and stack action.
# Should record the pick and stack calls.
def test_execute_grounded_pick_and_stack(tier1_task_001: Path):

    # Load the task.
    task = load_task(tier1_task_001)

    # Get the pick and place bricks.
    pick_brick = brick_for_pick(task, "4x2_brick_1")
    place_brick = brick_for_place(task, "4x2_brick_2")

    # Create the pick action.
    pick = GroundedAction(
        kind="pick",
        brick=pick_brick.name,
        brick_type=pick_brick.type,
        pos=pick_brick.pos,
        rotation_deg=pick_brick.rotation,
        half_size=(0.032, 0.016, 0.00955),
    )

    # Create the stack action.
    stack = GroundedAction(
        kind="stack",
        brick=place_brick.name,
        brick_type=place_brick.type,
        pos=place_brick.pos,
        rotation_deg=place_brick.rotation,
        half_size=(0.032, 0.016, 0.00955),
        support_brick="4x2_brick_1",
    )

    # Create the executor.
    executor = RecordingExecutor()

    # Execute the pick and stack actions.
    assert execute_grounded_plan(executor, (pick, stack))

    # Assert the pick call is recorded.
    assert executor.calls[0][0] == "pick"
    assert executor.calls[0][1] == "4x2_brick_1"
    np.testing.assert_allclose(executor.calls[0][2], pick_brick.pos)

    # Assert the stack call is recorded.
    assert executor.calls[1][0] == "place"
    assert executor.calls[1][1] == "4x2_brick_2"
    assert executor.calls[1][3] == "4x2_brick_1"
    np.testing.assert_allclose(executor.calls[1][2], place_brick.pos)

# Test executing a grounded plan stops on failure.
# Should record the pick call and stop on the failure.
def test_execute_grounded_stops_on_failure():

    # Create the executor.
    executor = RecordingExecutor()

    # Add a block to fail on.
    executor.fail_on.add("a")

    # Create the actions.
    actions = (
        GroundedAction(
            kind="pick",
            brick="a",
            brick_type="brick_4x2",
            pos=(0.0, 0.0, 0.0),
            rotation_deg=(0.0, 0.0, 0.0),
            half_size=(0.01, 0.01, 0.01),
        ),
        GroundedAction(
            kind="place",
            brick="a",
            brick_type="brick_4x2",
            pos=(1.0, 0.0, 0.0),
            rotation_deg=(0.0, 0.0, 0.0),
            half_size=(0.01, 0.01, 0.01),
        ),
    )

    # Execute the grounded plan.
    assert not execute_grounded_plan(executor, actions)

    # Assert the pick call is recorded.
    assert len(executor.calls) == 1
    assert executor.calls[0][0] == "pick"

# Test executing a PDDL plan on a PickPlaceLike executor.
# Should record the pick and place calls.
def test_execute_plan_grounds_tier1(tier1_task_001: Path):

    # Load the task.
    task = load_task(tier1_task_001)

    # Plan the task.
    plan = plan_task(task)
    assert plan.solved

    # Create the executor.
    executor = RecordingExecutor()
    assert execute_plan(executor, task, plan)

    # Assert the pick and place calls are recorded.
    kinds = [c[0] for c in executor.calls]
    assert kinds[0] == "pick"
    assert kinds.count("pick") == kinds.count("place")
    assert all(c[0] in ("pick", "place") for c in executor.calls)

# Test executing a PDDL plan on a DomainBridge.
# Should record the pick and place calls.
def test_domain_bridge_executes_tier1_plan(tier1_task_001: Path):

    # Load the task.
    task = load_task(tier1_task_001)

    # Plan the task.
    plan = plan_task(task)
    assert plan.solved

    # Create the executor.
    executor = RecordingExecutor()
    bridge = make_lego_bridge(env=object(), task=task, executor=executor)
    assert execute_plan_on_bridge(bridge, plan)

    # Bridge path uses YAML names as MuJoCo body ids.
    assert executor.calls[0][0] == "pick"
    assert executor.calls[0][1] == "4x2_brick_1"
    np.testing.assert_allclose(
        executor.calls[0][2],
        brick_for_pick(task, "4x2_brick_1").pos,
    )

    # Assert the last call is a place.
    last = executor.calls[-1]
    assert last[0] == "place"
    assert last[1] == "4x2_brick_2"
    assert last[3] == "4x2_brick_1"
