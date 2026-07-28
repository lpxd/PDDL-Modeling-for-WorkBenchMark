"""Wire PDDL actions to PickPlaceExecutor via DomainBridge.

Pose targets still come from YAML (see execution.py). 
MuJoCo body names must match YAML brick names (e.g. ``4x2_brick_1``).
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from workbenchmark_pddl.execution import (
    GroundedAction,
    brick_for_pick,
    brick_for_place,
    ground_plan,
    half_size_for,
    yaml_brick_name,
)
from workbenchmark_pddl.pddl_names import GRIPPER_NAME
from workbenchmark_pddl.planner import PlanResult
from workbenchmark_pddl.problem import domain_path
from workbenchmark_pddl.task import TaskSpec, Vec3
from tampanda.tamp import DomainBridge

# A minimal surface of TAMPanda's PickPlaceExecutor used here.
class PickPlaceLike(Protocol):
   
    # Pick a block from the environment.
    def pick(
        self,
        block_name: str,
        block_pos: np.ndarray,
        half_size: np.ndarray,
        block_quat: np.ndarray,
        candidates: Any = None,
    ) -> bool: ...

    # Place a block in the environment.
    def place(
        self,
        block_name: str,
        place_block_center: np.ndarray,
        ee_quat: np.ndarray | None = None,
        target_block_name: str | None = None,
        approach_height: float = 0.15,
        place_clearance: float = 0.003,
    ) -> bool: ...


# Convert XYZ (roll, pitch, yaw) degrees to MuJoCo wxyz quaternion.
def euler_deg_to_quat_wxyz(rotation_deg: Vec3) -> np.ndarray:

    # Convert the rotation degrees to radians.
    roll, pitch, yaw = np.deg2rad(np.asarray(rotation_deg, dtype=float))

    # Compute the cosine and sine of the roll, pitch, and yaw.
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)

    # Compute the quaternion.
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )


# Execute a grounded action on a PickPlaceLike executor.
def execute_grounded_action(
    executor: PickPlaceLike,
    action: GroundedAction,
) -> bool:

    # Convert the position and half-size to numpy arrays.
    pos = np.asarray(action.pos, dtype=float)
    half = np.asarray(action.half_size, dtype=float)

    # Convert the rotation degrees to a quaternion.
    quat = euler_deg_to_quat_wxyz(action.rotation_deg)

    # Dispatch the action to the executor.
    if action.kind == "pick":
        return executor.pick(action.brick, pos, half, quat)

    # Place or stack the block.
    if action.kind in ("place", "stack"):

        # Place or stack the block.
        return executor.place(
            action.brick,
            pos,
            ee_quat=None,
            target_block_name=action.support_brick,
        )

    # Raise an error if the action is not supported.
    raise ValueError(f"unsupported grounded action {action.kind!r}")


# Execute a grounded plan on a PickPlaceLike executor.
def execute_grounded_plan(
    executor: PickPlaceLike,
    actions: tuple[GroundedAction, ...] | list[GroundedAction],
) -> bool:

    # Run the grounded actions in order.
    for action in actions:

        # Execute the grounded action.
        # If the action fails, return False.
        if not execute_grounded_action(executor, action):
            return False

    # Return True if all actions succeeded.
    return True


# Create a DomainBridge for a LEGO task.
def make_lego_bridge(
    env: Any,
    task: TaskSpec,
    executor: PickPlaceLike | None = None,
):
    # Create a DomainBridge for the task.
    bridge = DomainBridge(domain_path(), env)

    # Create a fluent for the holding state.
    bridge.fluent("holding", initial=None)

    # Create a fluent for the gripper-empty state.
    bridge.fluent("gripper-empty", initial=[(GRIPPER_NAME,)])

    # If no executor is provided, return the bridge.
    if executor is None:
        return bridge

    # Register the pick action.
    @bridge.action("pick")
    def exec_pick(env, fluents, gripper, brick, stud, layer, rot, area, brick_type):

        # Get the brick name from the PDDL object name.
        name = yaml_brick_name(brick)

        # Get the brick specification from the task.
        spec = brick_for_pick(task, name)

        # Pick the block.
        ok = executor.pick(
            name,
            np.asarray(spec.pos, dtype=float),
            np.asarray(half_size_for(spec), dtype=float),
            euler_deg_to_quat_wxyz(spec.rotation),
        )

        # If the pick fails, return False.
        if not ok:
            return False, {}

        # Return True if the pick succeeded.
        return True, {
            ("holding", gripper, brick): True,
            ("gripper-empty", gripper): False,
        }

    # Register the place action.
    @bridge.action("place")
    def exec_place(env, fluents, gripper, brick, stud, layer, rot, area, brick_type):

        # Get the brick name from the PDDL object name.
        name = yaml_brick_name(brick)

        # Get the brick specification from the task.
        spec = brick_for_place(task, name)

        # Place the block.
        ok = executor.place(
            name,
            np.asarray(spec.pos, dtype=float),
            ee_quat=None,
            target_block_name=None,
        )

        # If the place fails, return False.
        if not ok:
            return False, {}

        # Return True if the place succeeded.
        return True, {
            ("holding", gripper, brick): False,
            ("gripper-empty", gripper): True,
        }

    # Register the stack action.
    @bridge.action("stack")
    def exec_stack(
        env,
        fluents,
        gripper,
        top,
        bottom,
        top_s,
        bot_s,
        top_l,
        bot_l,
        top_r,
        bot_r,
        area,
        t_top,
        t_bot,
    ):

        # Get the brick name from the PDDL object name.
        top_name = yaml_brick_name(top)
        bottom_name = yaml_brick_name(bottom)

        # Get the brick specification from the task.
        spec = brick_for_place(task, top_name)

        # Stack the block.
        ok = executor.place(
            top_name,
            np.asarray(spec.pos, dtype=float),
            ee_quat=None,
            target_block_name=bottom_name,
        )

        # If the stack fails, return False.
        if not ok:
            return False, {}

        # Return True if the stack succeeded.
        return True, {
            ("holding", gripper, top): False,
            ("gripper-empty", gripper): True,
        }

    return bridge

# Execute a PDDL plan on  DomainBridge.
def execute_plan_on_bridge(bridge, plan: PlanResult) -> bool:

    # If the plan is not solved, raise an error.
    if not plan.solved:
        raise ValueError(f"cannot execute unsolved plan ({plan.status})")

    # Execute the plan actions on the bridge.
    for action in plan.actions:

        # Execute the action on the bridge.
        ok, _ = bridge.execute_action(action.name, *action.params)

        # If the action fails, return False.
        if not ok:
            return False

    # Return True if all actions succeeded.
    return True


# Execute a PDDL plan on a PickPlaceLike executor.
def execute_plan(
    executor: PickPlaceLike,
    task: TaskSpec,
    plan: PlanResult,
) -> bool:

    # Ground the plan via YAML and run it on the executor.
    return execute_grounded_plan(executor, ground_plan(task, plan))
