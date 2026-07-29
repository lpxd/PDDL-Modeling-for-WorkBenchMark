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
from workbenchmark_pddl.frames import TableFrame, euler_deg_to_quat_wxyz
from workbenchmark_pddl.pddl_names import GRIPPER_NAME
from workbenchmark_pddl.planner import PlanResult
from workbenchmark_pddl.problem import domain_path
from workbenchmark_pddl.task import TaskSpec
from tampanda.environments.franka_env import FrankaEnvironment
from tampanda.tamp import DomainBridge

# Top-down place EE quat (TAMPanda TOP_DOWN_Y).
_PLACE_EE_QUAT = np.array([0.0, 1.0, 0.0, 0.0])

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

# Convert a YAML pose to an executor pose.
def _executor_pose(
    pos,
    rotation_deg,
    half_size,
    frame: TableFrame | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Convert the half size to a numpy array.
    half = np.asarray(half_size, dtype=float)

    # If no frame is provided, return the pose.
    if frame is None:

        # Return the pose.
        return (
            np.asarray(pos, dtype=float),
            euler_deg_to_quat_wxyz(rotation_deg),
            half,
        )

    # Calculate the center of the block.
    center = frame.center_from_yaml(pos, float(half[2]))

    # Return the pose.
    return center, euler_deg_to_quat_wxyz(rotation_deg), half

# Place while ignoring collisions with task bricks (dense Tier-3 stacks).
def _place_with_brick_exceptions(
    executor: PickPlaceLike,
    env: Any,
    brick_names: list[str],
    *args: Any,
    **kwargs: Any,
) -> bool:

    # If the environment is None or does not have add_collision_exception, return the place.
    if env is None or not hasattr(env, "add_collision_exception"):
        return executor.place(*args, **kwargs)

    # Get the collision exceptions.
    already = set(getattr(env, "collision_exceptions", []) or [])

    # Create a list of added collision exceptions.
    added: list[str] = []

    # Add the collision exceptions.
    for name in brick_names:
        if name not in already:
            env.add_collision_exception(name)
            added.append(name)

    # Try to place the block.
    try:
        return executor.place(*args, **kwargs)
    finally:
        for name in added:
            env.remove_collision_exception(name)

# Execute a grounded action on a PickPlaceLike executor.
def execute_grounded_action(
    executor: PickPlaceLike,
    action: GroundedAction,
    frame: TableFrame | None = None,
) -> bool:

    # Convert the action to an executor pose.
    pos, quat, half = _executor_pose(
        action.pos, action.rotation_deg, action.half_size, frame
    )

    # If the action is a pick, return the result of the pick.
    if action.kind == "pick":
        return executor.pick(action.brick, pos, half, quat)

    # Place or stack the block.
    if action.kind in ("place", "stack"):

        # Place or stack the block.
        return executor.place(
            action.brick,
            pos,
            ee_quat=_PLACE_EE_QUAT,
            target_block_name=action.support_brick,
        )

    # Raise an error if the action is not supported.
    raise ValueError(f"unsupported grounded action {action.kind!r}")


# Execute a grounded plan on a PickPlaceLike executor.
def execute_grounded_plan(
    executor: PickPlaceLike,
    actions: tuple[GroundedAction, ...] | list[GroundedAction],
    frame: TableFrame | None = None,
) -> bool:

    # Run the grounded actions in order.
    for action in actions:
        if not execute_grounded_action(executor, action, frame=frame):
            return False

    # Return True if all actions succeeded.
    return True


# Create a DomainBridge for a LEGO task.
def make_lego_bridge(
    env: Any,
    task: TaskSpec,
    executor: PickPlaceLike | None = None,
    frame: TableFrame | None = None,
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

    # Get the brick names from the initial blocks.
    brick_names = [b.name for b in task.initial_blocks]

    @bridge.action("pick")
    def exec_pick(env, fluents, gripper, brick, stud, layer, rot, area, brick_type):

        # Get the brick name from the PDDL object name.
        name = yaml_brick_name(brick)

        # Get the brick specification from the task.
        spec = brick_for_pick(task, name)
        half = half_size_for(spec)
        pos, quat, half_arr = _executor_pose(spec.pos, spec.rotation, half, frame)

        # Prefer top-down grasps
        # FRONT last (place uses a fixed top-down EE quat).

        # Create a list of candidates.
        candidates = None

        # Get the grasp planner.
        gp = getattr(executor, "grasp_planner", None)

        # If the grasp planner is not None.
        if gp is not None:

            # Import the GraspType from the grasp_planner module.
            from tampanda.planners.grasp_planner import GraspType

            # Generate the candidates.
            raw = gp.generate_candidates(pos, half_arr, quat)

            # Separate the candidates into top and front.
            top = [c for c in raw if c.grasp_type != GraspType.FRONT]
            front = [c for c in raw if c.grasp_type == GraspType.FRONT]
            candidates = top + front

        ok = executor.pick(name, pos, half_arr, quat, candidates=candidates)
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

        # Calculate the half size.
        half = half_size_for(spec)

        # Calculate the executor pose.
        pos, _quat, _half = _executor_pose(spec.pos, spec.rotation, half, frame)

        # Place the block while ignoring collisions with task bricks.
        ok = _place_with_brick_exceptions(
            executor,
            env,
            brick_names,
            name,
            pos,
            ee_quat=_PLACE_EE_QUAT,
            target_block_name=None,
            approach_height=0.20,
        )

        # Return False if the place failed.
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

        # Calculate the half size.
        half = half_size_for(spec)

        # Calculate the executor pose.
        pos, _quat, _half = _executor_pose(spec.pos, spec.rotation, half, frame)

        # Place the block while ignoring collisions with task bricks.
        ok = _place_with_brick_exceptions(
            executor,
            env,
            brick_names,
            top_name,
            pos,
            ee_quat=_PLACE_EE_QUAT,
            target_block_name=bottom_name,
            approach_height=0.20,
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

# Teleport the Panda to HOME.
def _snap_arm_home(env) -> None:
    
    # Calculate the number of joints to reset.
    n = min(9, len(FrankaEnvironment.HOME_QPOS), env.data.qpos.shape[0])

    # Reset the joints.
    env.data.qpos[:n] = FrankaEnvironment.HOME_QPOS[:n]

    # Reset the velocities.
    if hasattr(env, "reset_velocities"):
        env.reset_velocities()

    # Forward the environment.
    # This is needed to update the scene.
    if hasattr(env, "forward"):
        env.forward()

# Execute a PDDL plan on a DomainBridge.
def execute_plan_on_bridge(
    bridge,
    plan: PlanResult,
    *,
    return_home: bool = False,
) -> bool:

    # Raise an error if the plan is not solved.
    if not plan.solved:
        raise ValueError(f"cannot execute unsolved plan ({plan.status})")

    # Execute the actions in the plan.
    for action in plan.actions:
        # Execute the action.
        ok, _ = bridge.execute_action(action.name, *action.params)
        # If the action fails, return False.
        if not ok:
            return False
        # If return_home is True, snap the arm to HOME.
        if return_home:
            _snap_arm_home(bridge.env)
    return True


# Execute a PDDL plan on a PickPlaceLike executor.
def execute_plan(
    executor: PickPlaceLike,
    task: TaskSpec,
    plan: PlanResult,
    frame: TableFrame | None = None,
) -> bool:

    # Execute the grounded plan.
    return execute_grounded_plan(executor, ground_plan(task, plan), frame=frame)
