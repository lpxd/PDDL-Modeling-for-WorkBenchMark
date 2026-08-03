"""Ground PDDL plan actions to continuous poses via YAML lookup.

Stud/layer/rot stay symbolic for planning. Execution poses come from the
task YAML (initial_blocks for pick, goal blocks for place/stack) — see README.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from workbenchmark_pddl.geometry import BRICK_SIZE_STUDS
from workbenchmark_pddl.grid import LAYER_HEIGHT_M, STUD_PITCH_M
from workbenchmark_pddl.planner import PlanAction, PlanResult
from workbenchmark_pddl.task import BrickSpec, TaskSpec, Vec3

ActionKind = Literal["pick", "place", "stack"]


# One plan step ready for PickPlaceExecutor (no MuJoCo call here).
@dataclass(frozen=True)
class GroundedAction:

    # The action kind (pick, place, stack)
    kind: ActionKind

    # YAML brick name (without PDDL b_ prefix)
    brick: str

    # The brick type (e.g. "4x2_brick_1")
    brick_type: str

    # The position of the brick (x, y, z)
    pos: Vec3

    # The rotation of the brick
    rotation_deg: Vec3

    # Approximate half-extents for grasp sampling (metres)
    # PickPlaceExecutor.pick() needs them to generate grasp candidates.
    half_size: Vec3

    # For stack: the brick this one is placed onto (YAML name)
    support_brick: str | None = None

# This function is used to convert the PDDL object name to the YAML name.
# PDDL object b_4x2_brick_1 → YAML name 4x2_brick_1
def yaml_brick_name(pddl_object: str) -> str:
    # If the PDDL object name starts with "b_", return the name without the "b_" prefix.
    if pddl_object.startswith("b_"):
        return pddl_object[2:]
    # Otherwise, raise an error.
    # Every PDDL object name should start with "b_".
    raise ValueError(f"expected PDDL brick object with b_ prefix, got {pddl_object!r}")


# Look up a brick in initial_blocks (pick-area pose).
def brick_for_pick(task: TaskSpec, brick_name: str) -> BrickSpec:

    # Look up the brick in the initial_blocks.
    return _find_brick(task.initial_blocks, brick_name, field="initial_blocks")


# Look up a brick in goal blocks (assembly target pose).
def brick_for_place(task: TaskSpec, brick_name: str) -> BrickSpec:

    # Look up the brick in the goal blocks.
    return _find_brick(task.goal_blocks, brick_name, field="blocks")


# Half-size from Duplo type + yaw (swap width/depth at 90°).
def half_size_for(brick: BrickSpec) -> Vec3:

    # Get the width and depth of the brick in studs.
    width_studs, depth_studs = BRICK_SIZE_STUDS[brick.type]

    # Get the yaw of the brick in degrees.
    yaw = brick.rotation[2] % 180

    # If the yaw is between 45 and 135 degrees, swap the width and depth.
    if 45 <= yaw < 135:
        width_studs, depth_studs = depth_studs, width_studs

    # Calculate the half-size of the brick.
    half_x = width_studs * STUD_PITCH_M / 2.0
    half_y = depth_studs * STUD_PITCH_M / 2.0
    half_z = LAYER_HEIGHT_M / 2.0

    # Return the half-size of the brick.
    return (half_x, half_y, half_z)


# Map one PlanAction to a GroundedAction using YAML poses.
def ground_plan_action(task: TaskSpec, action: PlanAction) -> GroundedAction:

    # Get the action kind.
    kind = action.name

    # If the action kind is "pick", return the ground pick action.
    if kind == "pick":
        return _ground_pick(task, action)

    # If the action kind is "place", return the ground place action.
    if kind == "place":
        return _ground_place(task, action)

    # If the action kind is "stack", return the ground stack action.
    if kind == "stack":
        return _ground_stack(task, action)

    # Otherwise, raise an error.
    raise ValueError(f"unsupported plan action {kind!r}")


# Ground every action in a successful plan.
def ground_plan(task: TaskSpec, plan: PlanResult) -> tuple[GroundedAction, ...]:

    # If the plan is not solved, raise an error.
    if not plan.solved:
        raise ValueError(f"cannot ground unsolved plan ({plan.status})")

    # Initialize an empty list to store the grounded actions.
    grounded: list[GroundedAction] = []

    # Ground each action in the plan.
    for action in plan.actions:
        grounded.append(ground_plan_action(task, action))

    # Return the grounded actions.
    return tuple(grounded)


# Private helper function to ground a pick action.
def _ground_pick(task: TaskSpec, action: PlanAction) -> GroundedAction:

    # pick(?g, ?b, ?s, ?l, ?r, ?a, ?t) — brick is params[1]

    # Get the brick name from the action parameters.
    brick_name = yaml_brick_name(action.params[1])

    # Look up the brick in the initial_blocks.
    brick = brick_for_pick(task, brick_name)

    # Return the grounded pick action.
    return GroundedAction(
        kind="pick",
        brick=brick.name,
        brick_type=brick.type,
        pos=brick.pos,
        rotation_deg=brick.rotation,
        half_size=half_size_for(brick),
    )


# Private helper function to ground a place action.
def _ground_place(task: TaskSpec, action: PlanAction) -> GroundedAction:

    # place(?g, ?b, …) — target pose from goal YAML

    # Get the brick name from the action parameters.
    brick_name = yaml_brick_name(action.params[1])

    # Look up the brick in the goal blocks.
    brick = brick_for_place(task, brick_name)

    # Return the grounded place action.
    return GroundedAction(
        kind="place",
        brick=brick.name,
        brick_type=brick.type,
        pos=brick.pos,
        rotation_deg=brick.rotation,
        half_size=half_size_for(brick),
    )


# Private helper function to ground a stack action.
def _ground_stack(task: TaskSpec, action: PlanAction) -> GroundedAction:
    # stack(?g, ?top, ?bottom, …) — place top at its goal pose

    # Get the top brick name from the action parameters.
    top_name = yaml_brick_name(action.params[1])

    # Get the bottom brick name from the action parameters.
    bottom_name = yaml_brick_name(action.params[2])

    # Look up the top brick in the goal blocks.
    top = brick_for_place(task, top_name)

    # Ensure the support brick exists in the goal assembly.
    _find_brick(task.goal_blocks, bottom_name, field="blocks")

    # Return the grounded stack action.
    return GroundedAction(
        kind="stack",
        brick=top.name,
        brick_type=top.type,
        pos=top.pos,
        rotation_deg=top.rotation,
        half_size=half_size_for(top),
        support_brick=bottom_name,
    )


# Private helper function to find a brick in a list of bricks.
def _find_brick(
    bricks: tuple[BrickSpec, ...],
    name: str,
    *,
    field: str,
) -> BrickSpec:

    # Iterate over the bricks in the list.
    for brick in bricks:

        # If the brick name matches the name we are looking for, return the brick.
        if brick.name == name:
            return brick

    # If the brick is not found, raise an error.
    raise KeyError(f"brick {name!r} not found in {field}")

# This function is used to convert the layer_* token to the layer index.
def _layer_index(token: str) -> int:
    if not token.startswith("layer_"):
        raise ValueError(f"expected layer_* token, got {token!r}")
    return int(token.split("_", 1)[1])

# This function is used to reorder the plan so that the bottom-up order is followed.
# PDDL planners often finish one column before other base bricks. That is
# valid PDDL but forces the arm through finished towers. Prefer bottom-up
# while preserving support dependencies (stack after its support is placed).
def reorder_plan_bottom_up(plan: PlanResult) -> PlanResult:

    # If the plan is not solved or has no actions, return the plan.
    if not plan.solved or not plan.actions:
        return plan

    # Convert the plan actions to a list.
    actions = list(plan.actions)

    # If the number of actions is not even, return the plan.
    if len(actions) % 2 != 0:
        return plan

    # Initialize a list to store the pick, place, brick, layer, and support.
    units: list[tuple[PlanAction, PlanAction, str, int, str | None]] = []

    # Iterate over the actions in pairs.
    for i in range(0, len(actions), 2):

        # Get the pick and place actions.
        pick = actions[i]
        put = actions[i + 1]

        # If the pick action is not a pick action or the place action is not a place or stack action, return the plan.
        if pick.name != "pick" or put.name not in ("place", "stack"):
            return plan

        # Get the brick name from the pick action.
        brick = yaml_brick_name(pick.params[1])

        # If the place action is a place action, get the layer index from the place action.
        if put.name == "place":
            layer = _layer_index(put.params[3])

        # If the place action is a stack action, get the support brick name from the place action.
            support = None
            
        else:
            layer = _layer_index(put.params[5])
            support = yaml_brick_name(put.params[2])

        # Add the pick, place, brick, layer, and support to the list.
        units.append((pick, put, brick, layer, support))

    # Create a dictionary to store the index of the brick.
    by_brick = {u[2]: i for i, u in enumerate(units)}

    # Initialize a set to store the remaining bricks.
    remaining = set(range(len(units)))

    # Initialize a list to store the ordered bricks.
    ordered: list[int] = []

    # While there are remaining bricks, find the next brick to place.
    while remaining:

        # Find the next brick to place.
        ready = [
            i
            for i in remaining
            if units[i][4] is None or by_brick.get(units[i][4], -1) in ordered
        ]

        # If there are no ready bricks, return the plan.
        if not ready:
            return plan  # cyclic / malformed — keep original

        # Sort the ready bricks by layer and index.
        ready.sort(key=lambda i: (units[i][3], i))

        # Choose the first ready brick.
        chosen = ready[0]

        # Add the chosen brick to the ordered list.
        ordered.append(chosen)

        # Remove the chosen brick from the remaining set.
        remaining.remove(chosen)

    # Initialize a list to store the new actions.
    new_actions: list[PlanAction] = []

    # Iterate over the ordered bricks.
    for i in ordered:

        # Get the pick and place actions.
        pick, put, *_ = units[i]

        # Add the pick and place actions to the new actions list.
        new_actions.append(pick)
        new_actions.append(put)

    # If the new actions are the same as the original actions, return the plan.
    if tuple(new_actions) == plan.actions:
        return plan

    # Return the new plan.
    return PlanResult(
        solved=plan.solved,
        status=plan.status,
        actions=tuple(new_actions),
    )
