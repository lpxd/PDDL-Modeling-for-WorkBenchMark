#!/usr/bin/env python
"""
Show one task in the MuJoCo viewer.

Uses ``results/tierN/task_XXX/plan.json`` when present; otherwise plans once
in memory (not written to disk) and executes that.

Does not write or update any metrics / summary files — visual demo only.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Protocol, cast

from workbenchmark_pddl.bridge import make_lego_bridge
from workbenchmark_pddl.duplo_hold import get_duplo_hold
from workbenchmark_pddl.execution import reorder_plan_bottom_up
from workbenchmark_pddl.frames import DEFAULT_SURFACE_Z, TableFrame
from workbenchmark_pddl.grid import LAYER_HEIGHT_M
from workbenchmark_pddl.planner import PlanResult, plan_result_from_dict, plan_task
from workbenchmark_pddl.scene import (
    build_lego_env,
    make_lego_grasp_planner,
    place_initial_blocks,
)
from workbenchmark_pddl.task import TaskSpec, load_task

from tampanda.planners.grasp_planner import GRASP_CONTACT_OFFSET
from tampanda.planners.pick_place import PickPlaceExecutor
from tampanda.planners.rrt_star import RRTStar

import mujoco
import numpy as np

# Private helper function that flattens a quaternion to a yaw-preserving orientation.
# Used to keep the brick upright during place.
def _flatten_keep_yaw(quat_wxyz: np.ndarray) -> np.ndarray:
    
    # Convert the quaternion to a yaw-preserving orientation.
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)

    # Extract the quaternion components.
    w, x, y, z = (float(v) for v in q)

    # Calculate the yaw angle.
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    # Calculate the half yaw angle.
    half = 0.5 * yaw

    # Return the yaw-preserving orientation.
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)

# Private helper function that converts a quaternion to a rotation matrix.
# Used to convert the brick quaternion to a rotation matrix.
def _quat_wxyz_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:

    # Create a 9-element array to store the rotation matrix.
    mat = np.zeros(9, dtype=np.float64)

    # Convert the quaternion to a rotation matrix.
    mujoco.mju_quat2Mat(mat, np.asarray(quat_wxyz, dtype=np.float64).reshape(4))

    # Return the rotation matrix.
    return mat.reshape(3, 3)


# Private helper function that converts a rotation matrix to a quaternion.
# Used to convert the rotation matrix to a quaternion.
def _mat_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:

    # Create a 4-element array to store the quaternion.
    quat = np.zeros(4, dtype=np.float64)

    # Convert the rotation matrix to a quaternion.
    mujoco.mju_mat2Quat(quat, np.asarray(mat, dtype=np.float64).reshape(9))

    # Return the quaternion.
    return quat


# Private helper function that converts a brick quaternion to a EE orientation.
def _ee_quat_for_attached_brick(
    env: object,
    brick_quat_wxyz: np.ndarray,
) -> np.ndarray | None:
    
    # Get the attached brick.
    attached = getattr(env, "_attached", None)

    # If the attached brick is not a dictionary, return None.
    if not isinstance(attached, dict):
        return None

    # Get the relative rotation matrix.
    rel_mat = attached.get("rel_mat")

    # If the relative rotation matrix is None, return None.
    if rel_mat is None:
        return None

    # Convert the brick quaternion to a rotation matrix.
    brick_mat = _quat_wxyz_to_mat(_flatten_keep_yaw(brick_quat_wxyz))

    # Convert the relative rotation matrix to a numpy array.
    rel_m = np.asarray(rel_mat, dtype=np.float64).reshape(3, 3)

    # Convert the rotation matrix to a quaternion.
    # brick_mat = ee_mat @ rel_mat  →  ee_mat = brick_mat @ rel_mat.T
    return _mat_to_quat_wxyz(brick_mat @ rel_m.T)


# Showcase PickPlace: keep brick attached until the place pose (no mid-air drop).
class _ShowcasePickPlaceExecutor(PickPlaceExecutor):

    # Private helper function that waits for the gripper to open.
    def _wait_gripper_open(self, steps: int = 80) -> None:
        super()._wait_gripper_open(steps=steps)

    # Private helper function that waits for the gripper to close.
    def _wait_gripper_closed(self, max_steps: int = 125) -> None:
        super()._wait_gripper_closed(max_steps=max_steps)

    # Private helper function that waits for the arm to settle.
    def _wait_arm_settle(self, max_steps: int = 60) -> None:
        super()._wait_arm_settle(max_steps=max_steps)

    # Private helper function that puts the attached brick back on the real EE after any planner state poke.
    def _glue_held_brick(self) -> None:
        
        # Get the environment.
        env = self.env

        # If the attached brick is not a dictionary, return.
        if getattr(env, "_attached", None) is None:
            return

        # Apply the attachment.
        env._apply_attachment()

        # Forward the environment.
        mujoco.mj_forward(env.model, env.data)

        # Get the viewer.
        viewer = getattr(env, "viewer", None)

        if viewer is not None:
            # Sync the viewer.
            with contextlib.suppress(Exception):
                viewer.sync()

    # Private helper function that suspends TAMPanda's held-body teleport used by collision checks.
    @contextlib.contextmanager
    def _no_held_body_warp(self):
        
        # Get the environment.
        env = self.env

        # Get the attached brick.
        attached = getattr(env, "_attached", None)

        # Get the collision held body.
        had_warp = getattr(env, "_collision_held_body", None) is not None

        # If the collision held body is not None, clear it.
        if had_warp:
            # Clear the collision held body.
            env.clear_collision_held_body()
        try:
            yield
        finally:
            # If the collision held body is not None and the attached brick is a dictionary, set the collision held body.
            if had_warp and isinstance(attached, dict):

                # Set the collision held body.
                env.set_collision_held_body(
                    attached["body_name"],
                    np.asarray(attached["rel_pos"], dtype=np.float64),
                    np.asarray(attached["rel_mat"], dtype=np.float64),
                )

            # Put the attached brick back on the real EE.
            self._glue_held_brick()

    # Private helper function that plans to a pose while holding the attached brick.
    def _plan_to_pose_holding(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
    ) -> np.ndarray | None:
        
        # Get the environment.
        env = self.env

        # Save the current position and velocity.
        qpos_save = env.data.qpos.copy()

        # Save the current velocity.
        qvel_save = env.data.qvel.copy()

        # Plan to the pose.
        try:
            return self.planner.plan_to_pose(
                target_pos,
                target_quat,
                dt=0.005,
                max_iterations=self.max_plan_iters,
            )
        finally:
            # Restore the current position and velocity.
            env.data.qpos[:] = qpos_save
            env.data.qvel[:] = qvel_save

            # Forward the environment.
            mujoco.mj_forward(env.model, env.data)

            # Put the attached brick back on the real EE.
            self._glue_held_brick()

    # Private helper function that executes a path while holding the attached brick.
    def _execute_path_holding(self, path, step_size: float) -> None:
        
        # Queue the path and run it to completion.
        with self._no_held_body_warp():
            self.env.execute_path(path, self.planner, step_size=step_size)

        # Wait for the environment to settle.
        self.env.wait_idle(settle_steps=self.settle_steps)

        # Put the attached brick back on the real EE.
        self._glue_held_brick()

    # Function that places a brick.
    # Keep kinematic attach through the descent so the brick does not fall
    # from approach height (default PickPlace detaches before descending).
    def place(
        self,
        block_name: str,
        place_block_center: np.ndarray,
        ee_quat: np.ndarray | None = None,
        target_block_name: str | None = None,
        approach_height: float = 0.12,
        place_clearance: float = 0.0,
        place_quat: np.ndarray | None = None,
    ) -> bool:

        # Set the place clearance.
        _ = place_clearance

        # Put the attached brick back on the real EE.
        self._glue_held_brick()

        # Flush the place center.
        flush = self._flush_place_center(
            block_name, place_block_center, target_block_name
        )

        # If the last grasp quaternion is None and the EE quaternion is None, raise an error.
        if self._last_grasp_quat is None and ee_quat is None:
            raise ValueError(
                "ee_quat must be provided if no prior pick has been executed"
            )

        # Get the carry quaternion.
        carry_quat = (
            self._last_grasp_quat
            if self._last_grasp_quat is not None
            else np.asarray(ee_quat, dtype=np.float64)
        )

        # Carry to approach with pick orientation (reachable). Reorient above
        # the place — never plan goal-yaw from the pick site (that often fails
        # and the old kinematic-seat fallback teleported the brick).
        ee_approach = flush.copy()
        ee_approach[2] += GRASP_CONTACT_OFFSET + approach_height
        ee_place = flush.copy()
        ee_place[2] += GRASP_CONTACT_OFFSET
        ee_quat = carry_quat

        # Get the goal quaternion.
        goal_quat = None

        # If the place quaternion is not None and the attachment is used and the attached brick is not None, get the goal quaternion.
        if (
            place_quat is not None
            and self.use_attachment
            and getattr(self.env, "_attached", None) is not None
        ):
            goal_quat = _ee_quat_for_attached_brick(self.env, place_quat)

        # Add a collision exception for the block.
        self.env.add_collision_exception(block_name)

        # If the target block name is not None, add a collision exception for the target block.
        if target_block_name is not None:
            self.env.add_collision_exception(target_block_name)

        # Get the duplo hold.
        hold = get_duplo_hold(self.env)

        # Get the extra collision exceptions.
        extra_except: list[str] = []

        # If the duplo hold is not None, add collision exceptions for the pins and welds.
        if hold is not None:

            # Iterate over the pins and welds.
            for name in list(hold._pins) + list(hold._welds):
                if name not in (block_name, target_block_name):
                    # Add a collision exception for the pin or weld.
                    self.env.add_collision_exception(name)
                    # Add the pin or weld to the extra collision exceptions.
                    extra_except.append(name)

        # Private helper function that releases the extra collision exceptions.
        def _release_clear() -> None:

            # Iterate over the extra collision exceptions.
            for name in extra_except:
                # Remove the collision exception for the pin or weld.
                self.env.remove_collision_exception(name)

            # Clear the collision exceptions for the block and target block.
            self._clear_exceptions(block_name, target_block_name)

        # Private helper function that pins the brick only.
        def _pin_only() -> None:
            # Always goal yaw — wrong carry yaw overlaps neighbours when pinned.

            # Seat and pin the brick.
            self._seat_and_pin(
                block_name,
                place_block_center,
                place_quat=place_quat,
                target_block_name=target_block_name,
                cur_quat=None,
                prefer_release_yaw=False,
            )

        # Private helper function that seats and opens the gripper and lifts the brick.
        def _seat_open_lift() -> None:

            # Seat and pin the brick.
            _pin_only()

            # Open the gripper.
            self.env.controller.open_gripper()
            self._wait_gripper_open()

            # Pin the brick again. Fingers can nudge before pin re-applies.
            _pin_only()

            # Retreat while exceptions still active so fingers cannot scoop the brick.

            # Get the clear position.
            clear = ee_approach.copy()
            clear[2] = max(float(clear[2]), float(flush[2]) + 0.08)

            # Plan to the clear position.
            path_up = self.planner.plan_to_pose(
                clear,
                ee_quat,
                dt=0.005,
                max_iterations=self.max_plan_iters,
            )

            # Pin the brick again.
            _pin_only()

            # If the path up is not None, execute the path up.
            if path_up is not None:
                # Execute the path up.
                self._execute_path_holding(path_up, self.retreat_step_size)
            else:
                # Last resort: plan straight up from current EE.

                # Get the current EE position.
                ee_now = self.env.data.site_xpos[
                    self.env._ee_site_id
                ].copy()

                # Add 0.08 to the Z-axis of the current EE position.
                ee_now[2] += 0.08

                # Plan to the clear position.
                path_up = self.planner.plan_to_pose(
                    ee_now,
                    ee_quat,
                    dt=0.005,
                    max_iterations=self.max_plan_iters,
                )

                # Pin the brick again.
                _pin_only()

                # If the path up is not None, execute the path up.
                if path_up is not None:
                    # Execute the path up.
                    self._execute_path_holding(path_up, self.retreat_step_size)

            # Pin the brick again.
            _pin_only()

            # Release the extra collision exceptions.
            _release_clear()

        # Plan to the approach pose.
        path = self._plan_to_pose_holding(ee_approach, carry_quat)

        # If the path is None, print an error message and release the extra collision exceptions and return False.
        if path is None:
            # Print an error message.
            print("[ShowcasePickPlace] place approach plan failed")
            # Release the extra collision exceptions.
            _release_clear()
            # Return False.
            return False

        # Execute the path to the approach pose.
        self._execute_path_holding(path, self.approach_step_size)

        # Twist to goal yaw above the place (keep approach XY/Z — reachable).
        if goal_quat is not None:
            # Plan to the goal pose.
            path = self._plan_to_pose_holding(ee_approach, goal_quat)
            # If the path is not None, execute the path to the goal pose.
            if path is not None:
                # Execute the path to the goal pose.
                self._execute_path_holding(path, self.approach_step_size)
                # Set the EE quaternion to the goal quaternion.
                ee_quat = goal_quat

        # Plan to the place pose.
        path = self._plan_to_pose_holding(ee_place, ee_quat)

        # If the path is None, print an error message and release the extra collision exceptions and return False.
        if path is None:
            # Print an error message.
            print("[ShowcasePickPlace] place descent plan failed")
            # Release the extra collision exceptions.
            _release_clear()
            # Return False.
            return False

        # Execute the path to the place pose.
        self._execute_path_holding(path, self.place_step_size)

        # Seat and open the gripper and lift the brick.
        _seat_open_lift()

        # Print a success message.

        # Print a success message.
        print("[ShowcasePickPlace] place SUCCESS")
        # Return True.
        return True

    # Private helper function that flushes the place center.
    def _flush_place_center(
        self,
        block_name: str,
        place_block_center: np.ndarray,
        target_block_name: str | None,
    ) -> np.ndarray:

        # Create a copy of the place block center.
        snap_pos = np.asarray(place_block_center, dtype=np.float64).copy()

        # Get the half size of the block.
        child_half = self.env.get_object_half_size(block_name)

        # Get the height of the block.
        chz = (
            float(child_half[2])
            if child_half is not None
            else 0.5 * LAYER_HEIGHT_M
        )

        # If the target block name is not None, get the parent position and quaternion.
        if target_block_name is not None:

            # Get the parent position and quaternion.
            parent_pos, _pq = self.env.get_object_pose(target_block_name)

            # If the parent position is not None, get the parent half size.
            if parent_pos is not None:

                # Get the parent half size.
                parent_half = self.env.get_object_half_size(target_block_name)

                # Get the parent height.
                phz = (
                    float(parent_half[2])
                    if parent_half is not None
                    else 0.5 * LAYER_HEIGHT_M
                )

                # Set the Z-axis of the snap position to the parent position Z-axis plus the parent height plus the child height.
                snap_pos[2] = float(parent_pos[2]) + phz + chz
        else:
            # Set the Z-axis of the snap position to the default surface Z-axis plus the child height.
            snap_pos[2] = DEFAULT_SURFACE_Z + chz

        # Return the snap position.
        return snap_pos


    # Private helper function that seats and pins the brick.
    def _seat_and_pin(
        self,
        block_name: str,
        place_block_center: np.ndarray,
        *,
        place_quat: np.ndarray | None,
        target_block_name: str | None,
        cur_quat: np.ndarray | None,
        prefer_release_yaw: bool = False,
    ) -> None:

        # If the attachment is used and the attached brick is not None, detach the object.
        if self.use_attachment and getattr(self.env, "_attached", None) is not None:
            self.env.detach_object()

        # If the current quaternion is None, get the object pose.
        if cur_quat is None:
            # Get the object pose.
            _pos, cur_quat = self.env.get_object_pose(block_name)

        # If the current quaternion is None, set the current quaternion to the identity quaternion.
        if cur_quat is None:
            # Set the current quaternion to the identity quaternion.
            cur_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # If the arm already turned to goal yaw or the place quaternion is None, keep the release orientation.
        if prefer_release_yaw or place_quat is None:
            # Flatten the current quaternion and keep the yaw.
            snap_quat = _flatten_keep_yaw(cur_quat)
        else:
            # Flatten the place quaternion and keep the yaw.
            snap_quat = _flatten_keep_yaw(np.asarray(place_quat, dtype=np.float64))

        # Flush the place center.
        snap_pos = self._flush_place_center(
            block_name, place_block_center, target_block_name
        )

        # Set the object pose.
        self.env.set_object_pose(block_name, snap_pos, snap_quat)

        # Reset the velocities.
        self.env.reset_velocities()

        # Forward the environment.
        self.env.forward()

        # Get the duplo hold.
        hold = get_duplo_hold(self.env)

        # If the duplo hold is not None, pin the target block.
        if hold is not None:
            # If the target block name is not None, pin the target block.
            if target_block_name is not None:
                # Pin the target block.
                hold.pin(target_block_name)
            # Pin the block.
            hold.pin(block_name)


# Hide IK mocap target + collision capsules in the viewer.
def _hide_viewer_clutter(env: object, viewer: object) -> None:

    # Get the viewer options.
    opt = getattr(viewer, "opt", None)

    # Get the viewer geometry groups.
    groups = getattr(opt, "geomgroup", None) if opt is not None else None

    # If the viewer geometry groups is not None, hide the viewer geometry groups.
    if groups is not None:
        # Try to hide the viewer geometry groups.
        try:
            groups[3] = 0
        except Exception:  # noqa: BLE001
            pass

    # Get the model.
    model = getattr(env, "model", None)

    # If the model is None, return.
    if model is None:
        return

    # Iterate over the hand capsule and link7 capsule.
    for name in ("hand_capsule", "link7_capsule"):
        # Get the geometry ID.
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        # If the geometry ID is not -1, set the geometry rgba to transparent.
        if gid >= 0:
            # Set the geometry rgba to transparent.
            model.geom_rgba[gid][3] = 0.0

    # Translucent red box is the IK mocap "target" (next EE pose marker).
    target_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    if target_bid >= 0:
        for gid in range(int(model.ngeom)):
            if int(model.geom_bodyid[gid]) == target_bid:
                model.geom_rgba[gid][3] = 0.0

# Paths to the project and dataset including the ground truth poses
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "dataset"
GROUND_TRUTH_ROOT = DATASET_ROOT / "ground_truth"

# Path to the results directory (saved plans from eval_planning.py)
RESULTS_DIR = PROJECT_ROOT / "results"

# Name of the plan file
PLAN_FILENAME = "plan.json"

# Protocol for the bridge interface
class _BridgeLike(Protocol):

    # The environment
    env: object

    # Execute an action
    def execute_action(
        self, name: str, *params: object
    ) -> tuple[bool, object]: ...

# Protocol for the viewer interface
class _ViewerLike(Protocol):

    # Check if the viewer is running
    def is_running(self) -> bool: ...

    # Sync the viewer
    def sync(self) -> None: ...

# Path to the ground truth file for a given task
def ground_truth_path(tier: int, task_id: int) -> Path:
    return GROUND_TRUTH_ROOT / f"tier{tier}" / f"task_{task_id:03d}.yaml"

# Path to the plan file for a given task under results/
def task_plan_path(tier: int, task_id: int) -> Path:
    return RESULTS_DIR / f"tier{tier}" / f"task_{task_id:03d}" / PLAN_FILENAME


# Parse a task reference string like "3/20" or "tier3/task_020"
def parse_task_ref(raw: str) -> tuple[int, int]:

    # Strip whitespace and convert to lowercase, replacing backslashes with forward slashes
    text = raw.strip().lower().replace("\\", "/")

    # Raise an error if the task reference is empty
    if not text:
        raise argparse.ArgumentTypeError("empty task reference")

    # Split the task reference into tier and task id
    if "/" in text:
        left, right = text.split("/", 1)
    elif "," in text:
        left, right = text.split(",", 1)
    else:
        # Raise an error if the task reference is not in the expected format
        raise argparse.ArgumentTypeError(
            f"expected TIER/TASK like 3/20 or tier3/task_020, got {raw!r}"
        )

    # Strip whitespace and convert to lowercase, replacing backslashes with forward slashes
    left = left.strip()
    right = right.strip()

    # Remove the "tier" prefix if it exists
    if left.startswith("tier"):
        left = left[4:]

    # Remove the "task_" prefix if it exists
    if right.startswith("task_"):
        right = right[5:]

    # Remove the "task" prefix if it exists
    elif right.startswith("task"):
        right = right[4:]

    # Try to convert the tier and task id to integers
    try:
        tier = int(left)
        task_id = int(right)

    # Raise an error if the tier or task id is not an integer
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected TIER/TASK like 3/20 or tier3/task_020, got {raw!r}"
        ) from exc

    # Raise an error if the tier or task id is less than 1
    if tier < 1 or task_id < 1:
        raise argparse.ArgumentTypeError(
            f"tier and task_id must be >= 1, got {raw!r}"
        )

    # Return the tier and task id
    return tier, task_id


# Load a saved plan from a file
def load_saved_plan(plan_path: Path) -> PlanResult:
    # Load the plan from the file
    raw = cast(object, json.loads(plan_path.read_text(encoding="utf-8")))

    # Raise an error if the plan is not a dictionary
    if not isinstance(raw, dict):
        raise TypeError(f"expected object in {plan_path}")

    # Return the plan result
    return plan_result_from_dict(cast(dict[str, object], raw))


# Prefer results/plan.json; otherwise plan once in memory (no disk write).
def resolve_plan(task: TaskSpec, tier: int, task_id: int) -> tuple[PlanResult, str]:

    # Get the path to the plan file under results/
    plan_path = task_plan_path(tier, task_id)

    # Reuse a saved plan when available
    if plan_path.is_file():
        plan = load_saved_plan(plan_path)
        plan = reorder_plan_bottom_up(plan)
        return plan, f"{plan_path}  ({len(plan.actions)} actions, bottom-up)"

    # No saved plan — tell the user, then plan in memory without writing
    print(
        f"  No saved plan at {plan_path} — planning first (not written to disk)…",
        flush=True,
    )
    t0 = time.perf_counter()
    plan = reorder_plan_bottom_up(plan_task(task))
    elapsed = time.perf_counter() - t0
    return (
        plan,
        f"in-memory ({len(plan.actions)} actions, planned in {elapsed:.2f}s, bottom-up)",
    )


# Execute a plan and track the progress
def execute_tracked(
    bridge: _BridgeLike,
    plan: PlanResult,
    *,
    viewer: _ViewerLike | None = None,
) -> tuple[bool, int, str]:

    # Get the total number of actions
    total = len(plan.actions)

    # Execute each action
    for index, action in enumerate(plan.actions):

        # Raise an error if the viewer is closed
        if viewer is not None and not viewer.is_running():
            print("  Viewer closed — aborting.", flush=True)
            return False, index, action.name

        # Execute the action
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            ok, _ = bridge.execute_action(action.name, *action.params)

        # Print the status of the action
        status = "SUCCESS" if ok else "FAIL"
        print(
            f"  Action {index + 1}/{total} {action.name} {status}",
            flush=True,
        )

        # Raise an error if the action failed
        if not ok:
            return False, index, action.name

    # Return the success status, the number of actions executed, and an empty string
    return True, len(plan.actions), ""

# Show a task in the MuJoCo viewer (saved plan or in-memory plan)
def show_task(tier: int, task_id: int) -> int:

    # Get the path to the ground truth file
    yaml_path = ground_truth_path(tier, task_id)

    # Raise an error if the ground truth file is not found
    if not yaml_path.is_file():
        print(f"ERROR: ground-truth YAML not found: {yaml_path}", file=sys.stderr)
        return 1

    # Load the task from the YAML file
    task = load_task(yaml_path)

    # Load a saved plan, or plan once in memory if none exists
    plan, plan_label = resolve_plan(task, tier, task_id)

    # Raise an error if the plan is unsolved or empty
    if not plan.solved or not plan.actions:
        print(
            f"ERROR: plan is unsolved or empty (status={plan.status!r})",
            file=sys.stderr,
        )
        return 1

    # Create a table frame
    frame = TableFrame()

    # Print the task information
    print(f"tier{tier} task_{task_id:03d}")
    print(f"  YAML  {yaml_path}")
    print(f"  plan  {plan_label}", flush=True)
    print("  Opening MuJoCo viewer (close the window to exit)…", flush=True)

    # Create a temporary directory for the environment
    with tempfile.TemporaryDirectory(prefix=f"show_t{tier}_{task_id:03d}_") as tmp:

        # Build the environment
        env = build_lego_env(task, template_dir=Path(tmp))
        try:

            # Place the initial blocks
            place_initial_blocks(env, task, frame=frame)

            # Create the motion planner
            motion = RRTStar(env)

            # Duplo-tuned grasp height so fingers close around the brick
            grasp = make_lego_grasp_planner(table_z=frame.surface_z)
            executor = _ShowcasePickPlaceExecutor(
                env,
                motion,
                grasp,
                use_attachment=True,
                max_plan_iters=5000,
                settle_steps=10,
            )

            # Create the bridge
            bridge = cast(
                _BridgeLike,
                make_lego_bridge(env, task, executor=executor, frame=frame),
            )

            # Launch the viewer
            with env.launch_viewer() as viewer:
                # Red translucent box is the hand collision capsule — hide it.
                _hide_viewer_clutter(env, viewer)

                # Rest the environment
                env.rest(0.12)
                if not viewer.is_running():
                    return 0

                # Execute the plan and track the progress
                t0 = time.perf_counter()
                ok, actions_ok, failed = execute_tracked(
                    bridge, plan, viewer=viewer
                )

                # Calculate the elapsed time
                elapsed = time.perf_counter() - t0

                # Print the result
                if ok:
                    print(
                        f"  Done OK  actions={actions_ok}/{len(plan.actions)}"
                        f"  t={elapsed:.2f}s",
                        flush=True,
                    )
                else:
                    detail = f" fail={failed}" if failed else ""
                    print(
                        f"  Done FAIL  actions={actions_ok}/{len(plan.actions)}"
                        f"{detail}  t={elapsed:.2f}s",
                        flush=True,
                    )

                print("  Close the viewer window to exit.", flush=True)

                # Wait for the viewer to close
                while viewer.is_running():
                    _hide_viewer_clutter(env, viewer)
                    viewer.sync()
                    time.sleep(0.05)

        finally:
            # Close the environment
            close = getattr(env, "close", None)
            if callable(close):
                _ = close()

    return 0


def main(argv: list[str] | None = None) -> int:

    # Parse the command line arguments
    parser = argparse.ArgumentParser(description=__doc__)

    # Add the task argument
    _ = parser.add_argument(
        "--task",
        required=True,
        metavar="TIER/TASK",
        help="Single task, e.g. 1/1 or tier2/task_005",
    )
    ns = parser.parse_args(argv)

    # Parse the task reference
    try:
        tier, task_id = parse_task_ref(cast(str, ns.task))
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    # Raise an error if the ground truth root is missing
    if not GROUND_TRUTH_ROOT.is_dir():
        print(
            f"ERROR: dataset ground_truth not found: {GROUND_TRUTH_ROOT}",
            file=sys.stderr,
        )
        return 1

    return show_task(tier, task_id)


if __name__ == "__main__":
    raise SystemExit(main())
