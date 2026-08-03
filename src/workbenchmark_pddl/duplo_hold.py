"""Pin placed Duplo proxies so the arm cannot shove or interpenetrate stacks.

Box proxies have no studs. After place/stack we lock each brick's world pose
every sim step (same idea as EE attachment). Pick releases the pin first.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# Private function to convert a matrix to a quaternion.
def _mat2quat(mat: FloatArray) -> FloatArray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(mat, dtype=np.float64).reshape(9))
    return quat


# Class to hold the duplo bricks.
# World-pose pins + optional parent welds for stacked free bricks.
class DuploHold:

    # Initialize the duplo hold.
    def __init__(self, env: Any) -> None:
        self.env = env
        self._welds: dict[str, dict[str, Any]] = {}
        self._pins: dict[str, dict[str, Any]] = {}
        env.step = self._step  # type: ignore[method-assign]
        env.duplo_hold = self

    # Private function to get the joint addresses.
    def _joint_addrs(self, body_name: str) -> tuple[int, int]:

        # Get the joint name.
        joint_name = f"{body_name}_freejoint"

        # Get the joint id.
        joint_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

        # If the joint id is less than 0, raise an error.
        if joint_id < 0:
            raise ValueError(f"free joint not found for {body_name!r}")

        # Return the joint addresses.
        return (
            int(self.env.model.jnt_qposadr[joint_id]),
            int(self.env.model.jnt_dofadr[joint_id]),
        )

    # Pin a brick at its current world pose (immovable to the arm).
    def pin(self, brick_name: str) -> None:
       
        # Get the environment.
        env = self.env
        mujoco.mj_forward(env.model, env.data)

        # Pop the weld for the brick.
        self._welds.pop(brick_name, None)

        # Get the joint addresses.
        qadr, vadr = self._joint_addrs(brick_name)

        # Store the pin.
        self._pins[brick_name] = {
            "qadr": qadr,
            "vadr": vadr,
            "pos": env.data.qpos[qadr : qadr + 3].copy(),
            "quat": env.data.qpos[qadr + 3 : qadr + 7].copy(),
        }

    # Keep a brick seated on a parent body (stud lock).
    def weld(self, child_name: str, parent_name: str) -> None:
       
        # Keep child name seated on parent name (stud lock).

        # Get the environment.
        env = self.env

        # Forward the environment.
        mujoco.mj_forward(env.model, env.data)

        # Get the child id.
        child_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, child_name)

        # Get the parent id.
        parent_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, parent_name)
        if child_id < 0 or parent_id < 0:
            raise ValueError(f"unknown bodies for weld: {child_name!r}, {parent_name!r}")

        # Get the joint addresses.
        qadr, vadr = self._joint_addrs(child_name)

        # Get the parent position.
        parent_pos = env.data.xpos[parent_id].copy()

        # Get the parent matrix.
        parent_mat = env.data.xmat[parent_id].reshape(3, 3).copy()

        # Get the child position.
        child_pos = env.data.xpos[child_id].copy()

        # Get the child matrix.
        child_mat = env.data.xmat[child_id].reshape(3, 3).copy()

        # Pop the pin for the child.
        self._pins.pop(child_name, None)

        # Store the weld.
        self._welds[child_name] = {
            "parent": parent_name,
            "parent_id": parent_id,
            "qadr": qadr,
            "vadr": vadr,
            "rel_pos": parent_mat.T @ (child_pos - parent_pos),
            "rel_mat": parent_mat.T @ child_mat,
        }

    # Release a brick before pick so the gripper can carry it.
    def release(self, brick_name: str) -> None:
       
        # Pop the weld for the brick.
        self._welds.pop(brick_name, None)

        # Pop the pin for the brick.
        self._pins.pop(brick_name, None)

    # Release all bricks before pick so the gripper can carry them.
    def release_all(self) -> None:
       
        # Clear the welds.
        self._welds.clear()

        # Clear the pins.
        self._pins.clear()

    # Private function to apply the welds.
    def _apply_welds(self) -> None:
       
        # If there are no welds, return.
        if not self._welds:
            return

        # Get the environment.
        env = self.env

        # Get the pending welds.
        pending = dict(self._welds)

        # For each pending weld.
        for _ in range(len(pending) + 1):

            # If there are no pending welds, break.
            if not pending:
                break

            # Flag to track if progress was made.
            progressed = False

            # For each pending weld.
            for child, weld in list(pending.items()):

                # If the parent is also pending, continue.
                if weld["parent"] in pending:
                    continue

                # Apply the weld.
                self._apply_weld_one(weld)

                # Delete the child from the pending welds.
                del pending[child]

                # Set the progressed flag to True.
                progressed = True

            # If progress was made, forward the environment.
            if progressed:
                mujoco.mj_forward(env.model, env.data)

            # If no progress was made, apply the remaining welds.
            if not progressed:

                # For each pending weld.
                for weld in pending.values():

                    # Apply the weld.
                    self._apply_weld_one(weld)

                # Forward the environment.
                mujoco.mj_forward(env.model, env.data)
                break

    # Private function to apply a single weld.
    def _apply_weld_one(self, weld: dict[str, Any]) -> None:
       
        # Get the environment.
        env = self.env

        # Get the parent id.
        parent_id = int(weld["parent_id"])

        # Get the parent position.
        parent_pos = env.data.xpos[parent_id]

        # Get the parent matrix.
        parent_mat = env.data.xmat[parent_id].reshape(3, 3)

        # Get the new position.
        new_pos = parent_pos + parent_mat @ weld["rel_pos"]

        # Get the new quaternion.
        new_quat = _mat2quat(parent_mat @ weld["rel_mat"])

        # Get the joint addresses.
        qadr = int(weld["qadr"])

        # Get the joint velocities.
        vadr = int(weld["vadr"])

        # Set the new position and quaternion.
        env.data.qpos[qadr : qadr + 3] = new_pos
        env.data.qpos[qadr + 3 : qadr + 7] = new_quat

        # Set the joint velocities to 0.
        env.data.qvel[vadr : vadr + 6] = 0.0

    # Private function to apply the pins.
    def _apply_pins(self) -> None:
       
        # Get the environment.
        env = self.env

        # For each pin.
        for pin in self._pins.values():

            # Get the joint addresses.
            qadr = int(pin["qadr"])
            vadr = int(pin["vadr"])

            # Set the new position and quaternion.
            env.data.qpos[qadr : qadr + 3] = pin["pos"]
            env.data.qpos[qadr + 3 : qadr + 7] = pin["quat"]

            # Set the joint velocities to 0.
            env.data.qvel[vadr : vadr + 6] = 0.0

    # Private function to step the environment.
    def _step(self) -> None:
       
        # Get the environment.
        env = self.env

        # If the environment is attached, apply the attachment.
        if env._attached is not None:
            env._apply_attachment()
            mujoco.mj_forward(env.model, env.data)

        # Apply the welds.
        self._apply_welds()

        # Apply the pins.
        self._apply_pins()

        # Step the environment.
        mujoco.mj_step(env.model, env.data)

        # Re-pin after physics so contact with the arm cannot shove bricks.
        self._apply_welds()
        self._apply_pins()

        # Forward the environment.
        mujoco.mj_forward(env.model, env.data)

        # Increment the simulation time.
        env.sim_time += env.rate.dt

        # If the viewer is not None, sync the viewer.
        if env.viewer is not None:
            # Sync the viewer.
            env.viewer.sync()

        # Sleep the rate.
        env.rate.sleep()

# Enable the duplo hold.
def enable_duplo_hold(env: Any) -> DuploHold:
   
    # Get the existing duplo hold.
    existing = getattr(env, "duplo_hold", None)

    # If the existing duplo hold is a DuploHold, return it.
    if isinstance(existing, DuploHold):
        return existing

    # Create a new duplo hold and return it.
    return DuploHold(env)


# Get the duplo hold.
def get_duplo_hold(env: object) -> DuploHold | None:
   
    # Get the existing duplo hold.
    hold = getattr(env, "duplo_hold", None)

    # If the existing duplo hold is a DuploHold, return it.
    return hold if isinstance(hold, DuploHold) else None
