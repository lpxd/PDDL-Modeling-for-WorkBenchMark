"""Map WorkBenchMark table-frame poses onto the TAMPanda symbolic table."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from workbenchmark_pddl.execution import half_size_for
from workbenchmark_pddl.task import BrickSpec, Vec3

# Matches tampanda table_symbolic + make_blocks_builder placement.
DEFAULT_ORIGIN_XY = (0.0, 0.4)
DEFAULT_SURFACE_Z = 0.27

# Convert Euler degrees to a quaternion.
def euler_deg_to_quat_wxyz(rotation_deg: Vec3) -> NDArray[np.float64]:
    
    # Convert the Euler degrees to radians.
    roll, pitch, yaw = np.deg2rad(np.asarray(rotation_deg, dtype=float))

    # Calculate the sine and cosine of the roll, pitch, and yaw.
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)

    # Calculate the quaternion.
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )

# A frame for the table.
# YAML table frame -> MuJoCo world on the symbolic Franka table.
@dataclass(frozen=True)
class TableFrame:

    # WorkBenchMark ``pos.z`` is treated as brick-bottom height above the table
    # surface (goal bricks sit at z=0, z=LAYER_HEIGHT, …). Body centres add half_z. 
    # XY is offset by the table body origin used in TAMPanda.

    # The origin of the table.
    origin_xy: tuple[float, float] = DEFAULT_ORIGIN_XY
    # The surface height of the table.
    surface_z: float = DEFAULT_SURFACE_Z

    # Convert a YAML position to a center position.
    def center_from_yaml(self, yaml_pos: Vec3, half_z: float) -> NDArray[np.float64]:

        # Convert the YAML position to a numpy array.
        x, y, z = yaml_pos

        # Calculate the center position.
        return np.array(
            [
                self.origin_xy[0] + x,
                self.origin_xy[1] + y,
                self.surface_z + z + half_z,
            ],
            dtype=float,
        )

    # Convert a brick specification to a center position, quaternion, and half size.
    def pose_from_brick(
        self, brick: BrickSpec
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:

        # Convert the half size to a numpy array.
        half = np.asarray(half_size_for(brick), dtype=float)

        # Convert the position to a center position.
        center = self.center_from_yaml(brick.pos, float(half[2]))

        # Convert the rotation to a quaternion.
        quat = euler_deg_to_quat_wxyz(brick.rotation)
        return center, quat, half
