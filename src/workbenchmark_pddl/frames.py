"""Map WorkBenchMark table-frame poses onto the TAMPanda symbolic table."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from workbenchmark_pddl.execution import half_size_for
from workbenchmark_pddl.task import BrickSpec, Vec3

# Matches tampanda table_symbolic + make_blocks_builder placement.
DEFAULT_ORIGIN_XY = (0.0, 0.4)
DEFAULT_SURFACE_Z = 0.27

# Dataset pick-area floor: every initial_blocks[].pos.z is this value (layer 0).
# Map it to the MuJoCo table top; do not add it on top of surface_z.
PICK_AREA_Z_ORIGIN = 0.0495

# Convert Euler degrees to a quaternion.
def euler_deg_to_quat_wxyz(rotation_deg: Vec3) -> NDArray[np.float64]:
    
    # Convert the Euler degrees to radians.
    roll, pitch, yaw = (math.radians(angle) for angle in rotation_deg)

    # Calculate the sine and cosine of the roll, pitch, and yaw.
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)

    # Calculate the quaternion.
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )

# A frame for the table.
# YAML table frame -> MuJoCo world on the symbolic Franka table.
@dataclass(frozen=True)
class TableFrame:

    # WorkBenchMark ``pos.z`` is brick-bottom height above a local floor.
    # Goals use floor z=0 on the assembly table; pick-area YAML uses
    # PICK_AREA_Z_ORIGIN — pass that as z_origin so bricks sit on the sim table.
    # Body centres add half_z. XY is offset by the TAMPanda table origin.

    # The origin of the table.
    origin_xy: tuple[float, float] = DEFAULT_ORIGIN_XY
    # The surface height of the table.
    surface_z: float = DEFAULT_SURFACE_Z

    # Convert a YAML position to a center position.
    def center_from_yaml(
        self,
        yaml_pos: Vec3,
        half_z: float,
        *,
        z_origin: float = 0.0,
    ) -> NDArray[np.float64]:

        # Convert the YAML position to a numpy array.
        x, y, z = yaml_pos

        # Calculate the center position.
        return np.array(
            [
                self.origin_xy[0] + x,
                self.origin_xy[1] + y,
                self.surface_z + (z - z_origin) + half_z,
            ],
            dtype=np.float64,
        )

    # Convert a pick-area brick specification to sim pose (on the table).
    def pose_from_brick(
        self, brick: BrickSpec
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:

        # Convert the half size to a numpy array.
        half_x, half_y, half_z = half_size_for(brick)
        half = np.array([half_x, half_y, half_z], dtype=np.float64)

        # Pick-area YAML z is relative to PICK_AREA_Z_ORIGIN, not the table top.
        center = self.center_from_yaml(
            brick.pos, half_z, z_origin=PICK_AREA_Z_ORIGIN
        )

        # Convert the rotation to a quaternion.
        quat = euler_deg_to_quat_wxyz(brick.rotation)
        return center, quat, half
