"""MuJoCo scene with Duplo box proxies for a WorkBenchMark task."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from tampanda.environments.franka_env import FrankaEnvironment
from tampanda.planners.grasp_planner import (
    FINGER_TIP_BELOW,
    GraspPlanner,
    GraspType,
)
from tampanda.scenes import ArmSceneBuilder

from workbenchmark_pddl.frames import TableFrame, DEFAULT_SURFACE_Z
from workbenchmark_pddl.geometry import BRICK_SIZE_STUDS
from workbenchmark_pddl.grid import LAYER_HEIGHT_M, STUD_PITCH_M
from workbenchmark_pddl.task import BrickSpec, TaskSpec, Vec3

# YAML colour names -> RGBA for geom override at instantiate time.
COLOR_RGBA: dict[str, list[float]] = {
    "red": [1.0, 0.2, 0.2, 1.0],
    "green": [0.2, 0.8, 0.2, 1.0],
    "blue": [0.2, 0.2, 1.0, 1.0],
    "yellow": [1.0, 1.0, 0.2, 1.0],
    "orange": [1.0, 0.5, 0.2, 1.0],
    "purple": [0.6, 0.2, 0.8, 1.0],
    "white": [0.95, 0.95, 0.95, 1.0],
    "black": [0.15, 0.15, 0.15, 1.0],
    "cyan": [0.2, 0.9, 0.9, 1.0],
    "magenta": [1.0, 0.2, 1.0, 1.0],
}

# Default RGBA for the bricks. (grey)
_DEFAULT_RGBA = [0.7, 0.7, 0.7, 1.0]

# The assets directory.
_ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# Wider than TAMPanda table_symbolic so Tier-3 pick/goal XY stay on the surface.
_TABLE_TEMPLATE = _ASSETS_DIR / "table_workbenchmark.xml"

# Duplo bricks are ~19mm tall. TAMPanda's default table_clearance (25mm) lifts
# top-down grasps above the brick so fingers never close around it.
DUPLO_GRASP_TABLE_CLEARANCE_M = 0.002

# Sink EE slightly so the brick seats between the finger pads.
DUPLO_GRASP_DEPTH_BIAS_M = 0.008

# Typed array of floats.
FloatArray = NDArray[np.float64]


# Typed view of ArmSceneBuilder
# TAMPanda stubs leave these members Unknown.
class _TypedSceneBuilder(Protocol):

    def add_resource(self, name: str, source: str) -> None: ...

    def add_object(
        self,
        type_name: str,
        pos: list[float],
        *,
        name: str | None = None,
        quat: list[float] | None = None,
        rgba: list[float] | None = None,
    ) -> None: ...

    def build_env(self, rate: float = 200.0) -> FrankaEnvironment: ...


# Typed view of FrankaEnvironment pose helpers.
class _TypedSimEnv(Protocol):
    def set_object_pose(
        self, body_name: str, pos: FloatArray, quat: FloatArray | None = None
    ) -> None: ...

    def reset_velocities(self) -> None: ...

    def forward(self) -> None: ...


# Get the asset path for a brick type.
def brick_asset_path(brick_type: str) -> Path:

    # Get the path to the brick asset.
    path = _ASSETS_DIR / f"{brick_type}.xml"

    # If the asset is not found, raise an error.
    if not path.is_file():
        raise FileNotFoundError(f"missing brick asset for {brick_type!r}: {path}")
    return path


# Calculate the half size for a brick type.
def nominal_half_size(brick_type: str) -> Vec3:

    # Get the width and depth of the brick in studs.
    width_studs, depth_studs = BRICK_SIZE_STUDS[brick_type]

    # Calculate the half size.
    return (
        width_studs * STUD_PITCH_M / 2.0,
        depth_studs * STUD_PITCH_M / 2.0,
        LAYER_HEIGHT_M / 2.0,
    )


# GraspPlanner tuned so top-down fingers close around Duplo height, not above it.
def make_lego_grasp_planner(*, table_z: float = DEFAULT_SURFACE_Z) -> object:

    # Create the grasp planner.
    planner = GraspPlanner(

        # Set the table z.
        table_z=table_z,

        # Set the table clearance.
        table_clearance=DUPLO_GRASP_TABLE_CLEARANCE_M,

        # Set the approach distance.
        approach_dist=0.08,

         # Shorter standoff/lift so the hand stays near the brick (showcase).
        lift_height=0.08,
    )

    # Get the generate candidates function.
    generate = planner.generate_candidates

    # Sub Function for generating candidates for a block.
    def generate_candidates(
        block_pos: object,
        block_half_size: object = None,
        block_quat: object = None,
    ) -> list[object]:

        # Generate with Duplo table clearance.
        raw = generate(block_pos, block_half_size, block_quat)

        # List for biased candidates.
        biased: list[object] = []

        # For each candidate.
        for cand in raw:

            # Sink top-down grasps when tips still clear the table after the bias.
            if getattr(cand, "grasp_type", None) in (
                GraspType.TOP_DOWN_X,
                GraspType.TOP_DOWN_Y,
            ):

                # Get the grasp z.
                grasp_z = float(np.asarray(cand.grasp_pos)[2])

                # Calculate the maximum bias.
                max_bias = grasp_z - FINGER_TIP_BELOW - (table_z + 0.001)

                # Calculate the bias.
                bias = min(DUPLO_GRASP_DEPTH_BIAS_M, max(0.0, max_bias))

                # If the bias is greater than 0.0.
                if bias > 0.0:

                    # Calculate the delta.
                    delta = np.array([0.0, 0.0, -bias])

                    # Update the grasp position.
                    cand.grasp_pos = np.asarray(cand.grasp_pos, dtype=np.float64) + delta

                    # Update the approach position.
                    cand.approach_pos = (
                        np.asarray(cand.approach_pos, dtype=np.float64) + delta
                    )

                    # Update the lift position.
                    cand.lift_pos = np.asarray(cand.lift_pos, dtype=np.float64) + delta

            # Add the candidate to the biased list.
            biased.append(cand)

        # Return the biased candidates.
        return biased

    # Set the generate candidates function.
    planner.generate_candidates = generate_candidates  

    # Return the planner.
    return planner


# Create the ArmSceneBuilder with symbolic table + one free box per initial brick.
def make_lego_builder(
    task: TaskSpec,
    *,
    template_dir: Path | None = None,
) -> ArmSceneBuilder:
    _ = template_dir  # kept for call-site compatibility; bricks live under assets/

    # Create the ArmSceneBuilder.
    builder = ArmSceneBuilder()
    scene = cast(_TypedSceneBuilder, cast(object, builder))

    # Add the WorkBenchMark table (covers Tier-3 pick/goal extents).
    scene.add_resource("table_symbolic", str(_TABLE_TEMPLATE))

    # Same solver tuning as the blocks domain (stable contacts for stacking).
    # SceneBuilder has no public setters; setattr avoids private-attr access warnings.
    setattr(
        builder,
        "_options",
        {
            "timestep": "0.005",
            "iterations": "5",
            "ls_iterations": "8",
            "integrator": "implicitfast",
            "solver": "Newton",
            "density": "1.2",
            "viscosity": "0.01",
        },
    )

    # Set the option flags and custom numerics.
    setattr(builder, "_option_flags", {"eulerdamp": "disable"})
    setattr(builder, "_custom_numerics", {"max_contact_points": "12"})

    # Add the symbolic table.
    scene.add_object(
        "table_symbolic",
        name="simple_table",
        pos=[0.0, 0.4, 0.0],
        quat=[1.0, 0.0, 0.0, 0.0],
    )

    # Register the bricks.
    registered: set[str] = set()
    # For each initial brick.
    for brick in task.initial_blocks:
        # If the brick type is not registered.
        if brick.type not in registered:
            scene.add_resource(brick.type, str(brick_asset_path(brick.type)))
            registered.add(brick.type)

        # Calculate the half size.
        hz = nominal_half_size(brick.type)[2]
        # Get the RGBA for the brick color.
        rgba = COLOR_RGBA.get(brick.color.lower(), _DEFAULT_RGBA)
        # Add the brick to the scene.
        scene.add_object(
            brick.type,
            name=brick.name,
            pos=[100.0, 0.0, hz],
            rgba=rgba,
        )

    return builder

# Compile the task scene into a FrankaEnvironment.
def build_lego_env(
    task: TaskSpec, *, rate: float = 200.0, template_dir: Path | None = None
) -> FrankaEnvironment:
    # Compile the task scene into a FrankaEnvironment.
    scene = cast(
        _TypedSceneBuilder,
        cast(object, make_lego_builder(task, template_dir=template_dir)),
    )

    # Build the environment.
    env = scene.build_env(rate=rate)
    return env

# Teleport each brick to its pick-area pose in the sim frame.
def place_initial_blocks(
    env: FrankaEnvironment,
    task: TaskSpec,
    frame: TableFrame | None = None,
) -> None:

    # Teleport each brick to its pick-area pose in the sim frame.
    if frame is None:
        frame = TableFrame()

    sim = cast(_TypedSimEnv, cast(object, env))

    # For each initial brick.
    for brick in task.initial_blocks:
        # Get the center and quaternion for the brick.
        center, quat, _half = frame.pose_from_brick(brick)
        sim.set_object_pose(brick.name, center, quat)

    # Reset the velocities.
    # Otherwise, the bricks will keep moving.
    sim.reset_velocities()

    # Forward the environment.
    # This is needed to update the scene.
    sim.forward()

# Teleport one brick (e.g. after a failed place, for debugging).
def place_brick(
    env: FrankaEnvironment,
    brick: BrickSpec,
    frame: TableFrame | None = None,
) -> None:

    # If the frame is not provided, use the default table frame.
    if frame is None:
        frame = TableFrame()

    # Get the center and quaternion for the brick.
    center, quat, _ = frame.pose_from_brick(brick)
    
    # Set the object pose.
    cast(_TypedSimEnv, cast(object, env)).set_object_pose(brick.name, center, quat)
