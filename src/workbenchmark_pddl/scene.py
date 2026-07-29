"""MuJoCo scene with Duplo box proxies for a WorkBenchMark task."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from tampanda.environments.franka_env import FrankaEnvironment
from tampanda.scenes import TABLE_SYMBOLIC_TEMPLATE, ArmSceneBuilder

from workbenchmark_pddl.frames import TableFrame
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

FloatArray = NDArray[np.float64]


# Typed view of ArmSceneBuilder — TAMPanda stubs leave these members Unknown.
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


# Calculate the  half size for a brick type.
def nominal_half_size(brick_type: str) -> Vec3:

    # Get the width and depth of the brick in studs.
    width_studs, depth_studs = BRICK_SIZE_STUDS[brick_type]
    return (
        width_studs * STUD_PITCH_M / 2.0,
        depth_studs * STUD_PITCH_M / 2.0,
        LAYER_HEIGHT_M / 2.0,
    )

# Create the XML for a brick template.
def brick_template_xml(half_size: Vec3, *, mass: float | None = None) -> str:
    
    # Get the half sizes.
    hx, hy, hz = half_size

    # Calculate the mass if it is not provided.
    if mass is None:
        # Calculate the volume of the brick.
        volume = (2.0 * hx) * (2.0 * hy) * (2.0 * hz)
        mass = max(0.02, volume * 800.0)  # ~Duplo-ish density
    # Solid-box principal inertias about centre (sides = 2*half).
    ixx = (1.0 / 3.0) * mass * (hy * hy + hz * hz)
    iyy = (1.0 / 3.0) * mass * (hx * hx + hz * hz)
    izz = (1.0 / 3.0) * mass * (hx * hx + hy * hy)

    # Return the XML for the brick template.
    return (
        '<body pos="0 0 0">\n'
        '  <joint name="_freejoint" type="free"/>\n'
        f'  <inertial mass="{mass:.6g}" pos="0 0 0" '
        f'diaginertia="{ixx:.6g} {iyy:.6g} {izz:.6g}"/>\n'
        f'  <geom name="_geom" type="box" size="{hx:.6g} {hy:.6g} {hz:.6g}" '
        'rgba="0.8 0.2 0.2 1" solimp="0.99 0.99 0.01" solref="0.02 1"/>\n'
        "</body>\n"
    )

# Create the ArmSceneBuilder with symbolic table + one free box per initial brick.
def make_lego_builder(
    task: TaskSpec,
    *,
    template_dir: Path | None = None,
) -> ArmSceneBuilder:

    # Create the template directory if it is not provided.
    if template_dir is None:
        template_dir = Path(tempfile.mkdtemp(prefix="lego_bricks_"))
    # Otherwise, use the provided template directory.
    else:
        template_dir = Path(template_dir)
        template_dir.mkdir(parents=True, exist_ok=True)

    # Create the ArmSceneBuilder.
    builder = ArmSceneBuilder()
    scene = cast(_TypedSceneBuilder, cast(object, builder))

    # Add the symbolic table template.
    scene.add_resource("table_symbolic", str(TABLE_SYMBOLIC_TEMPLATE))

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
            # Calculate the half size.
            half = nominal_half_size(brick.type)
            # Create the path to the template file.
            path = template_dir / f"{brick.type}.xml"
            _ = path.write_text(brick_template_xml(half), encoding="utf-8")
            scene.add_resource(brick.type, str(path.resolve()))
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
    return scene.build_env(rate=rate)

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
