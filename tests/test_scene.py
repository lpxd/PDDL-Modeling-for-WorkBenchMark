"""Tests for Duplo proxy scene + table-frame mapping."""

from pathlib import Path

import numpy as np
import pytest

from workbenchmark_pddl.frames import TableFrame, euler_deg_to_quat_wxyz
from workbenchmark_pddl.scene import (
    brick_asset_path,
    build_lego_env,
    make_lego_builder,
    nominal_half_size,
    place_initial_blocks,
)
from workbenchmark_pddl.task import load_task

# Test the nominal half size for a 4x2 brick.
# Should be 32mm x 16mm x 9.55mm.
def test_nominal_half_size_4x2():

    # Calculate the half size.
    hx, hy, hz = nominal_half_size("brick_4x2")

    # Assert the half size is correct.
    assert hx == pytest.approx(0.032)
    assert hy == pytest.approx(0.016)
    assert hz == pytest.approx(0.0191 / 2)

# Test the brick asset has a freejoint and the correct size.
def test_brick_asset_has_freejoint_and_size():

    xml = brick_asset_path("brick_4x2").read_text(encoding="utf-8")

    # Assert the freejoint and size are correct.
    assert '_freejoint' in xml
    assert 'type="box"' in xml
    assert "0.032" in xml
    assert "0.016" in xml

# Test the table frame maps the goal bottom brick to the correct position.
# Should be at the surface + half_z.
def test_table_frame_maps_goal_bottom_brick():

    # Create the table frame.
    frame = TableFrame()

    # Goal bottom brick: yaml z=0 → centre at surface + half_z
    center = frame.center_from_yaml((0.0, 0.148, 0.0), half_z=0.00955)
    np.testing.assert_allclose(center, [0.0, 0.548, 0.27955])


def test_table_frame_maps_pick_area_onto_surface():
    frame = TableFrame()
    # Dataset pick-area floor z=0.0495 → brick bottom on the MuJoCo table.
    center = frame.center_from_yaml(
        (0.3776, -0.114, 0.0495),
        half_z=0.00955,
        z_origin=0.0495,
    )
    np.testing.assert_allclose(center, [0.3776, 0.286, 0.27955])

# Test the make_lego_builder registers the YAML names and creates the brick templates.
# Should register the table and the bricks.
def test_make_lego_builder_registers_yaml_names(tier1_task_001: Path, tmp_path: Path):

    # Load the task.
    task = load_task(tier1_task_001)

    # Create the builder.
    builder = make_lego_builder(task, template_dir=tmp_path)

    # Get the names of the objects.
    names = {obj.name for obj in builder._objects}

    # Assert the table and bricks are registered.
    assert "simple_table" in names
    assert "4x2_brick_1" in names
    assert "4x2_brick_2" in names
    assert brick_asset_path("brick_4x2").is_file()

# Test the build_lego_env and place_initial_blocks place the initial bricks at the correct positions.
# Should place the bricks at the correct positions.
def test_build_and_place_tier1(tier1_task_001: Path, tmp_path: Path):

    # Import the mujoco and tampanda modules.
    pytest.importorskip("mujoco")
    pytest.importorskip("tampanda")

    # Load the task.
    task = load_task(tier1_task_001)

    # Create the table frame.
    frame = TableFrame()

    # Build the environment.
    env = build_lego_env(task, template_dir=tmp_path)
    place_initial_blocks(env, task, frame=frame)

    # For each initial brick.
    for brick in task.initial_blocks:

        # Get the expected position and quaternion.
        expected, _quat, half = frame.pose_from_brick(brick)

        # Get the position and quaternion of the brick.
        pos, quat = env.get_object_pose(brick.name)

        # Assert the position and quaternion are correct.
        assert pos is not None
        np.testing.assert_allclose(pos, expected, atol=1e-6)
        np.testing.assert_allclose(quat, euler_deg_to_quat_wxyz(brick.rotation), atol=1e-6)
        half_env = env.get_object_half_size(brick.name)
        np.testing.assert_allclose(half_env, half, atol=1e-4)
