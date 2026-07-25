"""Integration tests: YAML → PDDL → pyperplan finds a plan."""

from pathlib import Path

import pytest

from workbenchmark_pddl.planner import plan_task
from workbenchmark_pddl.task import load_task

# Integration tests: YAML -> PDDL -> pyperplan finds a plan.

# Test the pyperplan planner with the different tiers of dataset tasks.
@pytest.mark.parametrize(
    "fixture_name,min_actions",
    [
        ("tier1_task_001", 4),  # pick, place, pick, stack
        ("tier2_task_001", 6),  # pick, place, pick, stack, pick, stack
        ("tier3_task_001", 8),  # multi-column with at least 4 bricks moved (pick/place/stack)
    ],
)
# Test if the pyperplan planner can solve the dataset task.
# Should find a plan with at least the minimum number of actions.
def test_pyperplan_solves_dataset_task(fixture_name: str, min_actions: int, request):

    # Get the task path from the fixture.
    task_path: Path = request.getfixturevalue(fixture_name)

    # Load the task from the path.
    task = load_task(task_path)

    # Plan the task.
    result = plan_task(task)

    # Assert the plan was found.
    assert result.solved, f"planner failed: {result.status}"

    # Assert the plan has at least the minimum number of actions.
    assert len(result.actions) >= min_actions

    # Assert the first action is a pick.
    assert result.action_names[0] == "pick"

    # Assert the plan only contains pick, place, and stack actions.
    assert set(result.action_names) <= {"pick", "place", "stack"}

    # Assert every goal brick must be placed or stacked at least once.
    moved = {a.params[1] for a in result.actions if a.name in {"place", "stack"}}
    for brick in task.goal_blocks:
        assert f"b_{brick.name}" in moved
