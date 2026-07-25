"""Plan WorkBenchMark tasks with UPF + pyperplan."""

from __future__ import annotations

from dataclasses import dataclass

import unified_planning.shortcuts as up
from unified_planning.io import PDDLReader
from unified_planning.plans import SequentialPlan
from unified_planning.shortcuts import OneshotPlanner

from workbenchmark_pddl.problem import domain_path, render_problem
from workbenchmark_pddl.task import TaskSpec

# Default planner to use for planning.
DEFAULT_PLANNER = "pyperplan"

# A single action in the plan.
@dataclass(frozen=True)
class PlanAction:
    name: str
    params: tuple[str, ...]

# The result of a planning call.
@dataclass(frozen=True)
class PlanResult:

    # Whether the plan was found.
    solved: bool

    # The status of the plan.
    status: str

    # The actions in the plan.
    actions: tuple[PlanAction, ...]

    # The names of the actions in the plan.
    @property
    def action_names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.actions)


# TaskSpec -> PDDL -> pyperplan plan.
def plan_task(
    task: TaskSpec,
    # keyword-only arguments
    *,
    planner_name: str = DEFAULT_PLANNER,
    problem_name: str | None = None,
) -> PlanResult:

    # Render the problem to a PDDL string.
    problem_text = render_problem(task, problem_name=problem_name)

    # Read the domain from the file.
    domain_text = domain_path().read_text(encoding="utf-8")

    # Parse the problem from the PDDL string.
    problem = PDDLReader().parse_problem_string(domain_text, problem_text)

    # Solve the problem.
    return solve_problem(problem, planner_name=planner_name)


# Run a oneshot planner on an already-parsed UPF problem.
def solve_problem(problem, *, planner_name: str = DEFAULT_PLANNER) -> PlanResult:

    # Disable credits stream.
    # This prevent the planner from printing credits to the console.
    # Just a cosmetic thing.
    up.get_environment().credits_stream = None

    # Solve the problem.
    with OneshotPlanner(name=planner_name) as planner:
        result = planner.solve(problem)

    # Get the status of the plan.
    status = str(result.status)

    # Get the plan.
    plan = result.plan

    # If the plan is None, return a failed plan result.
    if plan is None:
        return PlanResult(solved=False, status=status, actions=())

    # Extract the actions from the plan.
    actions = _extract_actions(plan)

    # Return the plan result.
    return PlanResult(solved=True, status=status, actions=actions)


# private function to extract the actions from a plan.
def _extract_actions(plan: SequentialPlan) -> tuple[PlanAction, ...]:

    # Get the action list from the plan.
    # UP uses .actions but some bridges also accept .action_instances.
    action_list = getattr(plan, "actions", None) or plan.action_instances

    # Extract the actions from the plan.
    extracted: list[PlanAction] = []

    # Iterate over the action list.
    for step in action_list:

        # Get the name and parameters of the action.
        name = step.action.name

        # Get the parameters of the action.
        params = tuple(arg.object().name for arg in step.actual_parameters)

        # Append the action to the list.
        extracted.append(PlanAction(name=name, params=params))

    # Return the actions as a tuple.
    return tuple(extracted)
