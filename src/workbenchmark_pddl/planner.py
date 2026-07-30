"""Plan WorkBenchMark tasks with UPF + pyperplan."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, cast

from workbenchmark_pddl.problem import domain_path, render_problem
from workbenchmark_pddl.task import TaskSpec

# Default planner to use for planning.
DEFAULT_PLANNER = "pyperplan"


# UPF has no type stubs; Protocols keep basedpyright quiet at the call sites.
class _UPEnvironment(Protocol):
    credits_stream: object | None

# UPF Shortcuts Protocol.
class _UPShortcuts(Protocol):
    def get_environment(self) -> _UPEnvironment: ...


# UPF Plan Generation Result Protocol.
class _PlanGenResult(Protocol):
    status: object
    plan: object | None


# UPF Oneshot Engine Protocol.
class _OneshotEngine(Protocol):

    def solve(self, problem: object) -> _PlanGenResult: ...

    def __enter__(self) -> _OneshotEngine: ...

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        tb: object,
    ) -> bool | None: ...


# UPF Oneshot Planner Factory Protocol.
class _OneshotPlannerFactory(Protocol):
    def __call__(self, *, name: str) -> object: ...


# UPF PDDL Reader Protocol.
class _PDDLReader(Protocol):
    def parse_problem_string(
        self,
        domain_str: str,
        problem_str: str | None = None,
    ) -> object: ...


# UPF PDDL Reader Factory Protocol.
class _PDDLReaderFactory(Protocol):
    def __call__(self) -> _PDDLReader: ...


# UPF Shortcuts instance.
up = cast(_UPShortcuts, cast(object, import_module("unified_planning.shortcuts")))

# UPF Oneshot Planner instance.
OneshotPlanner = cast(
    _OneshotPlannerFactory,
    cast(
        object,
        getattr(import_module("unified_planning.shortcuts"), "OneshotPlanner"),
    ),
)

# UPF PDDL Reader instance.
PDDLReader = cast(
    _PDDLReaderFactory,
    cast(object, getattr(import_module("unified_planning.io"), "PDDLReader")),
)


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


# JSON-serializable form for eval artifacts / simulation reuse.
def plan_result_to_dict(result: PlanResult) -> dict[str, object]:

    # Return the dictionary representation of the plan result.
    return {
        "solved": result.solved,
        "status": result.status,
        "actions": [
            {"name": action.name, "params": list(action.params)}
            for action in result.actions
        ],
    }


# Inverse of plan_result_to_dict.
def plan_result_from_dict(data: Mapping[str, object]) -> PlanResult:

    # Create a tuple of plan actions from the data.
    actions_raw = data.get("actions", [])
    if not isinstance(actions_raw, list):
        raise TypeError("actions must be a list")
    actions = tuple(_action_from_json(step) for step in cast(list[object], actions_raw))

    # Return the plan result from the data.
    return PlanResult(
        solved=bool(data["solved"]),
        status=str(data["status"]),
        actions=actions,
    )


def _action_from_json(step: object) -> PlanAction:
    if not isinstance(step, dict):
        raise TypeError("each action must be an object")
    step_map = cast(dict[str, object], step)
    params_raw = step_map["params"]
    if not isinstance(params_raw, Sequence) or isinstance(params_raw, (str, bytes)):
        raise TypeError("params must be a sequence")
    return PlanAction(
        name=str(step_map["name"]),
        params=tuple(str(param) for param in params_raw),
    )


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
def solve_problem(problem: object, *, planner_name: str = DEFAULT_PLANNER) -> PlanResult:

    # Disable credits stream.
    # This prevent the planner from printing credits to the console.
    # Just a cosmetic thing.
    up.get_environment().credits_stream = None

    # Solve the problem.
    with cast(_OneshotEngine, OneshotPlanner(name=planner_name)) as planner:
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
def _extract_actions(plan: object) -> tuple[PlanAction, ...]:

    # Get the action list from the plan.
    # UP uses .actions but some bridges also accept .action_instances.
    actions_attr = cast(object, getattr(plan, "actions", None))
    action_list: object = (
        actions_attr
        if actions_attr is not None
        else cast(object, getattr(plan, "action_instances", None))
    )
    if action_list is None:
        raise TypeError("plan has neither .actions nor .action_instances")

    # Extract the actions from the plan.
    extracted: list[PlanAction] = []

    # Iterate over the action list.
    for step in _iter_steps(action_list):

        # Get the name and parameters of the action.
        action_obj = cast(object, getattr(step, "action"))
        name = str(cast(object, getattr(action_obj, "name")))

        # Get the parameters of the action.
        params = tuple(
            _param_name(arg)
            for arg in _iter_steps(cast(object, getattr(step, "actual_parameters")))
        )

        # Append the action to the list.
        extracted.append(PlanAction(name=name, params=params))

    # Return the actions as a tuple.
    return tuple(extracted)


def _iter_steps(action_list: object) -> Iterator[object]:
    return iter(cast(Iterable[object], action_list))


class _ParamArg(Protocol):
    def object(self) -> object: ...


def _param_name(arg: object) -> str:
    obj = cast(_ParamArg, arg).object()
    return str(cast(object, getattr(obj, "name")))
