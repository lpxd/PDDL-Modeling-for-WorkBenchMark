"""Shared paths, engine configs, and result schema for the planner study.

This extension only *imports* from ``workbenchmark_pddl``; it never edits the
main project, ``../tampanda`` or ``../dataset``. All outputs go to
``results/extension/``.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# extensions/planner_study/common.py -> project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT.parent / "dataset"
GROUND_TRUTH_ROOT = DATASET_ROOT / "ground_truth"
EXT_RESULTS = PROJECT_ROOT / "results" / "extension"
PLANNERS_DIR = EXT_RESULTS / "planners"

# Engine configs: label used in result paths -> UPF OneshotPlanner engine name.
# Labels are stable identifiers; the description says what the engine really does.
#   pyperplan      weighted A* + hadd (satisficing, pure Python)
#   pyperplan-opt  A* + LM-cut (optimal, pure Python)
#   fd-lama        Fast Downward, alias lama-first (satisficing)
#   fd-lmcut       Fast Downward, A* + LM-cut (optimal)
# SymK and Tamer are not usable on native Windows here (see README).
ENGINES: dict[str, str] = {
    "pyperplan": "pyperplan",
    "pyperplan-opt": "pyperplan-opt",
    "fd-lama": "fast-downward",
    "fd-lmcut": "fast-downward-opt",
}

# Same for all engines; a task that hits it is recorded as "timeout", not failure.
DEFAULT_TIMEOUT_S = 600.0

# Per-task artifact filenames (plan.json matches the main project's format).
PLAN_FILENAME = "plan.json"
METRICS_FILENAME = "planning_metrics.json"
SUMMARY_JSONL = "planning_summary.jsonl"
SUMMARY_CSV = "planning_summary.csv"

# Statuses: solved / timeout / failed (planner ran but returned no plan or crashed).
STATUS_SOLVED = "solved"
STATUS_TIMEOUT = "timeout"
STATUS_FAILED = "failed"

RECORD_FIELDS = [
    "planner",
    "engine",
    "tier",
    "task_id",
    "status",
    "upf_status",
    "plan_length",
    "planning_time_s",
    "wall_s",
    "timeout_s",
    "plan_valid",
    "yaml",
    "detail",
    "domain",
    "plan_valid_patched",
    "rule_feasible",
]


# One planning attempt; this is what planning_metrics.json holds.
@dataclass(frozen=True)
class PlanningRecord:
    # Config label (key of ENGINES).
    planner: str
    # UPF engine name that was called.
    engine: str
    tier: int
    task_id: int
    # solved / timeout / failed
    status: str
    # Raw UPF status string (or "" for a hard timeout / crash).
    upf_status: str
    plan_length: int
    # Same definition as scripts/eval_planning.py: render + parse + solve.
    # None for timeouts (censored: the true time is > timeout_s).
    planning_time_s: float | None
    # Parent-side wall time including interpreter start-up and imports.
    wall_s: float
    timeout_s: float
    # sequential_plan_validator verdict on the returned plan; None if no plan.
    plan_valid: bool | None
    # Dataset-relative task file, e.g. "tier1/task_001.yaml".
    yaml: str
    # Error text for failed runs, else "".
    detail: str
    # Phase 4: which domain was planned with ("original", "quantified", "closing", "prereq"),
    # the validator's verdict on the raw plan under that domain, and whether the plan
    # satisfies the geometric rule (no brick placed after one hanging over it).
    domain: str = "original"
    plan_valid_patched: bool | None = None
    rule_feasible: bool | None = None


# Stable short id of a plan's action list: identical plans share one id, so a
# plan found by several engines is executed only once.
def plan_id(actions: list[dict[str, Any]]) -> str:
    import hashlib

    blob = json.dumps([[a["name"], list(a["params"])] for a in actions])
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def ground_truth_path(tier: int, task_id: int) -> Path:
    return GROUND_TRUTH_ROOT / f"tier{tier}" / f"task_{task_id:03d}.yaml"


# results/extension/planners/<planner>/tierN/task_XXX/
def task_dir(root: Path, planner: str, tier: int, task_id: int) -> Path:
    return root / planner / f"tier{tier}" / f"task_{task_id:03d}"


# Atomic write so a kill mid-write cannot corrupt an artifact.
def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


# Parse "3/20", "tier3/task_020" or "tier3/20" into (tier, task_id).
def parse_task_ref(raw: str) -> tuple[int, int]:
    m = re.fullmatch(
        r"\s*(?:tier)?(\d+)\s*[/,]\s*(?:task_?)?(\d+)\s*", raw.lower().replace("\\", "/")
    )
    if not m:
        raise ValueError(f"expected TIER/TASK like 3/20 or tier3/task_020, got {raw!r}")
    tier, task_id = int(m.group(1)), int(m.group(2))
    if tier < 1 or task_id < 1:
        raise ValueError(f"tier and task_id must be >= 1, got {raw!r}")
    return tier, task_id


# Independent check of a plan with UPF's sequential_plan_validator.
def validate_plan(task: Any, plan: Any) -> bool:
    from unified_planning.engines import ValidationResultStatus
    from unified_planning.io import PDDLReader
    from unified_planning.plans import ActionInstance, SequentialPlan
    from unified_planning.shortcuts import ObjectExp, PlanValidator

    from workbenchmark_pddl.problem import domain_path, render_problem

    problem = PDDLReader().parse_problem_string(
        domain_path().read_text(encoding="utf-8"),
        render_problem(task, problem_name="validate"),
    )
    steps = [
        ActionInstance(
            problem.action(a.name),
            tuple(ObjectExp(problem.object(p)) for p in a.params),
        )
        for a in plan.actions
    ]
    with PlanValidator(name="sequential_plan_validator") as validator:
        result = validator.validate(problem, SequentialPlan(steps))
    return result.status == ValidationResultStatus.VALID


# Does the plan (list of {"name","params"}) place bricks in an order that satisfies
# the geometric rule? (No brick placed after a brick hanging over its goal cell.)
def plan_rule_feasible(task: Any, actions: list[dict[str, Any]]) -> bool:
    from brick_geometry import overhang_map

    over = overhang_map(task)
    placed: set[str] = set()
    for a in actions:
        if a["name"] == "pick":
            continue
        brick = a["params"][1].removeprefix("b_")
        if over[brick] & placed:
            return False
        placed.add(brick)
    return True


# UPF verdict on a RAW plan (with the extra slot parameters of the closing/prereq encodings)
# under the patched domain.
def validate_patched(task: Any, encoding: str, raw_actions: list[dict[str, Any]]) -> bool:
    import patched_pddl as pp
    from unified_planning.engines import ValidationResultStatus
    from unified_planning.plans import ActionInstance, SequentialPlan
    from unified_planning.shortcuts import ObjectExp, PlanValidator

    problem = pp.parse_patched(task, encoding, "validate")
    steps = [
        ActionInstance(problem.action(a["name"]), tuple(ObjectExp(problem.object(p)) for p in a["params"]))
        for a in raw_actions
    ]
    with PlanValidator(name="sequential_plan_validator") as validator:
        return validator.validate(problem, SequentialPlan(steps)).status == ValidationResultStatus.VALID


# Plan one YAML task with one UPF engine; runs inside the worker subprocess.
# Timing matches scripts/eval_planning.py (load_task outside, render+parse+solve inside).
# ``domain``: "original" (the project's domain) or a patched encoding (patched_pddl.py).
def solve_task(engine: str, yaml_path: Path, domain: str = "original") -> dict[str, Any]:
    from unified_planning.shortcuts import get_environment

    from workbenchmark_pddl.planner import plan_result_to_dict, plan_task, solve_problem
    from workbenchmark_pddl.task import load_task

    get_environment().credits_stream = None

    task = load_task(yaml_path)
    if domain == "original":
        t0 = time.perf_counter()
        result = plan_task(task, planner_name=engine)
        elapsed = time.perf_counter() - t0
        actions = plan_result_to_dict(result)["actions"]
        valid = validate_plan(task, result) if result.solved else None
        return {
            "upf_status": result.status,
            "solved": result.solved,
            "planning_time_s": round(elapsed, 4),
            "plan_valid": valid,
            "plan_valid_patched": None,
            "rule_feasible": plan_rule_feasible(task, actions) if result.solved else None,
            "plan": plan_result_to_dict(result),
            "plan_raw": None,
        }

    import patched_pddl as pp
    from workbenchmark_pddl.planner import PlanAction, PlanResult

    t0 = time.perf_counter()
    problem = pp.parse_patched(task, domain)
    result = solve_problem(problem, planner_name=engine)
    elapsed = time.perf_counter() - t0
    raw = plan_result_to_dict(result)
    trimmed_actions = pp.strip_slots(raw["actions"]) if domain in pp.SLOT_ENCODINGS else raw["actions"]
    trimmed = PlanResult(result.solved, result.status, tuple(PlanAction(a["name"], tuple(a["params"])) for a in trimmed_actions))
    plan_out = {"solved": result.solved, "status": result.status, "actions": trimmed_actions}
    return {
        "upf_status": result.status,
        "solved": result.solved,
        "planning_time_s": round(elapsed, 4),
        # trimmed plan against the ORIGINAL domain (this is what gets executed)
        "plan_valid": validate_plan(task, trimmed) if result.solved else None,
        # raw plan against the patched domain
        "plan_valid_patched": validate_patched(task, domain, raw["actions"]) if result.solved else None,
        "rule_feasible": plan_rule_feasible(task, trimmed_actions) if result.solved else None,
        "plan": plan_out,
        "plan_raw": raw if domain in pp.SLOT_ENCODINGS else None,
    }
