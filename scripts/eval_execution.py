#!/usr/bin/env python
"""
Batch execution evaluation (MuJoCo via DomainBridge).

Loads saved ``plan.json`` from planning eval (no re-planning), runs PickPlace
in TAMPanda, and writes per-task ``execution_metrics.json`` immediately
(abort-safe), plus rolling summary JSONL/CSV.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

from workbenchmark_pddl.bridge import make_lego_bridge, snap_arm_home
from workbenchmark_pddl.frames import TableFrame
from workbenchmark_pddl.planner import PlanResult, plan_result_from_dict
from workbenchmark_pddl.scene import (
    build_lego_env,
    make_lego_grasp_planner,
    place_initial_blocks,
)
from workbenchmark_pddl.task import load_task


from tampanda.planners.pick_place import PickPlaceExecutor
from tampanda.planners.rrt_star import RRTStar

# Define the project root and dataset root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "dataset"

# Define the ground truth root and default output directory.
GROUND_TRUTH_ROOT = DATASET_ROOT / "ground_truth"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results"

# Per-task artifact filenames.
PLAN_FILENAME = "plan.json"
METRICS_FILENAME = "execution_metrics.json"
SUMMARY_JSONL = "execution_summary.jsonl"
SUMMARY_CSV = "execution_summary.csv"

# Metrics field order for summaries.
FIELDNAMES = [
    "tier",
    "task_id",
    "success",
    "status",
    "plan_length",
    "actions_ok",
    "failed_action",
    "execution_time_s",
    "path",
    "plan_path",
]


# Minimal bridge surface used by this script.
class _BridgeLike(Protocol):

    # The environment.
    env: object

    # Execute an action.
    def execute_action(
        self, name: str, *params: object
    ) -> tuple[bool, object]: ...


# Define the dataclass for one execution row.
@dataclass(frozen=True)
class ExecutionRow:
    # The tier of the task.
    tier: int
    # The id of the task.
    task_id: int
    # Whether every plan action succeeded in sim.
    success: bool
    # Short status string (OK / FAIL / SKIP / ERROR:...).
    status: str
    # Number of actions in the loaded plan.
    plan_length: int
    # How many actions completed before failure (or all if success).
    actions_ok: int
    # Failed action name, or empty.
    failed_action: str
    # Wall time for the MuJoCo execution.
    execution_time_s: float
    # Path to the ground-truth YAML.
    path: str
    # Path to the loaded plan.json.
    plan_path: str


# Define the function to get the ground truth path for a task.
def ground_truth_path(tier: int, task_id: int) -> Path:
    return GROUND_TRUTH_ROOT / f"tier{tier}" / f"task_{task_id:03d}.yaml"


# Per-task output directory: results/tierN/task_XXX/
def task_out_dir(out_dir: Path, tier: int, task_id: int) -> Path:
    return out_dir / f"tier{tier}" / f"task_{task_id:03d}"


# Atomic JSON write so a kill mid-write does not corrupt the artifact.
def write_json_atomic(path: Path, payload: dict[str, object]) -> None:

    # Create the parent directory if it doesn't exist.
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create a temporary file with the same suffix.
    tmp = path.with_suffix(path.suffix + ".tmp")

    # Write the JSON object to the temporary file.
    _ = tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Replace the original file with the temporary file.
    _ = tmp.replace(path)


# Load a metrics JSON object from disk.
def _load_metrics(path: Path) -> dict[str, object]:

    # Load the JSON object from the file.
    raw = cast(object, json.loads(path.read_text(encoding="utf-8")))

    # Check if the object is a dictionary.
    if not isinstance(raw, dict):
        raise TypeError(f"expected object in {path}")

    # Return the dictionary.
    return cast(dict[str, object], raw)


# Rewrite JSONL from all on-disk execution_metrics.json (abort-safe).
def rewrite_summary_jsonl(out_dir: Path) -> Path:

    # Get the path to the summary JSONL file.
    jsonl_path = out_dir / SUMMARY_JSONL

    # Create the parent directory if it doesn't exist.
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize the list of lines.
    lines: list[str] = []

    # Loop through the metrics files.
    for metrics_path in sorted(out_dir.glob("tier*/task_*/" + METRICS_FILENAME)):

        # Load the metrics from the file.
        data = _load_metrics(metrics_path)

        # Append the line to the list.
        lines.append(json.dumps({key: data[key] for key in FIELDNAMES}))

    # Write the lines to the file.
    _ = jsonl_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )

    # Return the path to the summary JSONL file.
    return jsonl_path


# Rewrite CSV from all on-disk execution_metrics.json (abort-safe).
def rewrite_summary_csv(out_dir: Path) -> Path:

    # Initialize the list of rows.
    rows: list[dict[str, object]] = []

    # Loop through the metrics files.
    for metrics_path in sorted(out_dir.glob("tier*/task_*/" + METRICS_FILENAME)):

        # Load the metrics from the file.
        data = _load_metrics(metrics_path)

        # Append the row to the list.
        rows.append({key: data[key] for key in FIELDNAMES})

    # Get the path to the summary CSV file.
    csv_path = out_dir / SUMMARY_CSV

    # Create the parent directory if it doesn't exist.
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the rows to the file.
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

        # Write the header.
        writer.writeheader()

        # Write the rows.
        for row in rows:
            writer.writerow(row)

    # Return the path to the summary CSV file.
    return csv_path


# Persist metrics for one task, then refresh summaries from disk.
def write_task_artifacts(out_dir: Path, row: ExecutionRow) -> Path:

    # Get the task directory.
    task_dir = task_out_dir(out_dir, row.tier, row.task_id)

    # Write the metrics to the file.
    write_json_atomic(task_dir / METRICS_FILENAME, asdict(row))

    # Rewrite the summary JSONL file.
    _ = rewrite_summary_jsonl(out_dir)

    # Rewrite the summary CSV file.
    _ = rewrite_summary_csv(out_dir)

    # Return the path to the task directory.
    return task_dir


# Load a PlanResult from planning eval artifacts.
def load_saved_plan(plan_path: Path) -> PlanResult:

    # Load the JSON object from the file.
    raw = cast(object, json.loads(plan_path.read_text(encoding="utf-8")))

    # Check if the object is a dictionary.
    if not isinstance(raw, dict):
        raise TypeError(f"expected object in {plan_path}")

    # Return the plan result.
    return plan_result_from_dict(cast(dict[str, object], raw))


# Run plan actions on the bridge; return success, count ok, failed name.
def execute_tracked(
    bridge: _BridgeLike,
    plan: PlanResult,
    *,
    return_home: bool = True,
) -> tuple[bool, int, str]:

    total = len(plan.actions)

    # Loop through the actions in the plan.
    for index, action in enumerate(plan.actions):
        kind = action.name
        # Suppress noisy PickPlaceExecutor prints; we log one line per action.
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            ok, _ = bridge.execute_action(action.name, *action.params)

        status = "SUCCESS" if ok else "FAIL"
        print(
            f"  Action {index + 1}/{total} {kind} {status}",
            flush=True,
        )

        # Check if the action failed.
        if not ok:
            return False, index, action.name

        # Snap the arm home if requested.
        if return_home:
            snap_arm_home(bridge.env)

    # Return success, number of actions completed, and empty string.
    return True, len(plan.actions), ""


# Execute one task in MuJoCo using a saved plan.
def eval_one(tier: int, task_id: int, out_dir: Path) -> ExecutionRow:

    # Get the path to the ground truth YAML file.
    yaml_path = ground_truth_path(tier, task_id)

    # Check if the file exists.
    if not yaml_path.is_file():
        raise FileNotFoundError(yaml_path)

    # Get the path to the plan file.
    plan_path = task_out_dir(out_dir, tier, task_id) / PLAN_FILENAME

    # Check if the file exists.
    if not plan_path.is_file():
        return ExecutionRow(
            tier=tier,
            task_id=task_id,
            success=False,
            status="SKIP:NO_PLAN",
            plan_length=0,
            actions_ok=0,
            failed_action="",
            execution_time_s=0.0,
            path=str(yaml_path),
            plan_path=str(plan_path),
        )

    # Load the plan from the file.
    plan = load_saved_plan(plan_path)
    if not plan.solved or not plan.actions:
        return ExecutionRow(
            tier=tier,
            task_id=task_id,
            success=False,
            status="SKIP:UNSOLVED_PLAN",
            plan_length=len(plan.actions),
            actions_ok=0,
            failed_action="",
            execution_time_s=0.0,
            path=str(yaml_path),
            plan_path=str(plan_path),
        )

    # Load the task from the YAML file.
    task = load_task(yaml_path)

    # Create a table frame.
    frame = TableFrame()

    # Create a temporary directory with a prefix.
    with tempfile.TemporaryDirectory(prefix=f"exec_t{tier}_{task_id:03d}_") as tmp:

        # Build the environment.
        env = build_lego_env(task, template_dir=Path(tmp))


        try:
            # Place the initial blocks.
            place_initial_blocks(env, task, frame=frame)

            # Create a RRT star motion planner.
            motion = RRTStar(env)

            # Duplo-tuned grasp height so fingers close around the brick
            grasp = make_lego_grasp_planner(table_z=frame.surface_z)

            # Create a pick place executor.
            executor = PickPlaceExecutor(
                env, motion, grasp, use_attachment=True, max_plan_iters=5000
            )

            # Create a bridge.
            bridge = cast(
                _BridgeLike,
                make_lego_bridge(env, task, executor=executor, frame=frame),
            )

            # Start the timer.
            t0 = time.perf_counter()

            # Execute the plan.
            ok, actions_ok, failed = execute_tracked(
                bridge, plan, return_home=True
            )

            # Stop the timer.
            elapsed = time.perf_counter() - t0
        finally:

            # Close the environment.
            close = getattr(env, "close", None)
            if callable(close):
                _ = close()

    # Return the execution row.
    return ExecutionRow(
        tier=tier,
        task_id=task_id,
        success=ok,
        status="OK" if ok else "FAIL",
        plan_length=len(plan.actions),
        actions_ok=actions_ok,
        failed_action=failed,
        execution_time_s=round(elapsed, 4),
        path=str(yaml_path),
        plan_path=str(plan_path),
    )


# Define the function to print the summary of the execution results.
def print_summary(rows: list[ExecutionRow]) -> None:

    # Group the rows by tier.
    by_tier: dict[int, list[ExecutionRow]] = {}
    for row in rows:
        by_tier.setdefault(row.tier, []).append(row)

    # Print the summary.
    print("\n=== Execution summary ===")

    # Loop through the tiers.
    for tier in sorted(by_tier):

        # Get the rows for the tier.
        tier_rows = by_tier[tier]

        # Get the number of rows in the tier.
        n = len(tier_rows)

        # Get the number of successful rows in the tier.
        success = sum(1 for r in tier_rows if r.success)

        # Get the execution times of the successful rows.
        times = [r.execution_time_s for r in tier_rows if r.success]

        # Get the mean execution time of the successful rows.
        mean_t = sum(times) / len(times) if times else float("nan")

        # Print the summary.
        print(
            f"  tier {tier}: {success}/{n} success "
            + f"({100.0 * success / n:.1f}%), "
            + f"mean time {mean_t:.3f}s"
        )


# Define the function to parse the tiers.
def parse_tiers(raw: str) -> list[int]:

    # Parse the tiers.
    tiers = [int(part.strip()) for part in raw.split(",") if part.strip()]

    # Check if the tiers are empty.
    if not tiers:
        raise argparse.ArgumentTypeError("expected at least one tier")
    for tier in tiers:
        if tier < 1:
            raise argparse.ArgumentTypeError(f"invalid tier {tier}")
    return tiers


# Parse "3/20", "tier3/task_020", or "tier3/20" into (tier, task_id).
def parse_task_ref(raw: str) -> tuple[int, int]:

    # Strip the text and convert it to lowercase.
    text = raw.strip().lower().replace("\\", "/")

    # Check if the text is empty.
    if not text:
        raise argparse.ArgumentTypeError("empty task reference")

    # Check if the text contains a slash.
    if "/" in text:
        # Split the text into left and right parts.
        left, right = text.split("/", 1)
    elif "," in text:
        # Split the text into left and right parts.
        left, right = text.split(",", 1)
    else:
        # Raise an error if the text does not contain a slash or comma.
        raise argparse.ArgumentTypeError(
            f"expected TIER/TASK like 3/20 or tier3/task_020, got {raw!r}"
        )

    # Strip the left and right parts.
    left = left.strip()
    right = right.strip()

    # Check if the left part starts with "tier".
    if left.startswith("tier"):
        left = left[4:]

    # Check if the right part starts with "task_".
    if right.startswith("task_"):
        right = right[5:]

    # Check if the right part starts with "task".
    elif right.startswith("task"):
        right = right[4:]

    # Try to convert the left and right parts to integers.
    try:

        # Convert the left part to an integer.
        tier = int(left)

        # Convert the right part to an integer.
        task_id = int(right)

        # Raise an error if the conversion failed.
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected TIER/TASK like 3/20 or tier3/task_020, got {raw!r}"
        ) from exc

    # Check if the tier and task_id are greater than or equal to 1.
    if tier < 1 or task_id < 1:
        raise argparse.ArgumentTypeError(f"tier and task_id must be >= 1, got {raw!r}")

    # Return the tier and task_id.
    return tier, task_id


# Expand repeated/comma-separated --task values.
def parse_task_list(values: list[str] | None) -> list[tuple[int, int]] | None:

    # Check if the values are empty.
    if not values:
        # Return None.
        return None

    # Initialize the list of jobs.
    jobs: list[tuple[int, int]] = []

    # Loop through the values.
    for value in values:

        # Loop through the parts.
        for part in value.split(","):
            part = part.strip()

            # Check if the part is empty.
            if not part:
                continue

            # Add the part to the jobs list.
            jobs.append(parse_task_ref(part))

    # Check if the jobs list is empty.
    if not jobs:
        # Raise an error if the jobs list is empty.
        raise argparse.ArgumentTypeError("expected at least one task reference")

    # Return the jobs list.
    return jobs


# Define the dataclass for the command line arguments.
@dataclass(frozen=True)
class CliArgs:

    # Explicit (tier, task_id) jobs, or None to use tiers × 1..n.
    tasks: list[tuple[int, int]] | None
    # The tiers to evaluate.
    tiers: list[int]
    # The number of tasks per tier.
    n: int
    # The output / plan directory.
    out_dir: Path


# Run one task and persist artifacts; returns a row or None if YAML missing.
def run_and_write(out_dir: Path, tier: int, task_id: int) -> ExecutionRow | None:

    # Get the path to the ground truth YAML file.
    path = ground_truth_path(tier, task_id)

    # Print the task progress.
    print(f"tier{tier} task_{task_id:03d} ...", flush=True)

    # Try to evaluate the task.
    try:

        # Evaluate the task.
        row = eval_one(tier, task_id, out_dir)

    except FileNotFoundError:

        # Print the task progress.
        print("MISSING")
        return None

    except Exception as exc:

        print(f"ERROR {exc.__class__.__name__}: {exc}", flush=True)
        row = ExecutionRow(
            tier=tier,
            task_id=task_id,
            success=False,
            status=f"ERROR:{exc.__class__.__name__}",
            plan_length=0,
            actions_ok=0,
            failed_action="",
            execution_time_s=0.0,
            path=str(path),
            plan_path=str(task_out_dir(out_dir, tier, task_id) / PLAN_FILENAME),
        )

        # Write the task artifacts.
        task_dir = write_task_artifacts(out_dir, row)
        print(f"→ {task_dir}")
        return row

    # Print the task progress.
    mark = "OK" if row.success else row.status
    detail = ""

    # Check if the task succeeded.
    if row.success:
        detail = f" actions={row.actions_ok}/{row.plan_length}"

    # Check if the task failed.
    elif row.failed_action:
        detail = (
            f" actions={row.actions_ok}/{row.plan_length}"
            + f" fail={row.failed_action}"
        )

    # Write the task artifacts.
    task_dir = write_task_artifacts(out_dir, row)
    print(f"{mark}{detail} t={row.execution_time_s:.3f}s → {task_dir}")
    return row


def main(argv: list[str] | None = None) -> int:

    # Create the argument parser.
    parser = argparse.ArgumentParser(description=__doc__)

    # Add the tiers argument.
    _ = parser.add_argument(
        "--tiers",
        type=parse_tiers,
        default=[1, 2, 3],
        help="Comma-separated tiers (default: 1,2,3; ignored if --task is set)",
    )

    # Add the n argument.
    _ = parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of tasks per tier, starting at task_001 (default: 20)",
    )

    # Add the task argument.
    _ = parser.add_argument(
        "--task",
        action="append",
        default=None,
        metavar="TIER/TASK",
        help="Specific task(s), e.g. 1/1 or tier1/task_001 (repeatable)",
    )

    # Add the out-dir argument.
    _ = parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Results dir with plan.json files (default: {DEFAULT_OUT_DIR})",
    )

    # Parse the arguments.
    ns = parser.parse_args(argv)

    # Try to parse the task list.
    try:

        # Parse the task list.
        tasks = parse_task_list(cast(list[str] | None, ns.task))
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    # Create the command line arguments.
    args = CliArgs(
        tasks=tasks,
        tiers=cast(list[int], ns.tiers),
        n=cast(int, ns.n),
        out_dir=cast(Path, ns.out_dir),
    )

    # Check if the number of tasks is less than 1.
    if args.tasks is None and args.n < 1:
        parser.error("--n must be >= 1")

    # Check if the ground truth root is a directory.
    if not GROUND_TRUTH_ROOT.is_dir():
        print(
            f"ERROR: dataset ground_truth not found: {GROUND_TRUTH_ROOT}",
            file=sys.stderr,
        )
        return 1

    # Build the job list: explicit --task, or tiers × 1..n.
    if args.tasks is not None:
        jobs = args.tasks
    else:
        jobs = [
            (tier, task_id)
            for tier in args.tiers
            for task_id in range(1, args.n + 1)
        ]

    # Initialize the list of rows.
    rows: list[ExecutionRow] = []

    # Loop through the jobs.
    for tier, task_id in jobs:

        # Run and write the task.
        row = run_and_write(args.out_dir, tier, task_id)

        # Check if the row is not None.
        if row is not None:
            rows.append(row)

    # Print the number of tasks written.
    print(f"\nWrote {len(rows)} tasks under {args.out_dir}/tier*/task_*/")
    # Print the path to the summary JSONL file.
    print(f"Summary → {args.out_dir / SUMMARY_JSONL}")
    # Print the path to the summary CSV file.
    print(f"Summary → {args.out_dir / SUMMARY_CSV}")
    # Print the summary.
    print_summary(rows)

    # Return 0.
    return 0

# Run the script.
if __name__ == "__main__":
    raise SystemExit(main())
