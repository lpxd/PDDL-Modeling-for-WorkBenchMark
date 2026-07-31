#!/usr/bin/env python
"""
Batch planning evaluation (no MuJoCo).

This script is used to evaluate the planning performance of the planner.

Runs ``plan_task`` on WorkBenchMark ground-truth YAMLs and writes per-task
artifacts under ``results/tierN/task_XXX/`` immediately (abort-safe), plus a
rolling summary JSONL/CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from workbenchmark_pddl.planner import (
    PlanResult,
    plan_result_to_dict,
    plan_task,
)
from workbenchmark_pddl.task import load_task

# Define the project root and dataset root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "dataset"

# Define the ground truth root and default output directory.
GROUND_TRUTH_ROOT = DATASET_ROOT / "ground_truth"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results"

# Per-task artifact filenames (simulation reads plan.json).
PLAN_FILENAME = "plan.json"
METRICS_FILENAME = "planning_metrics.json"
SUMMARY_JSONL = "planning_summary.jsonl"
SUMMARY_CSV = "planning_summary.csv"

# Define the dataclass for the planning row.
@dataclass(frozen=True)
class PlanningRow:
    # The tier of the task.
    tier: int
    # The id of the task.
    task_id: int
    # Whether the task was solved.
    solved: bool
    # The status of the task.
    status: str
    # The length of the plan.
    plan_length: int
    # The time taken to plan the task.
    planning_time_s: float
    # The path to the task.
    path: str


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
    # Create a temporary file.
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Write the payload to the temporary file.
    _ = tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    # Replace the temporary file with the original file.
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


# Rewrite JSONL from all on-disk planning_metrics.json (abort-safe).
def rewrite_summary_jsonl(out_dir: Path) -> Path:

    # Define the fieldnames for the JSONL file.
    fieldnames = [
        "tier",
        "task_id",
        "solved",
        "status",
        "plan_length",
        "planning_time_s",
        "path",
    ]

    # Get the path to the JSONL file.
    jsonl_path = out_dir / SUMMARY_JSONL

    # Create the parent directory if it doesn't exist.
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize the list of lines.
    lines: list[str] = []
    
    # Loop through the metrics files.
    for metrics_path in sorted(out_dir.glob("tier*/task_*/" + METRICS_FILENAME)):

        # Load the metrics from the file.
        data = _load_metrics(metrics_path)

        # Add the data to the list of lines.
        lines.append(json.dumps({key: data[key] for key in fieldnames}))

    # Write the lines to the JSONL file.
    _ = jsonl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    # Return the path to the JSONL file.
    return jsonl_path


# Rewrite CSV from all on-disk planning_metrics.json (abort-safe).
def rewrite_summary_csv(out_dir: Path) -> Path:

    # Define the fieldnames for the CSV file.
    fieldnames = [
        "tier",
        "task_id",
        "solved",
        "status",
        "plan_length",
        "planning_time_s",
        "path",
    ]

    # Initialize the list of rows.
    rows: list[dict[str, object]] = []

    # Loop through the metrics files.
    for metrics_path in sorted(out_dir.glob("tier*/task_*/" + METRICS_FILENAME)):

        # Load the metrics from the file.
        data = _load_metrics(metrics_path)

        # Add the data to the list of rows.
        rows.append({key: data[key] for key in fieldnames})

    # Get the path to the CSV file.
    csv_path = out_dir / SUMMARY_CSV

    # Create the parent directory if it doesn't exist.
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Open the CSV file.
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        # Create a writer for the CSV file.
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # Write the header to the CSV file.
        writer.writeheader()
        # Loop through the rows.
        for row in rows:
            writer.writerow(row)

    # Return the path to the CSV file.
    return csv_path


# Persist plan + metrics for one task, then refresh summaries from disk.
def write_task_artifacts(
    out_dir: Path,
    row: PlanningRow,
    plan: PlanResult,
) -> Path:
    task_dir = task_out_dir(out_dir, row.tier, row.task_id)
    write_json_atomic(task_dir / PLAN_FILENAME, plan_result_to_dict(plan))
    write_json_atomic(task_dir / METRICS_FILENAME, asdict(row))
    _ = rewrite_summary_jsonl(out_dir)
    _ = rewrite_summary_csv(out_dir)
    return task_dir


# Define the function to evaluate one task.
def eval_one(tier: int, task_id: int) -> tuple[PlanningRow, PlanResult]:

    # Get the ground truth path for the task.
    path = ground_truth_path(tier, task_id)

    # Check if the path is a file.
    if not path.is_file():
        raise FileNotFoundError(path)

    # Load the task from the path.
    task = load_task(path)

    # Start the timer.
    t0 = time.perf_counter()

    # Plan the task.
    result = plan_task(task)

    # Calculate the elapsed time.
    elapsed = time.perf_counter() - t0

    # Return the planning row and plan (for serialization).
    row = PlanningRow(
        tier=tier,
        task_id=task_id,
        solved=result.solved,
        status=result.status,
        plan_length=len(result.actions),
        planning_time_s=round(elapsed, 4),
        path=str(path),
    )
    return row, result


# Define the function to print the summary of the planning results.
def print_summary(rows: list[PlanningRow]) -> None:

    # Group the rows by tier.
    by_tier: dict[int, list[PlanningRow]] = {}

    # Group the rows by tier.
    for row in rows:
        by_tier.setdefault(row.tier, []).append(row)

    # Print the summary.
    print("\n=== Planning summary ===")

    # Sort the tiers.
    for tier in sorted(by_tier):

        # Get the rows for the tier.
        tier_rows = by_tier[tier]

        # Get the number of rows for the tier.
        n = len(tier_rows)

        # Get the number of solved rows for the tier.
        solved = sum(1 for r in tier_rows if r.solved)

        # Get the times for the solved rows for the tier.
        times = [r.planning_time_s for r in tier_rows if r.solved]

        # Get the lengths for the solved rows for the tier.
        lengths = [r.plan_length for r in tier_rows if r.solved]

        # Get the mean time and length for the solved rows for the tier.
        mean_t = sum(times) / len(times) if times else float("nan")
        mean_l = sum(lengths) / len(lengths) if lengths else float("nan")

        # Print the summary for the tier.
        print(
            f"  tier {tier}: {solved}/{n} solved "
            + f"({100.0 * solved / n:.1f}%), "
            + f"mean time {mean_t:.3f}s, mean length {mean_l:.1f}"
        )


# Define the function to parse the tiers.
def parse_tiers(raw: str) -> list[int]:

    # Parse the tiers.
    tiers = [int(part.strip()) for part in raw.split(",") if part.strip()]

    # Check if the tiers are empty.
    if not tiers:
        # Raise an error if the tiers are empty.
        raise argparse.ArgumentTypeError("expected at least one tier")
    for tier in tiers:
        # Raise an error if the tier is less than 1.
        if tier < 1:
            raise argparse.ArgumentTypeError(f"invalid tier {tier}")
    # Return the tiers.
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
    # Check if the text contains a comma.
    elif "," in text:
        # Split the text into left and right parts.
        left, right = text.split(",", 1)
    # Raise an error if the text does not contain a slash or comma.
    else:
        raise argparse.ArgumentTypeError(
            f"expected TIER/TASK like 3/20 or tier3/task_020, got {raw!r}"
        )

    # Strip the left and right parts.
    left = left.strip()
    right = right.strip()

    # Check if the left part starts with "tier".
    if left.startswith("tier"):
        left = left[4:]
    if right.startswith("task_"):
        right = right[5:]
    elif right.startswith("task"):
        right = right[4:]

    try:
        # Convert the left part to an integer.
        tier = int(left)
        task_id = int(right)
    except ValueError as exc:
        # Raise an error if the left part is not an integer.
        raise argparse.ArgumentTypeError(
            f"expected TIER/TASK like 3/20 or tier3/task_020, got {raw!r}"
        ) from exc

    # Check if the tier and task_id are less than 1.
    if tier < 1 or task_id < 1:
        # Raise an error if the tier and task_id are less than 1.
        raise argparse.ArgumentTypeError(f"tier and task_id must be >= 1, got {raw!r}")

    # Return the tier and task_id.
    return tier, task_id


# Expand repeated/comma-separated --task values.
def parse_task_list(values: list[str] | None) -> list[tuple[int, int]] | None:

    # Check if the values are empty.
    if not values:
        return None

    # Initialize the list of jobs.
    jobs: list[tuple[int, int]] = []

    # Loop through the values.
    for value in values:

        # Loop through the parts of the value.
        for part in value.split(","):

            # Strip the part and check if it is empty.
            part = part.strip()

            # Check if the part is empty.
            if not part:
                continue

            # Add the job to the list of jobs.
            jobs.append(parse_task_ref(part))

    # Check if the jobs are empty.
    if not jobs:
        # Raise an error if the jobs are empty.
        raise argparse.ArgumentTypeError("expected at least one task reference")
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
    # The output directory.
    out_dir: Path


# Run one task and persist artifacts; returns a row or None if missing.
def run_and_write(
    out_dir: Path,
    tier: int,
    task_id: int,
) -> PlanningRow | None:

    # Get the ground truth path for the task.
    path = ground_truth_path(tier, task_id)
    print(f"tier{tier} task_{task_id:03d} ...", end=" ", flush=True)

    # Try to evaluate the task.
    try:
        # Evaluate the task.
        row, plan = eval_one(tier, task_id)
    # Raise an error if the task is not found.
    except FileNotFoundError:
        print("MISSING")
        return None
    # Raise an error if an exception occurs.
    except Exception as exc:
        # Print the error.
        print(f"ERROR {exc.__class__.__name__}: {exc}", end=" ", flush=True)

        # Create a new planning row.
        row = PlanningRow(
            tier=tier,
            task_id=task_id,
            solved=False,
            status=f"ERROR:{exc.__class__.__name__}",
            plan_length=0,
            planning_time_s=0.0,
            path=str(path),
        )

        # Create a new plan result.
        plan = PlanResult(solved=False, status=row.status, actions=())

        # Write the task artifacts.
        task_dir = write_task_artifacts(out_dir, row, plan)

        # Print the task directory.
        print(f"→ {task_dir}")

        # Return the planning row.
        return row

    # Print the mark.
    mark = "OK" if row.solved else "FAIL"

    # Print the mark.
    print(
        f"{mark} len={row.plan_length} t={row.planning_time_s:.3f}s "
        + f"({row.status})",
        end=" ",
        flush=True,
    )

    # Write the task artifacts.
    task_dir = write_task_artifacts(out_dir, row, plan)

    # Print the task directory.
    print(f"→ {task_dir}")

    # Return the planning row.
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

    # Add the number of tasks argument.
    _ = parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of tasks per tier, starting at task_001 (default: 20; ignored if --task is set)",
    )

    # Run one or more specific tasks instead of the tier/n sweep.
    _ = parser.add_argument(
        "--task",
        action="append",
        default=None,
        metavar="TIER/TASK",
        help="Specific task(s), e.g. 3/20 or tier3/task_020 (repeatable / comma-separated)",
    )

    # Add the output directory argument.
    _ = parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )

    # Parse the arguments and cast them to the CliArgs dataclass.
    ns = parser.parse_args(argv)
    try:
        tasks = parse_task_list(cast(list[str] | None, ns.task))
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

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
        print(f"ERROR: dataset ground_truth not found: {GROUND_TRUTH_ROOT}", file=sys.stderr)
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
    rows: list[PlanningRow] = []

    # Loop through the selected jobs.
    for tier, task_id in jobs:
        row = run_and_write(args.out_dir, tier, task_id)
        if row is not None:
            rows.append(row)

    print(f"\nWrote {len(rows)} tasks under {args.out_dir}/tier*/task_*/")
    print(f"Summary → {args.out_dir / SUMMARY_JSONL}")
    print(f"Summary → {args.out_dir / SUMMARY_CSV}")
    print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
