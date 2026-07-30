#!/usr/bin/env python
"""
Batch planning evaluation (no MuJoCo).  

This script is used to evaluate the planning performance of the planner.

Runs ``plan_task`` on WorkBenchMark ground-truth YAMLs and writes JSONL + CSV.
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

from workbenchmark_pddl.planner import plan_task
from workbenchmark_pddl.task import load_task

# Define the project root and dataset root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "dataset"

# Define the ground truth root and default output directory.
GROUND_TRUTH_ROOT = DATASET_ROOT / "ground_truth"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results"

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

# Define the function to evaluate one task.
def eval_one(tier: int, task_id: int) -> PlanningRow:

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

    # Return the planning row.
    return PlanningRow(
        tier=tier,
        task_id=task_id,
        solved=result.solved,
        status=result.status,
        plan_length=len(result.actions),
        planning_time_s=round(elapsed, 4),
        path=str(path),
    )


# Define the function to write the planning results to a JSONL file.
def write_jsonl(path: Path, rows: list[PlanningRow]) -> None:

    # Create the parent directory if it doesn't exist.
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write the rows to the JSONL file.
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            _ = f.write(json.dumps(asdict(row)) + "\n")

# Define the function to write the planning results to a CSV file.
def write_csv(path: Path, rows: list[PlanningRow]) -> None:

    # Create the parent directory if it doesn't exist.
    path.parent.mkdir(parents=True, exist_ok=True)

    # Get the fieldnames for the CSV file.
    fieldnames = list(asdict(rows[0]).keys()) if rows else [
        "tier",
        "task_id",
        "solved",
        "status",
        "plan_length",
        "planning_time_s",
        "path",
    ]

    # Write the rows to the CSV file.
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

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

# Define the dataclass for the command line arguments.
@dataclass(frozen=True)
class CliArgs:
    # The tiers to evaluate.
    tiers: list[int]
    # The number of tasks per tier.
    n: int
    # The output directory.
    out_dir: Path
    # The prefix for the output files.
    prefix: str


def main(argv: list[str] | None = None) -> int:

    # Create the argument parser.
    parser = argparse.ArgumentParser(description=__doc__)

    # Add the tiers argument.
    _ = parser.add_argument(
        "--tiers",
        type=parse_tiers,
        default=[1, 2, 3],
        help="Comma-separated tiers (default: 1,2,3)",
    )

    # Add the number of tasks argument.
    _ = parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of tasks per tier, starting at task_001 (default: 20)",
    )

    # Add the output directory argument.
    _ = parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )

    # Add the prefix argument.
    _ = parser.add_argument(
        "--prefix",
        type=str,
        default="planning",
        help="Filename prefix for JSONL/CSV (default: planning)",
    )

    # Parse the arguments and cast them to the CliArgs dataclass.
    ns = parser.parse_args(argv)
    args = CliArgs(
        tiers=cast(list[int], ns.tiers),
        n=cast(int, ns.n),
        out_dir=cast(Path, ns.out_dir),
        prefix=cast(str, ns.prefix),
    )

    # Check if the number of tasks is less than 1.
    if args.n < 1:
        parser.error("--n must be >= 1")

    # Check if the ground truth root is a directory.
    if not GROUND_TRUTH_ROOT.is_dir():
        print(f"ERROR: dataset ground_truth not found: {GROUND_TRUTH_ROOT}", file=sys.stderr)
        return 1

    # Initialize the list of rows.
    rows: list[PlanningRow] = []

    # Loop through the tiers.
    for tier in args.tiers:

        # Loop through the tasks.
        for task_id in range(1, args.n + 1):

            # Get the ground truth path for the task.
            path = ground_truth_path(tier, task_id)
            print(f"tier{tier} task_{task_id:03d} ...", end=" ", flush=True)

            # Try to evaluate the task.
            try:
                # Evaluate the task.
                row = eval_one(tier, task_id)
            except FileNotFoundError:
                # Print a message if the task is missing.
                print("MISSING")
                continue
            except Exception as exc:
                # Print an error message if the task evaluation failed.
                print(f"ERROR {exc.__class__.__name__}: {exc}")
                rows.append(
                    PlanningRow(
                        tier=tier,
                        task_id=task_id,
                        solved=False,
                        status=f"ERROR:{exc.__class__.__name__}",
                        plan_length=0,
                        planning_time_s=0.0,
                        path=str(path),
                    )
                )
                continue

            # Print the result of the task.
            mark = "OK" if row.solved else "FAIL"
            print(
                f"{mark} len={row.plan_length} t={row.planning_time_s:.3f}s "
                + f"({row.status})"
            )

            # Add the row to the list of rows.
            rows.append(row)

    # Get the output directory.
    out_dir = args.out_dir

    # Get the JSONL path.
    jsonl_path = out_dir / f"{args.prefix}.jsonl"

    # Get the CSV path.
    csv_path = out_dir / f"{args.prefix}.csv"

    # Write the rows to the JSONL file.
    write_jsonl(jsonl_path, rows)

    # Write the rows to the CSV file.
    write_csv(csv_path, rows)
    print(f"\nWrote {len(rows)} rows → {jsonl_path}")
    print(f"Wrote {len(rows)} rows → {csv_path}")
    print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
