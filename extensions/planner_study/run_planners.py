#!/usr/bin/env python
"""Plan WorkBenchMark tasks with several UPF engines (no simulation).

Writes to results/extension/planners/<planner>/tierN/task_XXX/: plan.json
(same format as the main project's) and planning_metrics.json, plus
planning_summary.jsonl/.csv rebuilt from disk after every task.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from patched_pddl import ENCODINGS, ENCODING_ENGINES
from common import (
    EXT_RESULTS,
    DEFAULT_TIMEOUT_S,
    ENGINES,
    GROUND_TRUTH_ROOT,
    METRICS_FILENAME,
    PLAN_FILENAME,
    PLANNERS_DIR,
    RECORD_FIELDS,
    STATUS_FAILED,
    STATUS_SOLVED,
    STATUS_TIMEOUT,
    SUMMARY_CSV,
    SUMMARY_JSONL,
    PlanningRecord,
    ground_truth_path,
    parse_task_ref,
    solve_task,
    task_dir,
    write_json_atomic,
)


# worker


# Subprocess entry point: solve one task, dump the result as JSON.
def worker_main(engine: str, yaml_path: Path, out_path: Path, domain: str = "original") -> int:
    payload = solve_task(engine, yaml_path, domain)
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    return 0


# parent


# Kill a subprocess and everything it spawned (Fast Downward starts children).
def kill_tree(proc: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        import signal

        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    proc.wait()


# Run one (planner, task) attempt in a worker; never raises for planner errors.
def run_attempt(planner: str, tier: int, task_id: int, timeout_s: float, domain: str = "original") -> tuple[PlanningRecord, dict[str, object], dict[str, object] | None]:
    engine = ENGINES[planner]
    yaml_path = ground_truth_path(tier, task_id)
    yaml_rel = f"tier{tier}/task_{task_id:03d}.yaml"

    def record(status: str, **kw: object) -> PlanningRecord:
        base: dict[str, object] = dict(
            planner=planner,
            engine=engine,
            tier=tier,
            task_id=task_id,
            status=status,
            upf_status="",
            plan_length=0,
            planning_time_s=None,
            wall_s=0.0,
            timeout_s=timeout_s,
            plan_valid=None,
            yaml=yaml_rel,
            detail="",
            domain=domain,
        )
        base.update(kw)
        return PlanningRecord(**base)  # type: ignore[arg-type]

    empty_plan: dict[str, object] = {"solved": False, "status": "", "actions": []}

    with tempfile.TemporaryDirectory(prefix="planner_study_") as tmp:
        out_path = Path(tmp) / "result.json"
        err_path = Path(tmp) / "stderr.txt"
        cmd = [
            sys.executable,
            "-W",
            "ignore",
            str(Path(__file__).resolve()),
            "--worker",
            "--engine",
            engine,
            "--yaml",
            str(yaml_path),
            "--out",
            str(out_path),
            "--domain",
            domain,
        ]
        t0 = time.perf_counter()
        with err_path.open("wb") as err:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=err,
                start_new_session=(os.name != "nt"),
            )
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                kill_tree(proc)
                wall = time.perf_counter() - t0
                return record(STATUS_TIMEOUT, wall_s=round(wall, 2)), empty_plan, None
        wall = time.perf_counter() - t0

        if proc.returncode != 0 or not out_path.is_file():
            tail = err_path.read_text(encoding="utf-8", errors="replace")[-400:].strip()
            return (
                record(
                    STATUS_FAILED,
                    wall_s=round(wall, 2),
                    detail=f"worker exit {proc.returncode}: {tail}",
                ),
                empty_plan,
                None,
            )

        data = json.loads(out_path.read_text(encoding="utf-8"))

    upf_status = str(data["upf_status"])
    plan = data["plan"]
    if data["solved"]:
        status = STATUS_SOLVED
    elif upf_status.endswith("TIMEOUT"):
        # Engine-reported timeout counts as a timeout too.
        status = STATUS_TIMEOUT
    else:
        status = STATUS_FAILED
    rec = record(
        status,
        upf_status=upf_status,
        plan_length=len(plan["actions"]),
        planning_time_s=None if status == STATUS_TIMEOUT else data["planning_time_s"],
        wall_s=round(wall, 2),
        plan_valid=data["plan_valid"],
        plan_valid_patched=data.get("plan_valid_patched"),
        rule_feasible=data.get("rule_feasible"),
    )
    return rec, plan, data.get("plan_raw")


# Skip finished tasks; re-run a timeout only if the new limit is larger.
def is_done(metrics_path: Path, timeout_s: float) -> bool:
    if not metrics_path.is_file():
        return False
    try:
        old = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if old.get("status") == STATUS_TIMEOUT and float(old.get("timeout_s", 0)) < timeout_s:
        return False
    return True


# Rebuild the rolling summaries from every metrics file on disk (abort-safe).
def rewrite_summaries(root: Path) -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob(f"*/tier*/task_*/{METRICS_FILENAME}")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({k: data.get(k) for k in RECORD_FIELDS})
    (root / SUMMARY_JSONL).write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    with (root / SUMMARY_CSV).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(records: list[PlanningRecord]) -> None:
    print("\n=== Planning summary (this invocation) ===")
    keys = sorted({(r.planner, r.tier) for r in records})
    for planner, tier in keys:
        rs = [r for r in records if r.planner == planner and r.tier == tier]
        solved = [r for r in rs if r.status == STATUS_SOLVED]
        n_to = sum(r.status == STATUS_TIMEOUT for r in rs)
        n_fail = sum(r.status == STATUS_FAILED for r in rs)
        n_invalid = sum(r.plan_valid is False for r in solved)
        times = sorted(r.planning_time_s for r in solved if r.planning_time_s is not None)
        med = times[len(times) // 2] if times else float("nan")
        print(
            f"  {planner:14s} tier {tier}: {len(solved)}/{len(rs)} solved, "
            f"{n_to} timeout, {n_fail} failed, {n_invalid} invalid, "
            f"median time {med:.2f}s"
        )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--planners", default=",".join(ENGINES), help=f"comma-separated, from: {', '.join(ENGINES)} (default: all)")
    p.add_argument("--tiers", default="1,2,3", help="comma-separated tiers (ignored with --task)")
    p.add_argument("--n", type=int, default=20, help="tasks per tier, from task_001 (ignored with --task)")
    p.add_argument("--task", action="append", metavar="TIER/TASK", help="specific task(s), e.g. 1/3 or tier3/task_020 (repeatable / comma-separated)")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help=f"seconds per task, same for all engines (default: {DEFAULT_TIMEOUT_S:g})")
    p.add_argument("--out-dir", type=Path, default=None, help=f"default: {PLANNERS_DIR} (original) or results/extension/planners_patched_<domain>")
    p.add_argument("--domain", default="original", choices=["original", *ENCODINGS],
                   help="original = the project's domain (default); quantified / closing / prereq = patched domain "
                        "(Phase 4, see patched_pddl.py). Patched results go to results/extension/planners_patched_<domain>/")
    p.add_argument("--force", action="store_true", help="re-run tasks even if results exist")
    # Internal: subprocess mode.
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--engine", help=argparse.SUPPRESS)
    p.add_argument("--yaml", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.worker:
        return worker_main(args.engine, args.yaml, args.out, args.domain)

    planners = [p.strip() for p in args.planners.split(",") if p.strip()]
    unknown = [p for p in planners if p not in ENGINES]
    if unknown:
        print(f"ERROR: unknown planner(s) {unknown}; choose from {list(ENGINES)}", file=sys.stderr)
        return 2
    if args.domain != "original":
        bad = [p for p in planners if p not in ENCODING_ENGINES[args.domain]]
        if bad:
            print(f"ERROR: {bad} cannot solve the '{args.domain}' encoding; use {list(ENCODING_ENGINES[args.domain])}", file=sys.stderr)
            return 2
    if args.out_dir is None:
        args.out_dir = PLANNERS_DIR if args.domain == "original" else EXT_RESULTS / f"planners_patched_{args.domain}"
    if not GROUND_TRUTH_ROOT.is_dir():
        print(f"ERROR: dataset ground_truth not found: {GROUND_TRUTH_ROOT}", file=sys.stderr)
        return 1

    if args.task:
        jobs = [parse_task_ref(part) for v in args.task for part in v.split(",") if part.strip()]
    else:
        tiers = [int(t) for t in args.tiers.split(",") if t.strip()]
        jobs = [(t, i) for t in tiers for i in range(1, args.n + 1)]

    out_root: Path = args.out_dir
    records: list[PlanningRecord] = []
    for planner in planners:
        for tier, task_id in jobs:
            tag = f"[{planner}] tier{tier} task_{task_id:03d}"
            if not ground_truth_path(tier, task_id).is_file():
                print(f"{tag} MISSING", flush=True)
                continue
            tdir = task_dir(out_root, planner, tier, task_id)
            if not args.force and is_done(tdir / METRICS_FILENAME, args.timeout):
                print(f"{tag} skip (done)", flush=True)
                continue

            print(f"{tag} ...", end=" ", flush=True)
            rec, plan, raw = run_attempt(planner, tier, task_id, args.timeout, args.domain)
            write_json_atomic(tdir / PLAN_FILENAME, plan)
            if raw is not None:
                write_json_atomic(tdir / "plan_raw.json", raw)
            write_json_atomic(tdir / METRICS_FILENAME, asdict(rec))
            rewrite_summaries(out_root)
            records.append(rec)

            t = "-" if rec.planning_time_s is None else f"{rec.planning_time_s:.2f}s"
            extra = "" if not rec.detail else f" | {rec.detail[:120]}"
            valid = "" if rec.plan_valid is None else f" valid={rec.plan_valid}"
            if rec.domain != "original":
                valid += f" valid_patched={rec.plan_valid_patched} rule_feasible={rec.rule_feasible}"
            print(f"{rec.status.upper()} len={rec.plan_length} t={t} wall={rec.wall_s:.1f}s{valid}{extra}", flush=True)

    rewrite_summaries(out_root)
    print(f"\nWrote {len(records)} attempt(s) under {out_root}")
    print(f"Summary -> {out_root / SUMMARY_CSV}")
    print_summary(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
