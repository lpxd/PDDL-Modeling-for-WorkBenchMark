#!/usr/bin/env python
"""Record what the pyperplan engines do with each patched encoding.

run_planners.py refuses pyperplan for the ``quantified`` encoding, so that outcome
was never written to a file. This script attempts every (encoding, pyperplan
engine) pair on a small set of tasks and records the outcome, including the exact
UPF error when the domain is not supported. Timings are single runs and are only
indicative (they were taken while other jobs may have been running).

Output: results/extension/patched/pyperplan_on_patched.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import EXT_RESULTS, ground_truth_path, parse_task_ref, solve_task, write_json_atomic

OUT = EXT_RESULTS / "patched" / "pyperplan_on_patched.json"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default="1/1,3/1", help="TIER/TASK list")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()

    results: dict[str, object] = json.loads(args.out.read_text(encoding="utf-8")) if args.out.is_file() else {}
    for ref in args.tasks.split(","):
        tier, tid = parse_task_ref(ref)
        for enc in ("quantified", "closing", "prereq"):
            for engine in ("pyperplan", "pyperplan-opt"):
                key = f"{enc}/{engine}/tier{tier}/task_{tid:03d}"
                if key in results:
                    continue
                t0 = time.time()
                try:
                    r = solve_task(engine, ground_truth_path(tier, tid), enc)
                    entry = {"outcome": "solved" if r["solved"] else "not solved", "upf_status": r["upf_status"],
                             "plan_length": len(r["plan"]["actions"]), "plan_valid_original_domain": r["plan_valid"],
                             "plan_valid_patched_domain": r["plan_valid_patched"], "rule_feasible": r["rule_feasible"],
                             "seconds": round(time.time() - t0, 1)}
                except Exception as exc:  # noqa: BLE001 - the error IS the result
                    entry = {"outcome": "error", "exception": type(exc).__name__,
                             "message": " ".join(str(exc).split())[:400], "seconds": round(time.time() - t0, 1)}
                results[key] = entry
                print(f"{key:48s} {entry['outcome']:10s} " + (entry.get("exception", "") or f"len={entry['plan_length']} feasible={entry['rule_feasible']} {entry['seconds']}s"), flush=True)
                write_json_atomic(args.out, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
