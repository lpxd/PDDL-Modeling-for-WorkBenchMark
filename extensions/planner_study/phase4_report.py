#!/usr/bin/env python
"""Phase 4 report: one place for every number the summary uses (no simulation).

Reads results/extension/{planners, planners_patched_<enc>, patched/...} and prints
and saves results/extension/patched/phase4_summary.json:

 (a) ordering-set comparison per domain (from order_set_comparison.json)
 (b) planning before/after: solved, time, plan length, validity, rule feasibility
 (c) Tiers 1-2: patched plans identical to the original plans?
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path
from typing import Any

from common import EXT_RESULTS, ground_truth_path, plan_id, plan_rule_feasible, write_json_atomic

PATCHED = EXT_RESULTS / "patched"
DIRS = {
    "original": EXT_RESULTS / "planners",
    "quantified": EXT_RESULTS / "planners_patched_quantified",
    "closing": EXT_RESULTS / "planners_patched_closing",
    "prereq": EXT_RESULTS / "planners_patched_prereq",
}
CONFIGS = [("original", "fd-lama"), ("quantified", "fd-lama"), ("closing", "fd-lama"),
           ("original", "fd-lmcut"), ("closing", "fd-lmcut")]


def metrics(domain: str, planner: str, tier: int, tid: int) -> dict[str, Any] | None:
    p = DIRS[domain] / planner / f"tier{tier}" / f"task_{tid:03d}" / "planning_metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def plan_actions(domain: str, planner: str, tier: int, tid: int):
    p = DIRS[domain] / planner / f"tier{tier}" / f"task_{tid:03d}" / "plan.json"
    return json.loads(p.read_text(encoding="utf-8"))["actions"]


def main() -> int:
    from workbenchmark_pddl.task import load_task

    out: dict[str, Any] = {}

    # (a)
    oc = json.loads((PATCHED / "order_set_comparison.json").read_text(encoding="utf-8"))
    out["a_ordering_sets"] = {
        dom: {"tasks_exact_match": sum(r[dom]["exact_match"] for r in oc.values() if dom in r),
              "tasks": sum(1 for r in oc.values() if dom in r)}
        for dom in ("original", "quantified", "closing", "prereq")
    }
    out["a_original_accepts_infeasible_total"] = sum(r["original"]["accepted_but_not_feasible"] for r in oc.values())
    out["a_pddl_valid_total"] = sum(r["pddl_valid"] for r in oc.values())
    out["a_feasible_total"] = sum(r["feasible"] for r in oc.values())
    print("(a) exact ordering-set match:", out["a_ordering_sets"])

    # (b) + rule feasibility of the ORIGINAL plans (not stored in the old metrics)
    print("\n(b) planning, per configuration")
    print(f"{'domain':11s} {'engine':9s} tier  solved  timeouts  time median/mean/max (s)   mean len  valid(orig)  rule-feasible")
    rows = []
    for tier in (3, 2, 1):
        for domain, planner in CONFIGS:
            ms = [(tid, metrics(domain, planner, tier, tid)) for tid in range(1, 21)]
            ms = [(t, m) for t, m in ms if m]
            if not ms:
                continue
            solved = [(t, m) for t, m in ms if m["status"] == "solved"]
            times = [m["planning_time_s"] for _, m in solved]
            feas = 0
            for t, m in solved:
                task = load_task(ground_truth_path(tier, t))
                feas += plan_rule_feasible(task, plan_actions(domain, planner, tier, t))
            row = {"tier": tier, "domain": domain, "engine": planner, "attempted": len(ms), "solved": len(solved),
                   "timeouts": sum(m["status"] == "timeout" for _, m in ms),
                   "time_median": round(st.median(times), 1), "time_mean": round(st.mean(times), 1), "time_max": round(max(times), 1),
                   "mean_plan_length": round(st.mean(m["plan_length"] for _, m in solved), 2),
                   "valid_original_domain": sum(bool(m["plan_valid"]) for _, m in solved),
                   "rule_feasible": feas}
            rows.append(row)
            print(f"{domain:11s} {planner:9s} {tier}     {len(solved):2d}/{len(ms):2d}   {row['timeouts']}        "
                  f"{row['time_median']:6.1f} / {row['time_mean']:6.1f} / {row['time_max']:6.1f}     {row['mean_plan_length']:5.1f}     "
                  f"{row['valid_original_domain']:2d}          {feas:2d}/{len(solved)}")
    out["b_planning"] = rows

    # per-task plan length before/after on tasks solved in both
    lens = {}
    for domain, planner in CONFIGS:
        if domain == "original":
            continue
        n = same = 0
        for tid in range(1, 21):
            a, b = metrics("original", planner, 3, tid), metrics(domain, planner, 3, tid)
            if a and b and a["status"] == "solved" and b["status"] == "solved":
                n += 1
                same += a["plan_length"] == b["plan_length"]
        lens[f"{domain}/{planner}"] = f"{same}/{n} tasks with equal plan length (tier 3)"
    out["b_plan_length_equal"] = lens
    print("\nplan length before vs after:", lens)

    # (c)
    print("\n(c) Tiers 1-2: patched plan identical to the original plan")
    reg = {}
    for tier in (1, 2):
        for domain, planner in CONFIGS:
            if domain == "original":
                continue
            same = sum(1 for t in range(1, 21)
                       if metrics(domain, planner, tier, t) and metrics(domain, planner, tier, t)["status"] == "solved"
                       and plan_id(plan_actions("original", planner, tier, t)) == plan_id(plan_actions(domain, planner, tier, t)))
            att = sum(1 for t in range(1, 21) if metrics(domain, planner, tier, t))
            if att:
                reg[f"tier{tier}/{domain}/{planner}"] = f"{same}/{att}"
                print(f"  tier {tier} {domain:10s} {planner:9s}: identical {same}/{att}")
    out["c_regression_identical_plans"] = reg

    write_json_atomic(PATCHED / "phase4_summary.json", out)
    print("\n->", PATCHED / "phase4_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
