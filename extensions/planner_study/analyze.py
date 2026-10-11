#!/usr/bin/env python
"""Phase 5: tables for the presentation.

Reads only files already written by run_planners.py / order_sensitivity.py /
phase4_report.py / patched_orderings.py. Computes nothing new; it aggregates.

Writes planning_tier3.csv, planning_tiers1_2_compact.csv, and
domain_patch_before_after.csv to results/extension/comparison/ -- the three
tables actually used in the presentation (plan length never differs between
engines here, see SUMMARY.md). The order-sensitivity and
pyperplan-on-patched-domain findings are still computed and printed to the
console, but no longer written as CSVs, since neither is shown as a table;
their raw data still lives in results/extension/order_sensitivity/ and
results/extension/patched/pyperplan_on_patched.json.
"""

from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path
from typing import Any

from common import EXT_RESULTS, PLANNERS_DIR

COMPARISON = EXT_RESULTS / "comparison"
ORDER_DIR = EXT_RESULTS / "order_sensitivity"
PATCHED = EXT_RESULTS / "patched"

TIER3_ENGINES = ("pyperplan", "pyperplan-opt", "fd-lama", "fd-lmcut")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(",".join(fieldnames or []) + "\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# Table 1


def planning_tables() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = list(csv.DictReader(open(PLANNERS_DIR / "planning_summary.csv", encoding="utf-8")))
    by_task: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for r in summary:
        by_task.setdefault((int(r["tier"]), int(r["task_id"])), []).append(r)

    def best_len(tier: int, task_id: int) -> int | None:
        lens = [int(r["plan_length"]) for r in by_task[(tier, task_id)] if r["status"] == "solved"]
        return min(lens) if lens else None

    tier3_rows = []
    for engine in TIER3_ENGINES:
        rs = [r for r in summary if r["planner"] == engine and int(r["tier"]) == 3]
        solved = [r for r in rs if r["status"] == "solved"]
        times = [float(r["planning_time_s"]) for r in solved]
        rel = [int(r["plan_length"]) / best_len(3, int(r["task_id"])) for r in solved if best_len(3, int(r["task_id"]))]
        tier3_rows.append(
            {
                "engine": engine,
                "tasks": len(rs),
                "solved": len(solved),
                "success_rate": round(len(solved) / len(rs), 3) if rs else None,
                "timeout_rate": round(sum(r["status"] == "timeout" for r in rs) / len(rs), 3) if rs else None,
                "median_planning_time_s": round(st.median(times), 2) if times else None,
                "mean_plan_length_vs_best": round(st.mean(rel), 3) if rel else None,
            }
        )

    compact_rows = []
    for tier in (1, 2):
        for engine in TIER3_ENGINES:
            rs = [r for r in summary if r["planner"] == engine and int(r["tier"]) == tier]
            if not rs:
                continue
            solved = [r for r in rs if r["status"] == "solved"]
            times = [float(r["planning_time_s"]) for r in solved]
            compact_rows.append(
                {
                    "tier": tier,
                    "engine": engine,
                    "solved": f"{len(solved)}/{len(rs)}",
                    "median_planning_time_s": round(st.median(times), 2) if times else None,
                    "identical_plan_across_engines": None,  # filled below
                }
            )
    # tiers 1-2: is every engine's plan the same (already established in SUMMARY.md)?
    from common import plan_id

    for row in compact_rows:
        tier, engine = row["tier"], row["engine"]
        ids = set()
        for e in TIER3_ENGINES:
            for tid in range(1, 21):
                d = PLANNERS_DIR / e / f"tier{tier}" / f"task_{tid:03d}"
                m = d / "planning_metrics.json"
                if m.is_file() and json.loads(m.read_text(encoding="utf-8"))["status"] == "solved":
                    ids.add(plan_id(json.loads((d / "plan.json").read_text(encoding="utf-8"))["actions"]))
        row["identical_plan_across_engines"] = len(ids) <= 20  # one id per task at most -> forced order

    return tier3_rows, compact_rows


# Order sensitivity


def order_sensitivity_table() -> list[dict[str, Any]]:
    counts = json.loads((ORDER_DIR / "order_counts.json").read_text(encoding="utf-8"))
    variants = list(csv.DictReader(open(ORDER_DIR / "variant_summary.csv", encoding="utf-8")))
    rows = []
    for tid in range(1, 21):
        c = counts.get(str(tid))
        if not c:
            continue
        vs = [v for v in variants if int(v["task"]) == tid]
        rows.append(
            {
                "task": tid,
                "bricks": c["bricks"],
                "pddl_valid_orderings": c["pddl_valid"],
                "geometrically_feasible_orderings": c["rule_feasible"],
                "feasible_share_pct": round(100 * c["rule_feasible"] / c["pddl_valid"], 1),
                "base_plan_order_feasible": c["original_order_rule_feasible"],
                "variants_sampled": len(vs),
            }
        )
    return rows


# Table 2


def patch_before_after_table() -> list[dict[str, Any]]:
    p4 = json.loads((PATCHED / "phase4_summary.json").read_text(encoding="utf-8"))
    return p4["b_planning"]


def patch_pyperplan_table() -> list[dict[str, Any]]:
    pp = json.loads((PATCHED / "pyperplan_on_patched.json").read_text(encoding="utf-8"))
    rows = []
    for key, v in pp.items():
        encoding, engine, tier, task = key.split("/")
        rows.append(
            {
                "encoding": encoding,
                "engine": engine,
                "task": f"{tier}/{task}",
                "outcome": v["outcome"],
                "detail": v.get("exception", "") or (f"len={v.get('plan_length')} rule_feasible={v.get('rule_feasible')}"),
            }
        )
    return rows


# main


def main() -> int:
    COMPARISON.mkdir(parents=True, exist_ok=True)

    tier3_rows, compact_rows = planning_tables()
    write_csv(COMPARISON / "planning_tier3.csv", tier3_rows)
    write_csv(COMPARISON / "planning_tiers1_2_compact.csv", compact_rows)
    print("=== Planning, tier 3 (per engine) ===")
    for r in tier3_rows:
        print(f"  {r['engine']:12s} solved {r['solved']}/{r['tasks']}  timeout_rate {r['timeout_rate']}  "
              f"median_time {r['median_planning_time_s']}s  mean_len_vs_best {r['mean_plan_length_vs_best']}")

    order_rows = order_sensitivity_table()
    print(f"\n=== Order sensitivity: {sum(r['variants_sampled'] for r in order_rows)} sampled orderings across {len(order_rows)} tier-3 tasks ===")

    patch_rows = patch_before_after_table()
    write_csv(COMPARISON / "domain_patch_before_after.csv", patch_rows)
    pp_rows = patch_pyperplan_table()
    print("\n=== Domain patch: pyperplan on the patched domain(s) ===")
    for r in pp_rows:
        print(f"  {r['encoding']:11s} {r['engine']:14s} {r['task']:8s} {r['outcome']:8s} {r['detail']}")

    print(f"\nTables -> {COMPARISON}/*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
