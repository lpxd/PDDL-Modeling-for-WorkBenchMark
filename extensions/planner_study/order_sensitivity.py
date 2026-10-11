#!/usr/bin/env python
"""Phase 3: how many valid orderings of a plan are actually feasible?

Takes one plan per Tier-3 task (default: the fd-lama plan), enumerates every
ordering of its pick+place units that the PDDL model would accept, and
classifies each with the same geometric rule used elsewhere (no brick
placed under one that already covers it) -- counts are exact. A handful of
orderings other than the original are also sampled and validated with UPF's
sequential_plan_validator, for spot-checking.

Writes to results/extension/order_sensitivity/: order_counts.json (exact
counts per task), variants.json per task, and the summary CSVs, all
rebuilt from disk so the script is resumable.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from brick_geometry import overhang_map
from common import EXT_RESULTS, PLANNERS_DIR, ground_truth_path, validate_plan, write_json_atomic

OUT_DIR = EXT_RESULTS / "order_sensitivity"


def units_of(actions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for i in range(0, len(actions), 2):
        pick, put = actions[i], actions[i + 1]
        assert pick["name"] == "pick" and put["name"] in ("place", "stack")
        out[pick["params"][1].removeprefix("b_")] = [pick, put]
    return out


def named_support(unit: list[dict[str, Any]]) -> str | None:
    return unit[1]["params"][2].removeprefix("b_") if unit[1]["name"] == "stack" else None


# Every PDDL-valid ordering, with (first_blocked_position, blockers) or None if
# the ordering satisfies the geometric rule.
def enumerate_orders(units: dict[str, list[dict[str, Any]]], over: dict[str, set[str]]):
    names = list(units)
    named = {n: named_support(units[n]) for n in names}
    results: list[tuple[tuple[str, ...], tuple[int, tuple[str, ...]] | None]] = []

    def rec(order: list[str], placed: set[str], blocked: tuple[int, tuple[str, ...]] | None) -> None:
        if len(order) == len(names):
            results.append((tuple(order), blocked))
            return
        for n in names:
            if n in placed or (named[n] is not None and named[n] not in placed):
                continue
            hit = over[n] & placed
            nb = blocked if blocked is not None or not hit else (len(order), tuple(sorted(hit)))
            order.append(n)
            placed.add(n)
            rec(order, placed, nb)
            placed.discard(n)
            order.pop()

    rec([], set(), None)
    return results


# Sample up to ``cap`` orderings other than ``original``: feasible and blocked
# classes are both represented when both exist.
def sample(all_orders, original: tuple[str, ...], cap: int, rng: random.Random):
    feas = [o for o in all_orders if o[1] is None and o[0] != original]
    blk = [o for o in all_orders if o[1] is not None and o[0] != original]
    if len(feas) + len(blk) <= cap:
        return feas + blk
    nf = min(len(feas), 2)
    nb = min(len(blk), cap - nf)
    nf = min(len(feas), cap - nb)
    return rng.sample(feas, nf) + rng.sample(blk, nb)


def summarize(root: Path, tasks: list[int]) -> None:
    rows: list[dict[str, Any]] = []
    for tid in tasks:
        vpath = root / "tier3" / f"task_{tid:03d}" / "variants.json"
        if not vpath.is_file():
            continue
        v = json.loads(vpath.read_text(encoding="utf-8"))
        for var in v["variants"]:
            rows.append(
                {
                    "task": tid,
                    "variant": var["id"],
                    "order": " ".join(var["order"]),
                    "pddl_valid": var["pddl_valid"],
                    "rule_feasible": var["rule_feasible"],
                    "first_blocked_position": var["first_blocked_position"],
                    "blockers": " ".join(var["blockers"]),
                }
            )
    if not rows:
        return
    with (root / "variant_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    counts = json.loads((root / "order_counts.json").read_text(encoding="utf-8"))
    trows = []
    for tid in tasks:
        c = counts.get(str(tid))
        rs = [r for r in rows if r["task"] == tid]
        if not c or not rs:
            continue
        trows.append(
            {
                "task": tid,
                "bricks": c["bricks"],
                "pddl_valid_orders": c["pddl_valid"],
                "rule_feasible_orders": c["rule_feasible"],
                "variants_sampled": len(rs),
                "sampled_rule_feasible": sum(bool(r["rule_feasible"]) for r in rs),
            }
        )
    with (root / "task_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trows[0]))
        w.writeheader()
        w.writerows(trows)
    print("\n=== Order sensitivity ===")
    print(f"tasks summarized: {len(trows)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default=",".join(str(i) for i in range(1, 21)), help="Tier-3 task ids (default: 1..20)")
    p.add_argument("--base-planner", default="fd-lama", help="plan whose units are reordered")
    p.add_argument("--cap", type=int, default=5, help="orderings sampled per task, excluding the original (default 5)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()

    tasks = [int(t) for t in args.tasks.split(",") if t.strip()]
    root: Path = args.out_dir
    root.mkdir(parents=True, exist_ok=True)
    counts_path = root / "order_counts.json"
    counts: dict[str, Any] = json.loads(counts_path.read_text(encoding="utf-8")) if counts_path.is_file() else {}

    if not args.summarize_only:
        from workbenchmark_pddl.planner import PlanAction, PlanResult
        from workbenchmark_pddl.task import load_task

        for tid in tasks:
            task = load_task(ground_truth_path(3, tid))
            base = json.loads((PLANNERS_DIR / args.base_planner / "tier3" / f"task_{tid:03d}" / "plan.json").read_text(encoding="utf-8"))["actions"]
            units = units_of(base)
            original = tuple(units)
            vpath = root / "tier3" / f"task_{tid:03d}" / "variants.json"

            if vpath.is_file():
                spec = json.loads(vpath.read_text(encoding="utf-8"))  # keep the sample stable across restarts
            else:
                allo = enumerate_orders(units, overhang_map(task))
                counts[str(tid)] = {
                    "bricks": len(units),
                    "pddl_valid": len(allo),
                    "rule_feasible": sum(1 for o in allo if o[1] is None),
                    "original_order_rule_feasible": next(o[1] is None for o in allo if o[0] == original),
                    "base_planner": args.base_planner,
                }
                chosen = sample(allo, original, args.cap, random.Random(args.seed * 1000 + tid))
                variants = []
                for i, (order, blocked) in enumerate(chosen):
                    flat = [a for n in order for a in units[n]]
                    valid = validate_plan(task, PlanResult(True, "", tuple(PlanAction(a["name"], tuple(a["params"])) for a in flat)))
                    variants.append(
                        {
                            "id": i,
                            "order": list(order),
                            "rule_feasible": blocked is None,
                            "first_blocked_position": None if blocked is None else blocked[0],
                            "blockers": [] if blocked is None else list(blocked[1]),
                            "pddl_valid": valid,
                            "actions": flat,
                        }
                    )
                spec = {"task": tid, "base_planner": args.base_planner, "original_order": list(original), "counts": counts[str(tid)], "variants": variants}
                write_json_atomic(vpath, spec)
                write_json_atomic(counts_path, counts)
            c = spec["counts"]
            print(f"task {tid:2d}: {c['bricks']} bricks, PDDL-valid {c['pddl_valid']}, rule-feasible {c['rule_feasible']}, "
                  f"sampled {len(spec['variants'])} ({sum(v['rule_feasible'] for v in spec['variants'])} feasible), "
                  f"all PDDL-valid: {all(v['pddl_valid'] for v in spec['variants'])}", flush=True)

    summarize(root, tasks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
