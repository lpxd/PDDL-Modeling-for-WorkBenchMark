#!/usr/bin/env python
"""Phase 4 check (a): does the patched domain accept exactly the feasible orderings?

For each Tier-3 task, steps through one plan's units in UPF's sequential
simulator and finds every ordering a given domain accepts. The original
domain's accepted set should equal the PDDL-valid set from
order_sensitivity.py; each patched encoding's accepted set should equal the
geometrically feasible set. The simulator is the oracle for what the PDDL
text actually says -- it tests that the encodings implement the rule, not
that the rule itself is physically right.

Successors are memoised per set of placed bricks (state doesn't depend on
placement order); that assumption is spot-checked without memoisation
wherever the valid-ordering count is small enough (--check-limit).

Output: results/extension/patched/order_set_comparison.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from common import EXT_RESULTS, PLANNERS_DIR, ground_truth_path, write_json_atomic

OUT = EXT_RESULTS / "patched" / "order_set_comparison.json"


def build_problem(task: Any, domain: str):
    import patched_pddl as pp
    from unified_planning.io import PDDLReader
    from workbenchmark_pddl.problem import domain_path, render_problem

    if domain == "original":
        return PDDLReader().parse_problem_string(domain_path().read_text(encoding="utf-8"), render_problem(task, problem_name="enum"))
    return pp.parse_patched(task, domain, "enum")


# {brick: [action instances]} for the units of the plan, in the given domain's arity.
def unit_instances(task: Any, problem: Any, units: dict[str, list[dict[str, Any]]], domain: str):
    import patched_pddl as pp
    from unified_planning.plans import ActionInstance
    from unified_planning.shortcuts import ObjectExp

    out = {}
    for brick, acts in units.items():
        acts = pp.add_slots(task, acts) if domain in pp.SLOT_ENCODINGS else acts
        out[brick] = [
            ActionInstance(problem.action(a["name"]), tuple(ObjectExp(problem.object(p)) for p in a["params"]))
            for a in acts
        ]
    return out


# Every ordering of the units that the domain accepts. ``memo``: successors are
# cached per set of placed bricks (valid because the state depends only on that set).
def accepted_orders(problem: Any, inst: dict[str, list[Any]], memo: bool) -> set[tuple[str, ...]]:
    from unified_planning.engines.sequential_simulator import UPSequentialSimulator

    names = list(inst)
    result: set[tuple[str, ...]] = set()
    with UPSequentialSimulator(problem) as sim:
        cache: dict[frozenset[str], dict[str, Any]] = {}

        def successors(placed: frozenset[str], state: Any) -> dict[str, Any]:
            if memo and placed in cache:
                return cache[placed]
            out: dict[str, Any] = {}
            for n in names:
                if n in placed:
                    continue
                s = state
                for act in inst[n]:
                    s = sim.apply(s, act)
                    if s is None:
                        break
                if s is not None:
                    out[n] = s
            if memo:
                cache[placed] = out
            return out

        def rec(order: list[str], placed: frozenset[str], state: Any) -> None:
            if len(order) == len(names):
                if sim.is_goal(state):
                    result.add(tuple(order))
                return
            for n, s in successors(placed, state).items():
                order.append(n)
                rec(order, placed | {n}, s)
                order.pop()

        rec([], frozenset(), sim.get_initial_state())
    return result


def main() -> int:
    from brick_geometry import overhang_map
    from order_sensitivity import enumerate_orders, units_of
    from workbenchmark_pddl.task import load_task

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default=",".join(str(i) for i in range(1, 21)))
    p.add_argument("--base-planner", default="fd-lama")
    p.add_argument("--domains", default="original,quantified,closing,prereq")
    p.add_argument("--check-limit", type=int, default=400, help="also run WITHOUT memoisation when the task has at most this many valid orderings")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()

    results: dict[str, Any] = json.loads(args.out.read_text(encoding="utf-8")) if args.out.is_file() else {}
    for tid in [int(t) for t in args.tasks.split(",")]:
        if str(tid) in results and all(d in results[str(tid)] for d in args.domains.split(",")):
            continue
        task = load_task(ground_truth_path(3, tid))
        base = json.loads((PLANNERS_DIR / args.base_planner / "tier3" / f"task_{tid:03d}" / "plan.json").read_text(encoding="utf-8"))["actions"]
        units = units_of(base)
        allo = enumerate_orders(units, overhang_map(task))
        pddl_valid = {o[0] for o in allo}
        feasible = {o[0] for o in allo if o[1] is None}
        row: dict[str, Any] = results.get(str(tid), {})
        row.update(bricks=len(units), pddl_valid=len(pddl_valid), feasible=len(feasible))

        for domain in args.domains.split(","):
            t0 = time.time()
            problem = build_problem(task, domain)
            inst = unit_instances(task, problem, units, domain)
            acc = accepted_orders(problem, inst, memo=True)
            target = pddl_valid if domain == "original" else feasible
            entry: dict[str, Any] = {
                "accepted": len(acc),
                "compared_with": "pddl_valid" if domain == "original" else "feasible",
                "target": len(target),
                "exact_match": acc == target,
                "accepted_not_in_target": len(acc - target),
                "target_not_accepted": len(target - acc),
                "examples_accepted_not_in_target": [list(o) for o in sorted(acc - target)[:3]],
                "examples_target_not_accepted": [list(o) for o in sorted(target - acc)[:3]],
                "seconds": round(time.time() - t0, 1),
            }
            if domain == "original":
                # negative control: the unpatched domain must accept infeasible orderings
                entry["accepted_but_not_feasible"] = len(acc - feasible)
            if len(pddl_valid) <= args.check_limit:
                entry["memo_check_without_memo_equal"] = accepted_orders(problem, inst, memo=False) == acc
            row[domain] = entry
            print(f"task {tid:2d} {domain:10s}: accepted {len(acc):6d} vs {entry['compared_with']} {len(target):6d}: "
                  f"{'EXACT MATCH' if entry['exact_match'] else 'DIFFERS'}"
                  f"{'' if 'memo_check_without_memo_equal' not in entry else ' (no-memo check: ' + str(entry['memo_check_without_memo_equal']) + ')'}"
                  f"  [{entry['seconds']}s]", flush=True)
        results[str(tid)] = row
        write_json_atomic(args.out, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
