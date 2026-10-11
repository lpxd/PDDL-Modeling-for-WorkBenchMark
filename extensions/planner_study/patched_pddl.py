"""Patched domain (Phase 4) and its problem text. Extension only; src/ untouched.

The rule: a brick may only be placed while nothing that hangs over its goal
cell has been placed yet (same relation as brick_geometry.overhang_map).
Three encodings of that rule live in domain_patched/. ``quantified``
(lego_domain_patched.pddl) is the literal forall/not precondition over a
static ``covers`` fact -- only fd-lama can run it, since Fast Downward
compiles it into an axiom and A*+LM-cut refuses axioms, and pyperplan
doesn't support the syntax at all. ``closing``
(lego_domain_patched_closing.pddl) restates the same rule axiom-free, via
delete effects that close off bricks once something is placed over them --
every engine can run it, but it hides dead ends from delete-relaxation
heuristics (fd-lmcut's Tier-3 task 20 times out as a result). ``prereq``
(lego_domain_patched_prereq.pddl) is a second axiom-free attempt, using a
positive prerequisite instead of a delete effect.

The problem text is the project's own ``render_problem`` output plus the
static facts derived from the goal layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from brick_geometry import overhang_map

DOMAIN_DIR = Path(__file__).resolve().parent / "domain_patched"
DOMAIN_FILES = {
    "quantified": DOMAIN_DIR / "lego_domain_patched.pddl",
    "closing": DOMAIN_DIR / "lego_domain_patched_closing.pddl",
    "prereq": DOMAIN_DIR / "lego_domain_patched_prereq.pddl",
}
# Encodings whose place/stack carry ARITY trailing slot parameters.
SLOT_ENCODINGS = ("closing", "prereq")
ENCODINGS = tuple(DOMAIN_FILES)

# Engines that can solve each encoding.
ENCODING_ENGINES = {
    "quantified": ("fd-lama",),
    "closing": ("pyperplan", "pyperplan-opt", "fd-lama", "fd-lmcut"),
    "prereq": ("pyperplan", "pyperplan-opt", "fd-lama", "fd-lmcut"),
}

# Number of trailing slot parameters of place/stack in the strips encoding: the
# largest number of bricks one brick hangs over in tiers 1-3 is 6 (tier 3 task 3).
ARITY = 6
DUMMY = "b_dummy"

# Parameter counts of the ORIGINAL actions (the strips encoding appends ARITY more).
ORIGINAL_ARITY = {"pick": 7, "place": 7, "stack": 12}


def domain_text(encoding: str) -> str:
    return DOMAIN_FILES[encoding].read_text(encoding="utf-8")


# {brick: [bricks it hangs over]} in the goal layout.
def hangs_over(task: Any) -> dict[str, list[str]]:
    under: dict[str, list[str]] = {}
    for lower, uppers in overhang_map(task).items():
        for upper in uppers:
            under.setdefault(upper, []).append(lower)
    return {k: sorted(v) for k, v in under.items()}


def slots_for(task: Any, brick: str) -> list[str]:
    """The ARITY slot objects of ``brick`` (YAML name): bricks it hangs over, padded."""
    from workbenchmark_pddl.pddl_names import brick_object

    objs = [brick_object(n) for n in hangs_over(task).get(brick, [])]
    if len(objs) > ARITY:
        raise ValueError(f"{brick} hangs over {len(objs)} bricks; ARITY={ARITY} is too small")
    return objs + [DUMMY] * (ARITY - len(objs))


def render_problem_patched(task: Any, encoding: str, problem_name: str | None = None) -> str:
    from workbenchmark_pddl.pddl_names import brick_object
    from workbenchmark_pddl.problem import render_problem

    text = render_problem(task, problem_name=problem_name)
    init_end = "\n  )\n\n  (:goal"  # end of :init in ProblemBuilder.render
    objects_end = "\n  )\n\n  (:init"  # end of :objects
    if text.count(init_end) != 1 or text.count(objects_end) != 1:
        raise RuntimeError("unexpected problem layout: cannot patch :objects/:init")

    if encoding == "quantified":
        facts = [
            f"    (covers {brick_object(upper)} {brick_object(lower)})"
            for lower, uppers in sorted(overhang_map(task).items())
            for upper in sorted(uppers)
        ]
        return text.replace(init_end, ("\n" + "\n".join(facts) if facts else "") + init_end)

    if encoding in SLOT_ENCODINGS:
        from workbenchmark_pddl.pddl_names import ASSEMBLY_AREA

        bricks = [b.name for b in task.initial_blocks]
        if encoding == "closing":
            facts = [f"    (open {brick_object(b)})" for b in bricks] + [f"    (open {DUMMY})"]
            static = "closes"
        else:
            facts = [f"    (in-area {DUMMY} {ASSEMBLY_AREA})"]  # the dummy counts as placed
            static = "hangs-over"
        facts += [f"    ({static} {brick_object(b)} {' '.join(slots_for(task, b))})" for b in bricks]
        text = text.replace(objects_end, f"\n    {DUMMY} - brick" + objects_end)
        return text.replace(init_end, "\n" + "\n".join(facts) + init_end)

    raise ValueError(f"unknown encoding {encoding!r}")


def parse_patched(task: Any, encoding: str, problem_name: str | None = None):
    from unified_planning.io import PDDLReader

    return PDDLReader().parse_problem_string(domain_text(encoding), render_problem_patched(task, encoding, problem_name))


# plans


# Drop the trailing slot parameters (strips encoding) -> a plan for the original domain.
def strip_slots(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"name": a["name"], "params": list(a["params"])[: ORIGINAL_ARITY[a["name"]]]} for a in actions]


# Append the slot parameters (strips encoding) to an original-domain plan.
def add_slots(task: Any, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for a in actions:
        if a["name"] == "pick":
            out.append({"name": "pick", "params": list(a["params"])})
            continue
        top = a["params"][1].removeprefix("b_")  # place: ?b ; stack: ?top (both index 1)
        out.append({"name": a["name"], "params": list(a["params"]) + slots_for(task, top)})
    return out
