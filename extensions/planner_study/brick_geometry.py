"""Goal-layout geometry for the planner study (no MuJoCo, no TAMPanda).

Used to explain WHY an ordering is physically blocked, independently of the
simulator: a brick that is placed after a brick that overhangs its goal cell
cannot be lowered into place from above.
"""

from __future__ import annotations

from workbenchmark_pddl.execution import half_size_for
from workbenchmark_pddl.grid import LAYER_HEIGHT_M
from workbenchmark_pddl.task import BrickSpec, TaskSpec

_EPS = 1e-4  # metres; footprint overlap slack (touching edges do not overlap)


def footprints_overlap(a: BrickSpec, b: BrickSpec) -> bool:
    ahx, ahy, _ = half_size_for(a)
    bhx, bhy, _ = half_size_for(b)
    return (
        abs(a.pos[0] - b.pos[0]) < ahx + bhx - _EPS
        and abs(a.pos[1] - b.pos[1]) < ahy + bhy - _EPS
    )


def layer_of(brick: BrickSpec) -> int:
    return round(brick.pos[2] / LAYER_HEIGHT_M)


# {brick: bricks directly below it in the GOAL layout (footprint overlap, one
# layer down)}: the legitimate stud contacts of that brick.
def geometric_supports(task: TaskSpec) -> dict[str, set[str]]:
    blocks = list(task.goal_blocks)
    return {
        top.name: {
            low.name
            for low in blocks
            if low.name != top.name
            and layer_of(low) == layer_of(top) - 1
            and footprints_overlap(top, low)
        }
        for top in blocks
    }


# {brick: bricks that hang over its goal cell} in the GOAL layout: every other
# brick on a higher layer whose footprint overlaps this brick's footprint.
# Whichever of them is placed before the brick blocks its vertical drop.
def overhang_map(task: TaskSpec) -> dict[str, set[str]]:
    blocks = list(task.goal_blocks)
    return {
        me.name: {
            other.name
            for other in blocks
            if other.name != me.name
            and layer_of(other) > layer_of(me)
            and footprints_overlap(other, me)
        }
        for me in blocks
    }


# Bricks in ``placed`` that already hang over the cell ``name`` must be lowered into.
def overhangers(task: TaskSpec, name: str, placed: set[str]) -> set[str]:
    return overhang_map(task)[name] & set(placed)
