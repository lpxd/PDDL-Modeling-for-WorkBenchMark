# Planner study: key findings

All numbers come from files under `results/extension/`. Feasibility is
checked analytically throughout (goal-layout geometry, `brick_geometry.py`),
never by simulation.

## 1. Planner comparison (planning only)

Four engines, tiers 1-3, 20 tasks each, 600 s limit, this machine.

| Engine | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| fd-lama | 20/20, median 1.3 s | 20/20, 1.8 s | 20/20, 8.2 s (max 79 s) |
| fd-lmcut | 20/20, 1.3 s | 20/20, 1.8 s | 20/20, 7.6 s (max 284 s) |
| pyperplan | 20/20, 0.8 s | 20/20, 1.4 s | 19/20 (1 timeout), 18.6 s |
| pyperplan-opt | 20/20, 0.8 s | 20/20, 1.3 s | 17/20 (3 timeouts), 23.5 s |

- 236/240 attempts solved, 4 timeouts, 0 failures; every plan passed the validator.
- Timeouts are a speed artifact, not infeasibility: every task is solved by at
  least one engine.
- Plan length never differed between engines (each brick is moved once).
- Tiers 1-2: all engines return the same plan. Tier 3: every task has at
  least 2 distinct plans across the 4 engines.
- SymK and Tamer do not work on native Windows here, so this is two solver
  codebases (pyperplan, Fast Downward), not four independent planners.
- Times are not comparable with the ones in `EVALUATION.md` (other, faster
  machine; measured directly task-by-task on the 19 commonly-solved Tier-3
  tasks, the ratio is 2.3x-5.0x, median/mean ~3.9x).

## 2. Order sensitivity (Tier 3): the model's own numbers

For each of the 20 Tier-3 tasks we took the fd-lama plan and reordered its
pick+place units. Every ordering valid for the PDDL model was enumerated
(exact counts), and up to 5 orderings other than the original were sampled
per task (93 in total), each validated with UPF's `sequential_plan_validator`.

- **Exact ordering counts:** every Tier-3 task has fewer geometrically
  feasible orders than PDDL-valid orders (median 40% feasible, range 7.6% to
  83.3%); every task has at least one feasible order (`order_counts.json`).
- **Pooled over all 20 tasks: 5,147 of 58,244 PDDL-valid orderings (8.8%) are
  geometrically feasible**, dominated by task 20's large combinatorial space.
- **No engine's own plan reliably lands in that feasible slice:** fd-lama
  3/20 (15%), fd-lmcut 2/20 (10%), pyperplan 0/19 (0%), pyperplan-opt 3/17
  (18%) of their own solved tasks. All four cluster in the same low range —
  none of their search heuristics represents this constraint.
- **Why it happens:** the PDDL `stack` action names one support, so a plan
  may place a brick's second support after the brick itself, i.e. under it.
  The model does not say that a brick resting on two bricks needs both in
  place first.
- **Reading:** the PDDL model accepts many orderings that cannot be built (a
  median of 60% per task), and picking a different planner does not
  meaningfully change how often that happens. Plan order matters for
  buildability only through one missing constraint.

## 3. Domain patch (Phase 4)

**The rule:** a brick that hangs over another brick (higher layer, overlapping
footprint, from the goal layout) must not be placed before it. It is added in
*copies* of the domain (`extensions/planner_study/domain_patched/`); the
project's domain is untouched. Same rule, three encodings:

| Encoding | In one sentence | Engines |
|---|---|---|
| `quantified` | `place`/`stack` require that no brick that hangs over the brick being placed is already in the assembly area (a `forall`/`not` precondition over a static `covers` fact). | `fd-lama` only |
| `closing` | Placing a brick marks every brick it hangs over as closed (delete effects on padded extra parameters), and a brick can only be placed while it is still open. | all four engines |
| `prereq` | `place`/`stack` require that every brick the placed brick hangs over is already placed (positive preconditions on padded extra parameters); the same orderings for complete plans, with no dead ends. | all four engines |

**Why `fd-lmcut` needed a different encoding than `fd-lama`:** Fast Downward
turns the quantified, negated precondition into axioms, and A* with LM-cut
refuses axioms ("This configuration does not support axioms!"); `fd-lama` has
no such limit. `closing` and `prereq` express the rule with plain STRIPS
preconditions/effects, so LM-cut (and pyperplan) can read them. The price is six
extra parameters on `place`/`stack` (six = the largest overhang in tiers 1-3),
which are dropped again before a plan is executed. `closing`'s extra planning
cost traces to LM-cut's heuristic being blind to its delete-effect-based
mechanism (the heuristic is computed via the delete relaxation, which ignores
exactly that mechanism) — not to `fd-lmcut`'s search being slower in general.

**pyperplan on the patched domain (recorded in `pyperplan_on_patched.json`):**
`quantified`: both pyperplan engines fail with `UPUnsupportedProblemTypeError`
(the domain parses; pyperplan does not support the universal/negative
precondition). `closing` and `prereq`: both pyperplan engines solve tier 1 task 1
and tier 3 task 1 with rule-feasible plans (those two tasks only; not run at scale).

**(a) Does the patched domain accept exactly the feasible orderings?** For each
Tier-3 task, every ordering of the units of one plan that a domain accepts was
enumerated by stepping through UPF's sequential simulator (independent of the
Python rule) and compared. Each row compares a different target:

| Domain | Accepted orderings compared with | Result |
|---|---|---|
| original | the PDDL-valid set (Phase 3's Python enumeration) | exact match on 20/20 tasks (58,244 orderings in total) |
| quantified, closing, prereq | the geometrically feasible set | exact match on 20/20 tasks each (5,147 orderings in total) |

The original domain's accepted set is *larger* than the feasible set, by design:
it accepts 53,097 orderings that are not feasible (every task has some). This is
the negative control: the comparison can tell the domains apart. What (a) shows is
that the patched encodings implement the rule correctly; it is a check on the
encodings, not an independent physical confirmation that the rule itself is
necessary (geometric overlap alone already makes that physically self-evident).

**(b) Planning before/after, Tier 3** (single runs on this machine; task 20 was
also retried with longer limits):

| Domain | Engine | Timeout | Solved | Time median / mean / max (s) | Mean plan length | Rule-feasible plans |
|---|---|---|---|---|---|---|
| original | fd-lama | 600 s | 20/20 | 8.2 / 13.7 / 78.8 | 12.0 | 3/20 |
| quantified | fd-lama | 600 s | **20/20** | 8.4 / 15.8 / 105.4 | 12.0 | **20/20** |
| closing | fd-lama | 600 s (task 20 also 1800 s) | 19/20 | 29.8 / 76.7 / 288.9 | 11.5* | 19/19 |
| original | fd-lmcut | 600 s | 20/20 | 7.6 / 24.3 / 284.0 | 12.0 | 2/20 |
| closing | fd-lmcut | 600 s (task 20 also 1800 s) | **19/20** | 23.5 / 72.7 / 269.5 | 11.5* | 19/19 |

\* mean over the 19 solved tasks (task 20 is the longest plan); per task the plan
length is identical to the original in every task solved in both (20/20, 19/19,
19/19). `prereq` was only run on tier 3 tasks 1 and 20 (900 s): task 1 solved
(about 22 s for both engines), task 20 timed out for both.

- **Coverage:** 20/20 holds for `fd-lama` with the literal `quantified` encoding
  (about the same time as before). It does **not** hold for `fd-lmcut`: on task 20
  neither axiom-free encoding solves it within 1800 s (`closing`) or 900 s
  (`prereq`); the original domain needs 284 s. Grounding is not the cause (the
  translator produces the same 72,220 operators in 57-74 s for original,
  quantified and closing); the search is slower for the reason given above.
- **Cost of the axiom-free encodings:** about 3x at the Tier-3 median, 1.6x on
  tier 2, 1.15x on tier 1.
- **Effect:** the share of rule-feasible plans goes from 3/20 (fd-lama) and 2/20
  (fd-lmcut) to 20/20 and 19/19 solved. Every patched plan is valid for the
  original domain and for the patched domain.

**(c) Regression, Tiers 1-2:** the patched plans are identical to the original
plans in all 120 of 120 cases.

## Caveats

- Feasibility is an analytic, geometric check derived from the goal layout
  (overlap + layer comparison), not a simulator-verified one. No physical
  execution, stability, or arm-reachability check is run anywhere in this
  document.
- No ABD numbers are used; they are not in the repository.
- Tier 3 only for order sensitivity and the patch experiments; tiers 1-2 have
  a forced order.
- The axiom-free patch encodings do not cover Tier-3 task 20 for `fd-lmcut`.
