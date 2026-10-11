# Planner study extension

Does the PDDL model still hold up if you swap the planner, or if the
planner happens to return a different (but still valid) order? Separate
extension, imports from `src/workbenchmark_pddl` only — nothing in `src/`,
`scripts/`, the existing `results/`, `../tampanda` or `../dataset` gets
touched. Everything this extension produces goes under `results/extension/`.

## Engines

Run from the project root with the shared venv (`..\venv`).

| Label | UPF engine | What it is |
|---|---|---|
| `pyperplan` | `pyperplan` | weighted A* + hadd, satisficing, pure Python |
| `pyperplan-opt` | `pyperplan-opt` | A* + LM-cut, optimal, pure Python |
| `fd-lama` | `fast-downward` | Fast Downward, alias `lama-first`, satisficing |
| `fd-lmcut` | `fast-downward-opt` | Fast Downward, A* + LM-cut, optimal |

Couldn't get these two working natively on Windows, so they're not in the
comparison:
- **SymK** (`up-symk`): the Windows wheel ships no compiled solver
  ("Could not find build 'release'") — would need a source build or WSL2.
- **Tamer** (`up-tamer`): its native `_pytamer` DLL just won't load.

Installed extras: `up-fast-downward`, `up-symk`, `up-tamer`, and
`setuptools<81` (needed so `up_symk` imports; not actually used for results).

## Phase 1: `run_planners.py`

```powershell
# smoke test (2-3 tier-1 tasks, all engines)
python extensions/planner_study/run_planners.py --task 1/1,1/2,1/3

# full sweep, tiers 1-3, N=20, all engines, 600 s per task
python extensions/planner_study/run_planners.py

# one engine / one tier / one task
python extensions/planner_study/run_planners.py --planners fd-lama --tiers 3
python extensions/planner_study/run_planners.py --task tier3/task_020 --planners pyperplan
```

Options: `--planners`, `--tiers`, `--n`, `--task`, `--timeout` (600 s by
default, same for every engine), `--out-dir`, `--force`.

Each solve runs in a worker subprocess that gets killed outright at the
timeout, because pyperplan just ignores UPF's `timeout` argument — without
this it would hang forever on a hard task. A timeout gets its own status
(`timeout`), it's not counted as a failure, and `planning_time_s` is `null`
since we only know the true time exceeds `timeout_s`.

It's resumable — finished tasks are skipped, and a `timeout` only gets
retried if you pass a bigger `--timeout` next time, so you're not stuck
re-running everything just to retry the one hard task. `--force` redoes a
task anyway.

For long runs, just leave it going in a background terminal — safe to kill
and restart anytime. Don't run several planner processes at once if you
want the timings to mean anything, though (they'll fight over CPU).

### Output

```
results/extension/planners/
  planning_summary.csv / .jsonl        # rebuilt from disk after every task
  <planner>/tierN/task_XXX/
    plan.json                          # same format as the main project
    planning_metrics.json              # one record, fields below
```

`planning_metrics.json` fields: `planner`, `engine`, `tier`, `task_id`,
`status` (`solved` / `timeout` / `failed`), `upf_status` (raw UPF status),
`plan_length`, `planning_time_s`, `wall_s`, `timeout_s`, `plan_valid`, `yaml`,
`detail`.

- `planning_time_s` is defined the same way as in `scripts/eval_planning.py`
  (render + parse + solve, not counting the YAML load).
- `wall_s` also includes interpreter startup/imports (~3.5 s per attempt),
  so don't read it as a planning metric.
- `plan_valid` is just UPF's `sequential_plan_validator` run on the returned
  plan, checked after the timer's already stopped.

### Comparing with the existing results

The times in `results/tierN/.../planning_metrics.json` come from a
different machine (macOS) — don't compare them to anything from this
extension directly. That's also why the extension re-runs `pyperplan`
itself instead of reusing those numbers.

## Phase 3: order sensitivity (`order_sensitivity.py`, `brick_geometry.py`)

For every Tier-3 task: enumerate every ordering of one plan's units that
the PDDL model itself would call valid, then classify each one
geometrically — a brick can only go in if nothing that hangs over it
(higher layer, overlapping footprint, worked out purely from the goal
layout in `brick_geometry.py`) is already placed. No simulator anywhere in
this step.

```powershell
python extensions/planner_study/order_sensitivity.py              # enumerate + sample + validate, all tier-3 tasks
python extensions/planner_study/order_sensitivity.py --summarize-only   # rebuild the CSVs from existing variants.json
```

Output: `results/extension/order_sensitivity/order_counts.json` (exact
counts per task), `tierN/task_XXX/variants.json` (the sampled orderings,
each checked against the model with UPF's `sequential_plan_validator`),
`variant_summary.csv`, `task_summary.csv`.

## Phase 4: domain patch (`domain_patched/`, `patched_pddl.py`)

The rule: a brick that hangs over another brick (higher layer, overlapping
footprint, from the goal layout) has to be placed after it. The original
domain stays untouched; the patch lives in copies under `domain_patched/`.

Three ways to write that same rule, because no single one works for every
engine:

- **`quantified`** (`lego_domain_patched.pddl`) — the literal version, a
  `forall`/`not` over a static `covers` fact. Only `fd-lama` can run it:
  pyperplan doesn't support the syntax at all, and Fast Downward compiles it
  into an axiom, which A*+LM-cut flatly refuses ("This configuration does
  not support axioms!").
- **`closing`** (`lego_domain_patched_closing.pddl`) — axiom-free rewrite,
  every engine can run it. Placing a brick closes the bricks it hangs over
  via delete effects (six padded trailing parameters — the most any one
  brick hangs over in Tiers 1-3).
- **`prereq`** (`lego_domain_patched_prereq.pddl`) — second axiom-free
  attempt, a positive prerequisite instead of a delete effect. Only tried on
  tasks 1 and 20.

Slot parameters get dropped again (`patched_pddl.strip_slots`) before a plan
actually runs. `run_planners.py --domain {quantified,closing,prereq}` plans
against a patched domain (writes to
`results/extension/planners_patched_<domain>/`, raw plans with slots kept in
`plan_raw.json`).

```powershell
python extensions/planner_study/patched_orderings.py      # (a) accepted orderings vs feasible set, all domains
python extensions/planner_study/run_planners.py --domain quantified --planners fd-lama --tiers 3
python extensions/planner_study/run_planners.py --domain closing --planners fd-lama,fd-lmcut --tiers 3
python extensions/planner_study/phase4_report.py          # all numbers in one file
```

`patched_orderings.py` enumerates, with UPF's sequential simulator, every
ordering a given domain accepts and checks it against the geometrically
feasible set. It tells you the encoding implements the rule correctly — not
that the rule itself is physically necessary.

## Phase 5: `analyze.py`

Pulls the result files above into the tables actually used in the
presentation. Nothing new gets computed here, it just reads what the
earlier scripts already wrote.

```powershell
python extensions/planner_study/analyze.py
```

Writes `planning_tier3.csv`, `planning_tiers1_2_compact.csv`, and
`domain_patch_before_after.csv` to `results/extension/comparison/`. It also
still computes the order-sensitivity and pyperplan-on-patched-domain
findings and prints them to the console, but doesn't write them as CSVs
anymore, since neither ended up as a table in the presentation — the raw
numbers behind them are still sitting in
`results/extension/order_sensitivity/` and
`results/extension/patched/pyperplan_on_patched.json` if you need them.

See `EVALUATION_EXTENSION.md` for the full write-up built from these numbers.
