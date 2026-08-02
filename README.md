# 🧱 PDDL Modeling for WorkBenchMark

**Project P1** — AI for Robotics, Summer Semester 2026 (RWTH Aachen)

Design a PDDL domain for LEGO Duplo assembly, plan with a general-purpose
planner, and execute in [TAMPanda](../tampanda) via `DomainBridge`.

- 🔗 Benchmark: [WorkBenchMark](https://workbenchmark.github.io)
- 📄 Paper: [arXiv:2606.19358](https://arxiv.org/abs/2606.19358)

## 🧭 What This Project Does

WorkBenchMark ships an Assembly-by-Disassembly (ABD) baseline. We model the
assembly task symbolically in PDDL and let a classical planner produce the
sequence. The generated plan is executed in TAMPanda through `DomainBridge`
and `PickPlaceExecutor`.

Pipeline: **YAML task spec → PDDL problem → planner → action plan → MuJoCo execution**

## 🎯 Scope

What we must deliver, what we aim for if time allows, and what we explicitly leave out.

### ✅ Minimal Requirements

- Tier 1 (2-brick) and Tier 2 (3–5 brick) vertical stacks
- Ground-truth poses from YAML (no perception)
- MuJoCo simulation via TAMPanda



### 🚀 Goal

- PDDL domain designed for Tier 3 up front (stud-grid poses, rotation,
`can-attach` / footprint geometry) — no rewrite later
- Tier 3 (3D layouts, half-overlap, multi-column) if time allows
- Compare planning time and plan quality against the ABD baseline



### 🚫 Not Part of the Project

- Tier 4 complex interlocking assemblies
- Real-robot experiments
- Perception stack (GroundingDINO, SAM, FoundationPose, …)
- Modifying TAMPanda itself



## 🗺️ Phases


| Phase                  | Goal                                                                        | Status         |
| ---------------------- | --------------------------------------------------------------------------- | -------------- |
| **1 — Setup**          | Repo, `pyproject.toml`, package skeleton, env smoke tests                   | ✅ done         |
| **2 — Symbolic layer** | `lego_domain.pddl`, YAML parser, problem generator, planner                 | ✅ done         |
| **3 — Execution**      | `DomainBridge` wiring, predicate grounding, motion executors                | ✅ done         |
| **4 — Evaluation**     | Batch runner, metrics vs. ABD, failure analysis (Tier 1–2; Tier 3 optional) | 🚧 in progress |


**Phase 2 deliverables:** 

- stud-grid PDDL domain with `pick` / `place` / `stack`
- Tier‑3-ready predicates YAML parser for WorkBenchMark `blocks[]` specs
- problem generator from initial + goal YAML
- classical planner (UPF) producing assembly plans

**Phase 3 deliverables:** 

- map PDDL actions to `PickPlaceExecutor` calls in TAMPanda
- resolve each action’s brick to a continuous pose via YAML lookup
- MuJoCo scene with Duplo box proxies (`scene.py`) + table-frame map (`frames.py`)

> **Note:** `stud` / `layer` / `rot` are for planning only.
> Execution does not invert the grid, because reverting is lossy.
> Instead they use the ground-truth YAML poses provided by WorkBenchMark.
> `pick` uses `initial_blocks` as the pose source,
> `place`/`stack` use goal `blocks` as the pose source.

**Phase 4 deliverables:** 

- systematic evaluation on Tier 1–2 (N = 20 tasks per tier)
- planning/execution success, time, plan length vs. ABD
- Tier 3 evaluation if implementation of it is completed.
- Failure analysis.



## 📊 Evaluation Metrics

- ✅ **Planning success** — planner finds a valid plan
- 🤖 **Execution success** — all actions succeed in TAMPanda
- ⏱️ **Planning time & plan length** — vs. ABD baseline where available
- 🔍 **Failure analysis** — which task structures expose model gaps



## 📦 Dataset

WorkBenchMark tasks live in the sibling clone `[../dataset](../dataset)`
([WorkBenchMark/dataset](https://github.com/WorkBenchMark/dataset)):

P1 skips perception, so the YAML→PDDL compiler reads `ground_truth/` for both
initial layout and goal. It is an explicit project simplification.

## ⚙️ Setup

Requires **Python 3.10+** and this sibling layout (names matter for imports / pyright):

```text
parent/
├── venv/                              # shared virtualenv
├── tampanda/                          # TAMPanda (editable install)
├── dataset/                           # WorkBenchMark tasks
└── PDDL-Modeling-for-WorkBenchMark/   # this repo (on the course machine: Project/)
```

### From scratch

```bash
# 1) Parent directory that will hold the siblings
mkdir -p ~/workbench && cd ~/workbench

# 2) Clone siblings
git clone https://github.com/snoato/manipulation.git tampanda
git clone https://github.com/WorkBenchMark/dataset.git dataset
git clone https://github.com/IX0Y3/PDDL-Modeling-for-WorkBenchMark.git
cd PDDL-Modeling-for-WorkBenchMark

# 3) Create the shared venv next to this repo (not inside it)
python3.12 -m venv ../venv
source ../venv/bin/activate
python -m pip install -U pip setuptools wheel

# 4) Editable installs + QP solver used by TAMPanda IK (mink)
python -m pip install -e ../tampanda --config-settings editable_mode=compat
python -m pip install -e .[dev]    # quote for zsh: ".[dev]"
python -m pip install quadprog

# 5) Smoke-check the toolchain
python scripts/check_env.py
pytest -v              
```

> **Note:** `pytest` may take several minutes: the suite includes full MuJoCo e2e smokes
> (Tier 1–3). For a fast check without those:

```bash
pytest -v -m "not slow"
```

Use `python -m pip` (not a bare system `pip3`) so installs go into the venv!

## 📂 Evaluation (Results)

Both the planning and execution results are stored in the `results` directory.

They can be recreated at any time by following the instructions below.

> **Note:**  Planninig and execution are very time-consuming, 
> so a full set of results is already available in the `results` directory.

The result directory layout is as follows:

```
results/
├── planning_summary.jsonl
├── planning_summary.csv
├── execution_summary.jsonl
├── execution_summary.csv
├── tier1/
│   ├── task_001/
│   │   ├── plan.json
│   │   ├── planning_metrics.json
│   │   └── execution_metrics.json
│   └── ...
├── tier2/
│   ├── task_001/
│   │   ├── plan.json
│   │   ├── planning_metrics.json
│   │   └── execution_metrics.json
│   └── ...
└── tier3/
    ├── task_001/
    │   ├── plan.json
    │   ├── planning_metrics.json
    │   └── execution_metrics.json
    └── ...
```



## 🧠 Evaluation (planning)

Batch planning only (no Simulation). Tiers 1, 2, and 3, first 20 tasks each.

> **Note:** Each solved task generates a `plan.json` and `planning_metrics.json` file.
> It also creates a summary file `planning_summary.jsonl` and `planning_summary.csv`.
> This file is updated incrementally.

To evaluate all tasks:

```bash
python scripts/eval_planning.py # optional: --tiers 1,2,3 --n 20 --out-dir results
```

To (re)evaluate a specifc task:

```bash
python scripts/eval_planning.py --task 3/20
python scripts/eval_planning.py --task tier3/task_020
```



## 🤖 Evaluation (execution)

Uses saved `plan.json` from the planning eval (no re-planning) and runs them
in MuJoCo via DomainBridge / PickPlace. Default: Tier 1–3, N = 20.

> **Note:** Each task writes `execution_metrics.json` immediately and updates
> `execution_summary.jsonl` / `execution_summary.csv`.

To evaluate all default tasks:

```bash
python scripts/eval_execution.py
# or: python scripts/eval_execution.py --tiers 1,2,3 --n 20 --out-dir results
```

To (re)evaluate a specific task:

```bash
python scripts/eval_execution.py --task 1/1
python scripts/eval_execution.py --task tier2/task_005
```

> **Note:** Use **tmux** for long runs (`tmux new -s exec`, detach with Ctrl+b d, reattach with `tmux attach -t exec`).

