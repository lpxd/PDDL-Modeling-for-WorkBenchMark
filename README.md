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

## 🎯 Deliverables

- Tier 1 (2-brick), Tier 2 (3–5 brick), and Tier 3 (3D layouts, half-overlap,
multi-column) assembly tasks
- Stud-grid PDDL domain (`pick` / `place` / `stack`) with rotation and
`can-attach` / footprint geometry
- Ground-truth poses from WorkBenchMark YAML (no perception)
- Classical planning via Unified Planning (UPF)
- MuJoCo simulation via TAMPanda (`DomainBridge` → `PickPlaceExecutor`)
- Batch planning and execution evaluation (tiers 1–3, N = 20) with metrics
under `results/`
- Interactive MuJoCo viewer to replay a single task (`scripts/show_execution.py`)

> **Note:** `stud` / `layer` / `rot` are for planning only.
> Execution does not invert the grid (lossy). Pose sources:
> `pick` → `initial_blocks`, `place`/`stack` → goal `blocks`.



## 🚫 Not Part of the Project

- Tier 4 complex interlocking assemblies
- Real-robot experiments
- Perception stack (GroundingDINO, SAM, FoundationPose, …)
- Modifying TAMPanda itself



## 📊 Evaluation

See **[EVALUATION.md](EVALUATION.md)** for metrics definitions and the full
tiers 1–3 (N = 20) planning / execution results.

Quick reference:

- ✅ **Planning success** — planner finds a valid plan
- 🤖 **Execution success** — all actions succeed in TAMPanda
- ⏱️ **Planning time & plan length**
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

> **Note:** Planning and execution are very time-consuming,
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

Batch planning only (no simulation). Default: Tier 1–3, N = 20.

> **Note:** Each solved task generates a `plan.json` and `planning_metrics.json` file.
> It also creates a summary file `planning_summary.jsonl` and `planning_summary.csv`.
> This file is updated incrementally.

To evaluate all tasks:

```bash
python scripts/eval_planning.py # optional: --tiers 1,2,3 --n 20 --out-dir results
```

To (re)evaluate a specific task:

```bash
python scripts/eval_planning.py --task 3/20
python scripts/eval_planning.py --task tier3/task_020
```



## 🤖 Evaluation (execution)

Uses saved `plan.json` from the planning eval (no re-planning) and runs them
in MuJoCo via DomainBridge / PickPlace. Default: Tier 1–3, N = 20.

> **Important:** Run planning first (`python scripts/eval_planning.py`).
> Execution loads existing `results/tierN/task_XXX/plan.json` and will skip
> or fail tasks that have no saved plan.

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



## 👀 Show execution (viewer)

Replay a single task in MuJoCo with the interactive viewer.
Does **not** write or overwrite any metrics / summaries.

Uses `results/tierN/task_XXX/plan.json` when present — otherwise plans before execution.

```bash
python scripts/show_execution.py --task 1/1
python scripts/show_execution.py --task tier2/task_005

# macOS: must use mjpython (not plain python) for the MuJoCo viewer
mjpython scripts/show_execution.py --task 1/1
mjpython scripts/show_execution.py --task tier2/task_005
```

Close the viewer window when done.