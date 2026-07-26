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
| **3 — Execution**      | `DomainBridge` wiring, predicate grounding, motion executors                | 🚧 in progress |
| **4 — Evaluation**     | Batch runner, metrics vs. ABD, failure analysis (Tier 1–2; Tier 3 optional) | ⏳ pending      |


**Phase 2 deliverables:** 

- stud-grid PDDL domain with `pick` / `place` / `stack`
- Tier‑3-ready predicates YAML parser for WorkBenchMark `blocks[]` specs
- problem generator from initial + goal YAML
- classical planner (UPF) producing assembly plans

**Phase 3 deliverables:** 

- map PDDL actions to `PickPlaceExecutor` calls in TAMPanda
- resolve each action’s brick to a continuous pose via YAML lookup

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

Python 3.10+, TAMPanda at `../tampanda`, dataset at `../dataset`.

```bash
cd /work/rleap1/aifr/prak4/Project
source /work/rleap1/aifr/prak4/venv/bin/activate
pip install -e ../tampanda
pip install -e .[dev]
python scripts/check_env.py
pytest
```

