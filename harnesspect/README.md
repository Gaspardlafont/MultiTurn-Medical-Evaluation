# MultiTurn-Medical-Evaluation
# Unified Multi-Turn Medical Agent Simulation Framework

## Context

This project is part of the **LiGHT lab at EPFL**, supervised by Yusuf Kesmen, Fabrice Nemo and Xavier Theimer-Lienhard.

## Problem Statement

LLMs perform well on static medical benchmarks (MedQA, etc.) where the full case is given upfront. However, they degrade significantly in **interactive, multi-turn settings** recognizing missing information and actively seeking it over multiple turns which is what clinical reasoning under uncertainty actually requires.

Several agentic simulation frameworks already exist for this (AgentClinic, MediQ, MEDDxAgent,...), each with a patient simulator holding hidden case information and a doctor agent under evaluation. The problem: they are all **standalone research codebases with incompatible abstractions**, making it painful to adopt any of them into a new pipeline or compare across them systematically.

---

## Objective

Build a **unified harness** that runs multiple patient simulators behind a single, uniform interface so any conforming doctor agent can be paired with any conforming patient simulator.

---

## Project Structure

### Stage 1 — Engineering (~4 weeks)

Build the unified framework:

- **Analyze existing frameworks** (AgentClinic, MediQ, MEDDxAgent, PatientSim) and identify their common anatomy: roles (patient / doctor / measurement / grader agents), hidden case state, turn logic, termination conditions
- **Investigate existing evaluation tools** (LM Harness, Inspect) to determine whether to build on top of one or start from scratch
- **Define a minimal shared backbone** with a uniform request/response schema and config-file-based simulator definitions — rather than wrapping upstream code with adapters
- **CLI orchestrator**: ability to run full simulations or manually probe any patient simulator
- **Full trace logging**: system prompts, case data, and turn-by-turn transcripts fully inspectable, with metrics output

### Stage 2 — Research (TBD)

The harness enables several research directions:

1. **Failure mode analysis of patient simulators**: info leakage, hallucinated symptoms, cross-turn inconsistency, susceptibility to leading questions — using identical probes across all simulators
2. **Multi-turn capabilities of open medical models**: e.g. Meditron as doctor agent across all simulators, benchmarked against proprietary models
3. **Reinforcement learning for multi-turn Meditron**: improving Meditron's ability to reason across turns via RL ++ this one can be very interesting
4. **A fully open multi-turn evaluation pipeline** with no closed-model dependencies
5. **A new patient simulator** designed protocol-first from the weaknesses found in (1)

---

## Running the Wrappers

Both wrappers live in `harnesspect/wrapped_inspect/`, run via Inspect's CLI. Full options are documented in each file's own docstring — the summary below just covers the base command.

### MediQ

```
inspect eval wrapped_inspect/inspect_mediq_wrapped.py --model vllm/Qwen/Qwen2.5-7B-Instruct
```

Requires `stellalisy/mediQ` cloned as a sibling of this repo (`../../mediQ`). Configurable via `-T name=value`: `dataset_path`, `limit`, `max_questions`, `expert_class`, `abstain_threshold`, `rationale_generation`, `self_consistency`.

### AgentClinic

```
inspect eval wrapped_inspect/inspect_agentclinic_wrapped.py --model vllm/Qwen/Qwen2.5-7B-Instruct
```

Requires `SamuelSchmidgall/AgentClinic` cloned as a sibling of this repo (`../../agentclinic`). Configurable via `-T name=value`: `dataset`, `limit`, `max_turns`, `doctor_bias`, `patient_bias`, `doctor_image_request`.

### Both

- `--model` sets the fallback model for every role; pin specific roles to different models with `--model-role <role>=<model>` (e.g. `--model-role expert=vllm/EPFLiGHT/Apertus-8B-MeditronFO`).
- Two local vLLM servers sharing one GPU need capped memory each, e.g. `--model-role expert="{model: vllm/..., model_args: {gpu_memory_utilization: 0.45}}"`, or the second server fails to start.
- View results with `inspect view --log-dir logs`.

---

## Team

| Name | Role |
|------|------|
| Yusuf Kesmen | Research supervisor (PhD student, LiGHT) |
| Fabrice Nemo | Co-supervisor (Applied AI Coordinator, LiGHT) |
| Xavier Theimer-Lienhard | Co-supervisor (Research Engineer, LiGHT) |
| Gaspard | Student |
| Zacharie | Student |

