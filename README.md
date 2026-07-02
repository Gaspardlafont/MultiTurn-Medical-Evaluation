# MultiTurn-Medical-Evaluation-
# Unified Multi-Turn Medical Agent Simulation Framework

## Context

This project is part of the **LiGHT lab at EPFL**, supervised by Yusuf Kesmen, Fabrice Nemo and Xavier Theimer-Lienhard.

## Problem Statement

LLMs perform well on static medical benchmarks (MedQA, etc.) where the full case is given upfront. However, they degrade significantly in **interactive, multi-turn settings** recognizing missing information and actively seeking it over multiple turns which is what clinical reasoning under uncertainty actually requires.

Several agentic simulation frameworks already exist for this (AgentClinic, MediQ, MEDDxAgent, PatientSim), each with a patient simulator holding hidden case information and a doctor agent under evaluation. The problem: they are all **standalone research codebases with incompatible abstractions**, making it painful to adopt any of them into a new pipeline or compare across them systematically.

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

## Current Status

- [x] Initial project proposal by Yusuf (Stage 1 + Stage 2 directions)
- [x] Kick-off meeting with Yusuf, Fabrice, Xavier — agreement on backbone approach over adapter wrappers
- [ ] Investigation of LM Harness and Inspect as potential foundations
- [ ] Summary of findings + coding plan (due Monday)
- [ ] Stage 1 implementation

---

## Team

| Name | Role |
|------|------|
| Yusuf Kesmen | Research supervisor (PhD student, LiGHT) |
| Fabrice Nemo | Co-supervisor (Applied AI Coordinator, LiGHT) |
| Xavier Theimer-Lienhard | Co-supervisor (Research Engineer, LiGHT) |
| Gaspard | Student |
| Zacharie | Student |

---

## Next Steps (before Monday)

- **Gaspard & Zacharie**: investigate LM Harness and Inspect — can they support multi-turn dialogue, hidden patient state, and full trace logging?
- **Gaspard**: research existing multi-turn medical benchmarks (AgentClinic, MediQ, HealthBench) and identify common structures
- **Gaspard**: search for multi-turn benchmarks in African/Indian languages
- **Gaspard & Zacharie**: write a summary of findings and meet with Fabrice Monday morning to establish a coding plan