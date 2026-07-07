# MultiTurn-Medical-Evaluation
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

## Inspect AI — Key Building Blocks

Candidate foundation for Stage 1. Quick reference for the pieces we'd actually use to implement the doctor/patient harness.

| Name | What it is | Why it matters here |
|---|---|---|
| `Task` | Bundles `dataset` + `solver` + `scorer` + `model` + limits into one runnable recipe (`@task` decorated function) | The top-level unit `eval()` runs — one `Task` per framework (MediQ, AgentClinic, ...) |
| `Sample` / `Dataset` | One case (`input`, `target`, free-form `metadata`) / a list of them | How each framework's raw case files (jsonl, etc.) get loaded in |
| `Solver` | `(TaskState, generate) -> TaskState`, chainable steps run in sequence for a sample | The classic building block; a whole doctor/patient loop can be written as one custom `@solver` |
| `Agent` / `AgentState` | `(AgentState) -> AgentState` — narrower than `Solver` (just messages + output, no dataset/target coupling) | The right shape for a reusable "patient" or "doctor" participant, usable standalone, as a tool, or delegated to |
| `@agent` / `@solver` / `@scorer` / `@task` | Decorators that register a factory function by name in a global registry | Lets components be referenced by name from the CLI/config instead of imported directly |
| `react()` | Built-in ReAct agent: tool-use loop + a synthesized `submit` tool + retry/attempts + context overflow handling | Good ready-made scaffold for the **doctor** role — `submit` replaces AgentClinic's fragile `"DIAGNOSIS READY:"` string parsing |
| `as_tool()` | Wraps an `Agent` as a `Tool` that sees only a single input string and returns the agent's last message | Right fit for the **patient**: it only answers the question it's given, never sees or drives the full conversation |
| `handoff()` | Wraps an `Agent` so it gets full conversation visibility + can append messages / take control | **Not** the right fit for a passive patient — this is for delegating to an autonomous sub-agent (contrast with `as_tool()`) |
| `run()` | Runs an `Agent` once given an input (string / messages / `AgentState`) | Needed if we hand-write the doctor↔patient turn-taking loop ourselves (no built-in "converse until X" helper exists) |
| `Scorer` / `Score` / `Target` / `model_graded_qa()` | `(TaskState, Target) -> Score`; `model_graded_qa()` is a ready-made LLM-judge scorer | Covers AgentClinic-style binary LLM grading out of the box; ranked-list (MEDDxAgent-style) scoring still needs a custom `@scorer` |
| `Tool` / `@tool` | Custom function exposed to a model as a callable tool | For a "measurement/exam" side-channel role, equivalent to AgentClinic's `MeasurementAgent` |
| `turn_limit()` / `message_limit()` / `token_limit()` / `time_limit()` / `apply_limits()` | Native, **ambient/cooperative** limits — every `generate()` call in scope counts against them automatically | Replaces hand-rolled counters like AgentClinic's `self.infs`; works across a hand-written multi-agent loop too |
| `eval()` + `.eval` log + Inspect view | Runner + structured log format + built-in web viewer | Full trace logging (system prompts, transcripts, metrics) for free — the biggest gap in all 4 upstream frameworks today |

**Caveat worth remembering**: `as_tool()` builds a *fresh* `AgentState` on every call — a patient exposed this way has no memory of earlier questions unless we manage its running history ourselves (closure-captured list, same trick AgentClinic uses with `self.agent_hist`).

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

- **Gaspard & Zacharie**: investigate LM Harness and Inspect can they support multi-turn dialogue, hidden patient state, and full trace logging?
- **Gaspard**: research existing multi-turn medical benchmarks (AgentClinic, MediQ, HealthBench) and identify common structures
- **Gaspard**: search for multi-turn benchmarks in African/Indian languages
- **Gaspard & Zacharie**: write a summary of findings and meet with Fabrice Monday morning to establish a coding plan

