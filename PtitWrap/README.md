# PtitWrap — an lm-eval-harness-style harness for multi-turn medical benchmarks

Unifies interactive medical benchmarks (MediQ, AgentClinic, …) behind one
interface, copying the abstraction that makes
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
work: **a model layer and a task layer that only know each other through one
narrow contract**, both referenced by name from the CLI.

## The two abstractions (mirroring lm-eval)

| lm-eval | PtitWrap | role |
|---|---|---|
| `LM` (`api/model.py`) | `LM` (`models/base.py`) | model interface: takes text, returns text |
| `@register_model` / `get_model` | same | reference models by name |
| `--model_args "k=v,k=v"` → `create_from_arg_string` | same | build a model from a CLI string |
| `Task` (`api/task.py`) | `MultiTurnTask` (`adapters/base.py`) | wraps a benchmark |
| `@register_task` / `get_task` | same | reference tasks by name |

**The one deviation for multi-turn:** lm-eval's core model primitive is
`generate_until(batch_of_requests)` — all requests known up front. Multi-turn
can't batch (the doctor's next question depends on the patient's last answer),
so our `LM` primitive is a single **chat** completion, `chat(messages) -> str`,
called turn-by-turn *inside* each benchmark's own loop.

## Model backends (`models/`)

Two ways to bring a model, both registered by name; heavy deps import lazily.

- **`openai-chat`** (aliases `api`, `local-chat-completions`) — any
  OpenAI-compatible `/v1/chat/completions` endpoint. Covers the OpenAI API,
  other hosted APIs, **and** a local vLLM *server*. "Bring your API key."
- **`vllm`** — loads a HuggingFace checkpoint **in-process** with `vllm.LLM`,
  no server to manage. "Bring your HF model."

## Task adapters (`adapters/`)

Each task runs the **upstream benchmark unmodified**. Both benchmarks funnel
every model call through a single function, so the adapter just replaces that
one function at runtime with a shim backed by our `LM` (monkeypatch, zero edits
to upstream):

- **`mediq`** → patches `helper.get_response`, drives MediQ's own
  `run_patient_interaction` loop.
- **`agentclinic`** → patches `query_model`, calls AgentClinic's own `main()`;
  wraps `compare_results` to capture accuracy.

Adding a benchmark = one new adapter that (1) patches its model-call function
and (2) invokes its loop. No new model code, nothing changed upstream.

## Usage

```bash
# Local HF model via in-process vLLM — MediQ, 3-case smoke test
python -m PtitWrap.cli --model vllm \
    --model_args pretrained=EPFLiGHT/Apertus-8B-MeditronFO,max_model_len=4096 \
    --task mediq --task_args limit=3,max_questions=5

# OpenAI-compatible API — AgentClinic, 1 scenario
python -m PtitWrap.cli --model openai-chat \
    --model_args model=gpt-4o,api_key_env=OPENAI_API_KEY \
    --task agentclinic --task_args num_scenarios=1,total_inferences=20

# A separately-started local vLLM server (OpenAI-compatible endpoint)
python -m PtitWrap.cli --model openai-chat \
    --model_args model=EPFLiGHT/Apertus-8B-MeditronFO,base_url=http://localhost:8000/v1 \
    --task mediq --task_args limit=3 --output results/mediq_run.json
```

`--model_args` / `--task_args` are `key=value,key=value` strings (types coerced
automatically), exactly like lm-eval. Metrics print to stdout; `--output`
writes the full per-sample JSON.

## Layout

```
PtitWrap/
  models/      LM base + registry (base.py), openai_chat.py, vllm_local.py
  adapters/    MultiTurnTask base + registry (base.py), MediQ_adapter.py, AgentClinic_adapter.py
  schema.py    Message type, EvalResult
  utils.py     simple_parse_args_string
  cli.py       entry point (python -m PtitWrap.cli)
  external/    vendored upstream benchmarks (mediQ, AgentClinic) — kept pristine
```

## Dependencies

The harness core needs nothing heavy. Each path pulls its own:
- `openai-chat` backend → `pip install openai`
- `vllm` backend → `pip install vllm transformers`
- `mediq` task → the MediQ src deps (`torch`, `transformers`, …)
- `agentclinic` task → `transformers`; `anthropic`/`replicate` are optional
  (auto-stubbed if absent, since we never call those provider branches).

## Status / limitations (v1)

- One model plays every role (doctor/patient/judge). Role separation — e.g. a
  separate judge to avoid the self-grading confound — is a clean future
  extension (give the task a role→model map instead of a single `model`).
- AgentClinic image requests aren't forwarded to the model (upstream only
  wires images for OpenAI vision models); text-only for now.
