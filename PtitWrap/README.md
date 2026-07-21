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
  no server to manage. "Bring your HF model." Best for a single model.
- **`vllm-server`** — the harness **launches a vLLM server subprocess** for the
  checkpoint, waits until it's ready, chats over HTTP, and kills it on exit.
  Same "harness launches it" UX as `vllm`, but lets two models coexist (see
  role separation below).

## Task adapters (`adapters/`)

Each task runs the **upstream benchmark unmodified**. Both benchmarks funnel
every model call through a single function, so the adapter just replaces that
one function at runtime with a shim backed by our `LM` (monkeypatch, zero edits
to upstream):

- **`mediq`** → patches `helper.get_response`, drives MediQ's own
  `run_patient_interaction` loop.
- **`agentclinic`** → patches `query_model`, calls AgentClinic's own `main()`;
  wraps `compare_results` to capture accuracy.
- **`meddxagent`** → no monkeypatch needed: MEDDxAgent already picks its LLM
  backend by config string (`init_model`), so the adapter registers one extra
  backend wrapping our `LM` and points every agent at it, then runs MEDDxAgent's
  own `DDxDriver` loop per patient. Doctor (driver + history-taking + diagnosis)
  = `--model`; simulated patient = `--judge_model`. Scores with MEDDxAgent's own
  `metrics.py` (GTPA@k), not an LLM judge. V1: iCraftMD, multi-turn, RAG/few-shot
  off — matches the Inspect wrapper for cross-harness comparison.

Adding a benchmark = one new adapter that drives its loop with our `LM`. No new
model code, nothing changed upstream.

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

### Role separation (avoiding the self-grading confound)

By default one model plays every role. Pass `--judge_model` to run the
patient/measurement/judge roles on a *different* model than the doctor under
test — otherwise the model judges its own diagnosis (and can score a wrong or
even empty answer as correct). The doctor uses `--model`; everyone else uses
`--judge_model`.

```bash
# Both local, harness-launched: the vllm-server backend starts a vLLM server
# per model itself (no manual server management), sharing one GPU via memory
# caps. Doctor = Meditron on port 8000, judge/patient = Qwen on port 8001.
python -m PtitWrap.cli \
    --model vllm-server --model_args pretrained=EPFLiGHT/Apertus-8B-MeditronFO,port=8000 \
    --judge_model vllm-server --judge_model_args pretrained=Qwen/Qwen2.5-7B-Instruct,port=8001 \
    --task agentclinic --task_args num_scenarios=5,total_inferences=20

# Doctor = local Meditron (in-process vLLM); judge/patient = a separate API
# model. No extra GPU RAM at all: the judge runs off-box via an API.
python -m PtitWrap.cli \
    --model vllm --model_args pretrained=EPFLiGHT/Apertus-8B-MeditronFO,max_model_len=4096 \
    --judge_model openai-chat --judge_model_args model=gpt-4o-mini,api_key_env=OPENAI_API_KEY \
    --task agentclinic --task_args num_scenarios=5,total_inferences=20
```

For two GPUs, pin each server with `cuda_visible_devices` in its args, e.g.
`--model_args pretrained=...,port=8000,cuda_visible_devices=0` and
`--judge_model_args pretrained=...,port=8001,cuda_visible_devices=1`
(and raise `gpu_memory_utilization` since they no longer share a GPU).

**Why `vllm-server` and not two in-process `vllm`?** Two in-process vLLM
engines in one process both grab GPU 0. `vllm-server` launches each model as its
own server subprocess (harness-managed lifecycle: started, health-checked, and
killed on exit), so two models coexist — on one GPU via memory caps, or on two
via `cuda_visible_devices`.

`--model_args` / `--task_args` are `key=value,key=value` strings (types coerced
automatically), exactly like lm-eval. Metrics print to stdout; `--output`
writes the full per-sample JSON.

### Config files (`--config`)

Any run above can be written once as a YAML file instead of retyped on the
command line every time — useful for anything you'll run more than once, or
want to hand to someone else without them re-assembling the flags. See
[`PtitWrap/configs/`](configs/) for ready-to-use examples:

```bash
python -m PtitWrap.cli --config PtitWrap/configs/agentclinic_meditron_vs_qwen.yaml
```

A config file mirrors the CLI flags — `model`, `model_args` (a mapping, not a
string), `judge_model`, `judge_model_args`, `task`, `task_args`, `output`,
`inspect_log`:

```yaml
model: vllm-server
model_args:
  pretrained: EPFLiGHT/Apertus-8B-MeditronFO
  port: 8000
judge_model: vllm-server
judge_model_args:
  pretrained: Qwen/Qwen2.5-7B-Instruct
  port: 8001
task: agentclinic
task_args:
  num_scenarios: 10
  total_inferences: 20
output: PtitWrap/results/run.json
inspect_log: PtitWrap/results/inspect/run.eval
```

**Any CLI flag overrides the matching config key** (config sets the defaults
for an experiment, a flag tweaks it ad hoc for one run) — e.g. this reuses the
config above but scales up to 50 scenarios without editing the file:
```bash
python -m PtitWrap.cli --config PtitWrap/configs/agentclinic_meditron_vs_qwen.yaml \
    --task_args num_scenarios=50
```
`pip install pyyaml` (imported lazily, only needed when `--config` is used).

## Layout

```
PtitWrap/
  models/      LM base + registry (base.py), openai_chat.py, vllm_local.py, vllm_server.py
  adapters/    MultiTurnTask base + registry (base.py), MediQ_adapter.py, AgentClinic_adapter.py, MEDDxAgent_adapter.py
  writers/     output formats beyond plain JSON (inspect_log.py)
  configs/     example --config YAML files
  schema.py    Message type, EvalResult
  utils.py     simple_parse_args_string
  config.py    --config YAML loading
  cli.py       entry point (python -m PtitWrap.cli)
  external/    vendored upstream benchmarks (mediQ, AgentClinic, meddxagent) — kept pristine
```

## Dependencies

The harness core needs nothing heavy. Each path pulls its own:
- `openai-chat` backend → `pip install openai`
- `vllm` backend → `pip install vllm transformers`
- `mediq` task → the MediQ src deps (`torch`, `transformers`, …)
- `agentclinic` task → `transformers`; `anthropic`/`replicate` are optional
  (auto-stubbed if absent, since we never call those provider branches).
- `meddxagent` task → the MEDDxAgent stack: `pip install -e PtitWrap/external/meddxagent`
  (pulls `datasets`, `faiss-cpu`, `transformers`, `colorama`, … — needed at
  import time even though RAG/few-shot are off).
- `--config` → `pip install pyyaml`
- `--inspect_log` → `pip install inspect_ai`

## Status / limitations

- Role separation is supported via `--judge_model` (see above). Without it, one
  model plays every role — fine for a quick smoke test, but the accuracy is not
  trustworthy (the model judges its own diagnosis).
- **Role bleeding (partly mitigated):** with no stop-sequence in AgentClinic's
  plain-text loop, a rambling model can generate *both* the doctor's question
  and the patient's reply in one turn — which also self-numbers questions wrong
  and balloons the history until it overflows the context window. Lowering the
  default `max_tokens` (now 256) shortens turns and helps; a proper fix would
  add stop-sequences and/or truncate the running history. Set `max_tokens` /
  `max_model_len` via `--model_args` if you still hit context-length errors.
- AgentClinic image requests aren't forwarded to the model (upstream only
  wires images for OpenAI vision models); text-only for now.
