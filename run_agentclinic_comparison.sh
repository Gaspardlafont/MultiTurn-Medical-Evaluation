#!/usr/bin/env bash
# Full AgentClinic comparison run: harnesspect (Inspect wrapper) vs PtitWrap
# (pristine upstream), on the SAME experiment, so we can check the two harnesses
# land on the same accuracy.
#
# Three doctor configurations, each with Qwen playing every other role
# (patient / measurement / moderator):
#   1. meditron_vs_qwen        doctor = EPFLiGHT/Apertus-8B-MeditronFO
#   2. apertus_instruct_vs_qwen doctor = swiss-ai/Apertus-8B-Instruct-2509
#   3. qwen_vs_qwen            doctor = Qwen/Qwen2.5-7B-Instruct (Qwen everywhere)
#
# Each config is run on BOTH harnesses => 6 runs total, sequentially (each spins
# up its own vLLM, so they can't share the GPU at the same time).
#
# Logs land in  <repo>/result/  (sibling of harnesspect/ and PtitWrap/):
#   result/harnesspect/<name>/        Inspect .eval logs (open with `inspect view`)
#   result/harnesspect/<name>.log     stdout (accuracy table)
#   result/ptitwrap/<name>.json       full PtitWrap results
#   result/ptitwrap/<name>.run.log    stdout (metrics)
#   result/ptitwrap/inspect/<name>.eval  PtitWrap's own Inspect .eval log
#
# Prereqs (once, in the pod venv): the repos' deps + `inspect` on PATH.
# Run (see the RCP how-to at the bottom of this file / the PR):
#   bash run_agentclinic_comparison.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

RESULT_DIR="$REPO_ROOT/result"

QWEN="Qwen/Qwen2.5-7B-Instruct"
MEDITRON="EPFLiGHT/Apertus-8B-MeditronFO"
APERTUS="swiss-ai/Apertus-8B-Instruct-2509"

DATASET="MedQA"
MAX_TURNS=20          # = AgentClinic's total_inferences (one doctor turn per round)
GMU=0.45              # gpu_memory_utilization per vLLM engine (doctor + judge share one GPU)

# --- harnesspect's wrapper resolves the AgentClinic repo as a sibling of the
# repo root (../agentclinic) via __file__, and loads both agentclinic.py and the
# dataset jsonl from there. Point that at the vendored copy so nothing extra
# needs cloning. -----------------------------------------------------------------
SIBLING_AC="$(dirname "$REPO_ROOT")/agentclinic"
if [ ! -e "$SIBLING_AC" ]; then
  ln -s "$REPO_ROOT/PtitWrap/external/AgentClinic" "$SIBLING_AC"
  echo "[setup] symlinked $SIBLING_AC -> PtitWrap/external/AgentClinic"
fi

# Full MedQA set (count the dataset rather than hardcoding).
NUM_SCENARIOS="$(wc -l < "$SIBLING_AC/agentclinic_medqa.jsonl" | tr -d ' ')"
echo "[setup] running the full MedQA set: $NUM_SCENARIOS scenarios, $MAX_TURNS turns each"

mkdir -p "$RESULT_DIR/harnesspect" "$RESULT_DIR/ptitwrap/inspect"

# enforce_eager skips torch.compile / CUDA-graph capture at load. The Apertus
# family (xIELU activation, Python fallback) can otherwise spend 10-15 min
# compiling and race the startup timeout. It only affects startup/speed, not the
# generated tokens, so adding it symmetrically keeps the comparison valid. Qwen
# doesn't need it.
doctor_needs_eager () { [ "$1" != "$QWEN" ]; }

run_harnesspect () {
  local name="$1" doctor="$2"
  local doctor_args="gpu_memory_utilization: $GMU"
  doctor_needs_eager "$doctor" && doctor_args="$doctor_args, enforce_eager: true"
  echo "=== [harnesspect] $name  (doctor=$doctor) ==="
  inspect eval harnesspect/wrapped_inspect/inspect_agentclinic_wrapped.py \
    --model "vllm/$QWEN" \
    --model-role doctor="{model: vllm/$doctor, model_args: {$doctor_args}}" \
    --model-role patient="{model: vllm/$QWEN, model_args: {gpu_memory_utilization: $GMU}}" \
    --model-role measurement="{model: vllm/$QWEN, model_args: {gpu_memory_utilization: $GMU}}" \
    --model-role moderator="{model: vllm/$QWEN, model_args: {gpu_memory_utilization: $GMU}}" \
    -T dataset="$DATASET" -T limit="$NUM_SCENARIOS" -T max_turns="$MAX_TURNS" \
    --log-dir "$RESULT_DIR/harnesspect/$name" \
    2>&1 | tee "$RESULT_DIR/harnesspect/$name.log"
}

run_ptitwrap () {
  local name="$1" doctor="$2"
  local model_args="pretrained=$doctor,port=8000,gpu_memory_utilization=$GMU"
  doctor_needs_eager "$doctor" && model_args="$model_args,enforce_eager=true"
  echo "=== [ptitwrap] $name  (doctor=$doctor) ==="
  python -m PtitWrap.cli \
    --model vllm-server \
    --model_args "$model_args" \
    --judge_model vllm-server \
    --judge_model_args "pretrained=$QWEN,port=8001,gpu_memory_utilization=$GMU" \
    --task agentclinic \
    --task_args "dataset=$DATASET,num_scenarios=$NUM_SCENARIOS,total_inferences=$MAX_TURNS" \
    --output "$RESULT_DIR/ptitwrap/$name.json" \
    --inspect_log "$RESULT_DIR/ptitwrap/inspect/$name.eval" \
    2>&1 | tee "$RESULT_DIR/ptitwrap/$name.run.log"
}

# name:doctor  (Qwen plays every other role in all three)
EXPERIMENTS=(
  "meditron_vs_qwen:$MEDITRON"
  "apertus_instruct_vs_qwen:$APERTUS"
  "qwen_vs_qwen:$QWEN"
)

for pair in "${EXPERIMENTS[@]}"; do
  name="${pair%%:*}"
  doctor="${pair#*:}"
  # Don't let one failed run abort the whole batch -- log and keep going.
  run_harnesspect "$name" "$doctor" || echo "[warn] harnesspect/$name failed, continuing"
  run_ptitwrap    "$name" "$doctor" || echo "[warn] ptitwrap/$name failed, continuing"
done

echo
echo "All runs done. Results under: $RESULT_DIR"
echo "  Inspect logs:  inspect view --log-dir $RESULT_DIR/harnesspect/<name>"
echo "  PtitWrap JSON: $RESULT_DIR/ptitwrap/<name>.json  (accuracy in the metrics block)"
