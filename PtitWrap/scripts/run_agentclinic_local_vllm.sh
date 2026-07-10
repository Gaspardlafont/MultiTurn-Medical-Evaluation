#!/usr/bin/env bash
# Automates: start two local vLLM OpenAI-compatible servers (doctor model +
# a separate judge model), then run AgentClinic against them via the VLLM_
# branch added to agentclinic.py.
#
# Doctor model = the one under test (default: Meditron). Judge model powers
# patient/measurement/moderator, kept separate from the doctor model to avoid
# the self-grading confound (a model judging its own diagnosis as correct).
#
# Run from inside the RCP pod, with a venv that already has vllm installed
# (e.g. the mediq_venv created for the MediQ run) reachable at $VENV.
#
# Usage (defaults shown):
#   export GASPAR=zbourlar
#   ./run_agentclinic_local_vllm.sh
#
# Override any default via environment variables, e.g.:
#   MODEL=EPFLiGHT/Apertus-8B-MeditronFO JUDGE_MODEL=Qwen/Qwen2.5-7B-Instruct \
#   NUM_SCENARIOS=10 TOTAL_INFERENCES=20 ./run_agentclinic_local_vllm.sh
set -euo pipefail

: "${GASPAR:?Please export GASPAR=<your-gaspar-username> first}"

VENV="${VENV:-/lightscratch/users/$GASPAR/mediq_venv}"
MODEL="${MODEL:-EPFLiGHT/Apertus-8B-MeditronFO}"
PORT="${PORT:-8000}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
JUDGE_PORT="${JUDGE_PORT:-8001}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.45}"
DATASET="${DATASET:-MedQA}"
NUM_SCENARIOS="${NUM_SCENARIOS:-1}"
TOTAL_INFERENCES="${TOTAL_INFERENCES:-5}"
AGENTCLINIC_DIR="${AGENTCLINIC_DIR:-/lightscratch/users/$GASPAR/MultiTurn-Medical-Evaluation/PtitWrap/external/AgentClinic}"
DOCTOR_LOG="${DOCTOR_LOG:-/tmp/vllm_server_${PORT}.log}"
JUDGE_LOG="${JUDGE_LOG:-/tmp/vllm_server_${JUDGE_PORT}.log}"

echo "[1/5] Activating venv: $VENV"
source "$VENV/bin/activate"

echo "[2/5] Ensuring AgentClinic's Python deps are present (openai, anthropic, replicate)"
pip install -q openai anthropic replicate

# Two vLLM servers sharing one GPU each default to ~90% memory reservation,
# so the second one to start fails unless both are capped explicitly.
echo "[3/5] Starting vLLM server for doctor model $MODEL on port $PORT (log: $DOCTOR_LOG)"
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    > "$DOCTOR_LOG" 2>&1 &
DOCTOR_PID=$!

echo "Starting vLLM server for judge model $JUDGE_MODEL on port $JUDGE_PORT (log: $JUDGE_LOG)"
python -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" --port "$JUDGE_PORT" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    > "$JUDGE_LOG" 2>&1 &
JUDGE_PID=$!

cleanup() {
    echo "Stopping vLLM servers (pids $DOCTOR_PID, $JUDGE_PID)"
    kill "$DOCTOR_PID" "$JUDGE_PID" 2>/dev/null || true
}
trap cleanup EXIT

wait_for_server() {
    local port="$1"
    local logfile="$2"
    for _ in $(seq 1 60); do
        if curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            return 0
        fi
        sleep 10
    done
    echo "vLLM server on port $port did not become ready in time (10min). Check $logfile"
    return 1
}

echo "[4/5] Waiting for both vLLM servers to become ready..."
wait_for_server "$PORT" "$DOCTOR_LOG"
echo "Doctor server ready."
wait_for_server "$JUDGE_PORT" "$JUDGE_LOG"
echo "Judge server ready."

echo "[5/5] Running AgentClinic: doctor=VLLM_${PORT}:${MODEL}, patient/measurement/moderator=VLLM_${JUDGE_PORT}:${JUDGE_MODEL}"
cd "$AGENTCLINIC_DIR"
python agentclinic.py \
    --doctor_llm "VLLM_${PORT}:${MODEL}" \
    --patient_llm "VLLM_${JUDGE_PORT}:${JUDGE_MODEL}" \
    --measurement_llm "VLLM_${JUDGE_PORT}:${JUDGE_MODEL}" \
    --moderator_llm "VLLM_${JUDGE_PORT}:${JUDGE_MODEL}" \
    --agent_dataset "$DATASET" \
    --num_scenarios "$NUM_SCENARIOS" \
    --total_inferences "$TOTAL_INFERENCES"
