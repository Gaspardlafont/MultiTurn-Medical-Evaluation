#!/usr/bin/env bash
# Automates: start a local vLLM OpenAI-compatible server, then run AgentClinic
# against it via the VLLM_ branch added to agentclinic.py.
#
# Run from inside the RCP pod, with a venv that already has vllm installed
# (e.g. the mediq_venv created for the MediQ run) reachable at $VENV.
#
# Usage (defaults shown):
#   export GASPAR=zbourlar
#   ./run_agentclinic_local_vllm.sh
#
# Override any default via environment variables, e.g.:
#   MODEL=Qwen/Qwen2.5-7B-Instruct PORT=8001 NUM_SCENARIOS=3 ./run_agentclinic_local_vllm.sh
set -euo pipefail

: "${GASPAR:?Please export GASPAR=<your-gaspar-username> first}"

VENV="${VENV:-/lightscratch/users/$GASPAR/mediq_venv}"
MODEL="${MODEL:-EPFLiGHT/Apertus-8B-MeditronFO}"
PORT="${PORT:-8000}"
DATASET="${DATASET:-MedQA}"
NUM_SCENARIOS="${NUM_SCENARIOS:-1}"
TOTAL_INFERENCES="${TOTAL_INFERENCES:-5}"
AGENTCLINIC_DIR="${AGENTCLINIC_DIR:-/lightscratch/users/$GASPAR/MultiTurn-Medical-Evaluation/PtitWrap/external/AgentClinic}"
VLLM_LOG="${VLLM_LOG:-/tmp/vllm_server_${PORT}.log}"

echo "[1/4] Activating venv: $VENV"
source "$VENV/bin/activate"

echo "[2/4] Ensuring AgentClinic's Python deps are present (openai, anthropic, replicate)"
pip install -q openai anthropic replicate

echo "[3/4] Starting vLLM server for $MODEL on port $PORT (log: $VLLM_LOG)"
python -m vllm.entrypoints.openai.api_server --model "$MODEL" --port "$PORT" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
    echo "Stopping vLLM server (pid $VLLM_PID)"
    kill "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to become ready..."
ready=false
for _ in $(seq 1 60); do
    if curl -sf "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
        ready=true
        break
    fi
    sleep 10
done
if [ "$ready" != "true" ]; then
    echo "vLLM server did not become ready in time (10min). Check $VLLM_LOG"
    exit 1
fi
echo "vLLM server ready."

echo "[4/4] Running AgentClinic against VLLM_${MODEL}"
cd "$AGENTCLINIC_DIR"
python agentclinic.py \
    --doctor_llm "VLLM_${MODEL}" \
    --patient_llm "VLLM_${MODEL}" \
    --measurement_llm "VLLM_${MODEL}" \
    --moderator_llm "VLLM_${MODEL}" \
    --agent_dataset "$DATASET" \
    --num_scenarios "$NUM_SCENARIOS" \
    --total_inferences "$TOTAL_INFERENCES"
