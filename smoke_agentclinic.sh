#!/usr/bin/env bash
# Fast smoke test for the AgentClinic harnesspect-vs-PtitWrap comparison.
# Goal: in a few minutes, find out WHERE the full run got stuck (it ran 3h and
# produced nothing), not to get real accuracy numbers.
#
# Tiny by design: 2 scenarios, 3 turns. Every step is wrapped in `timeout`, so a
# hang self-aborts after STEP_TIMEOUT instead of running forever, and the final
# summary tells you exactly which step hung / failed / passed.
#
# Steps go from safest to riskiest so the failure point is unambiguous:
#   1. harnesspect, Qwen for every role      -> plumbing + dataset + scorer OK?
#   2. PtitWrap,    Qwen vs Qwen             -> vllm-server backend OK?
#   3. harnesspect, Meditron (load-only)     -> does Inspect's vllm hang loading
#                                               Meditron? (main suspect)
#   4. PtitWrap,    Meditron vs Qwen (tiny)  -> vllm-server + enforce_eager path
#
# Run from the repo root:
#   bash smoke_agentclinic.sh
# Outputs + per-step logs land in result_smoke/ (gitignored).
set -uo pipefail   # NOT -e: one failing step must not abort the whole smoke

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

RESULT_DIR="$REPO_ROOT/result_smoke"
QWEN="Qwen/Qwen2.5-7B-Instruct"
MEDITRON="EPFLiGHT/Apertus-8B-MeditronFO"

DATASET="MedQA"
N=2                 # scenarios (tiny)
TURNS=3             # turns per scenario (tiny)
GMU=0.40            # gpu_memory_utilization per engine
STEP_TIMEOUT=1200   # 20 min hard cap per step; a hang trips this and moves on

log()  { echo -e "\n\033[1;36m[$(date +%H:%M:%S)] $*\033[0m"; }

declare -a SUMMARY

# run_step <name> <cmd...>   (cmd args must be plain tokens, no embedded spaces)
run_step () {
  local name="$1"; shift
  local logf="$RESULT_DIR/$name.log"
  log "▶ STEP $name"
  echo "    \$ $*" | tee "$logf"
  timeout "$STEP_TIMEOUT" "$@" 2>&1 | tee -a "$logf"
  local rc=${PIPESTATUS[0]}
  if   [ "$rc" -eq 0 ];   then SUMMARY+=("✅ OK    $name")
  elif [ "$rc" -eq 124 ]; then SUMMARY+=("⏳ HANG  $name  (>${STEP_TIMEOUT}s — stuck at model load / compile / download)")
  else                         SUMMARY+=("❌ FAIL  $name  (exit $rc — see tail of $logf)")
  fi
}

# --- setup -----------------------------------------------------------------
# harnesspect's wrapper resolves the AgentClinic repo as a sibling of the repo
# root (../agentclinic) and reads agentclinic.py + the dataset from there.
SIBLING_AC="$(dirname "$REPO_ROOT")/agentclinic"
if [ ! -e "$SIBLING_AC" ]; then
  ln -s "$REPO_ROOT/PtitWrap/external/AgentClinic" "$SIBLING_AC"
  log "symlinked $SIBLING_AC -> PtitWrap/external/AgentClinic"
fi
mkdir -p "$RESULT_DIR/harnesspect" "$RESULT_DIR/ptitwrap/inspect"

# --- STEP 0 : environment sanity (never hangs) -----------------------------
log "STEP 0 — environment"
echo "inspect: $(command -v inspect || echo MISSING)"
echo "python:  $(command -v python || command -v python3 || echo MISSING)"
python -c "import inspect_ai, vllm; print('imports OK: inspect_ai', inspect_ai.__version__)" 2>&1 | tail -3
echo "--- GPUs ---"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv 2>&1 | head

# --- STEP 1 : harnesspect, Qwen everywhere (simplest path) -----------------
run_step "1_harnesspect_qwen" \
  inspect eval harnesspect/wrapped_inspect/inspect_agentclinic_wrapped.py \
    --model "vllm/$QWEN" \
    -M gpu_memory_utilization="$GMU" \
    -T dataset="$DATASET" -T limit="$N" -T max_turns="$TURNS" \
    --log-dir "$RESULT_DIR/harnesspect/qwen"

# --- STEP 2 : PtitWrap, Qwen vs Qwen ---------------------------------------
run_step "2_ptitwrap_qwen" \
  python -m PtitWrap.cli \
    --model vllm-server \
    --model_args "pretrained=$QWEN,port=8000,gpu_memory_utilization=$GMU" \
    --judge_model vllm-server \
    --judge_model_args "pretrained=$QWEN,port=8001,gpu_memory_utilization=$GMU" \
    --task agentclinic \
    --task_args "dataset=$DATASET,num_scenarios=$N,total_inferences=$TURNS" \
    --output "$RESULT_DIR/ptitwrap/qwen.json" \
    --inspect_log "$RESULT_DIR/ptitwrap/inspect/qwen.eval"

# --- STEP 3 : harnesspect, Meditron LOAD test (main suspect) ---------------
# Single model = Meditron for every role, 1 scenario / 2 turns. If THIS hangs,
# the problem is Inspect's vllm provider loading/compiling Meditron (xIELU),
# and enforce_eager is not taking effect the way it does for PtitWrap.
run_step "3_harnesspect_meditron_load" \
  inspect eval harnesspect/wrapped_inspect/inspect_agentclinic_wrapped.py \
    --model "vllm/$MEDITRON" \
    -M gpu_memory_utilization=0.45 -M enforce_eager=true \
    -T dataset="$DATASET" -T limit=1 -T max_turns=2 \
    --log-dir "$RESULT_DIR/harnesspect/meditron_load"

# --- STEP 4 : PtitWrap, Meditron vs Qwen (tiny) ----------------------------
run_step "4_ptitwrap_meditron" \
  python -m PtitWrap.cli \
    --model vllm-server \
    --model_args "pretrained=$MEDITRON,port=8000,gpu_memory_utilization=0.45,enforce_eager=true" \
    --judge_model vllm-server \
    --judge_model_args "pretrained=$QWEN,port=8001,gpu_memory_utilization=$GMU" \
    --task agentclinic \
    --task_args "dataset=$DATASET,num_scenarios=$N,total_inferences=$TURNS" \
    --output "$RESULT_DIR/ptitwrap/meditron.json" \
    --inspect_log "$RESULT_DIR/ptitwrap/inspect/meditron.eval"

# --- summary ---------------------------------------------------------------
log "SMOKE SUMMARY"
printf '  %s\n' "${SUMMARY[@]}"
echo
echo "Per-step logs: $RESULT_DIR/*.log"
echo "If a step shows HANG, paste me the tail of its .log and 'nvidia-smi'."
