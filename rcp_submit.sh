#!/usr/bin/env bash
# Delete then resubmit a job on RCP (light-$GASPAR project, h100 node-pool).
# Usage: bash rcp_submit.sh
set -euo pipefail

JOB_NAME=agentclinic-job

runai delete job "$JOB_NAME" 2>/dev/null || true

runai submit \
  --name "$JOB_NAME" \
  --image registry.rcp.epfl.ch/multiturn-eval-harness/$GASPAR/basic:amd64-cuda-$GASPAR-latest \
  --pvc light-scratch:/lightscratch \
  --large-shm \
  -e NAS_HOME=/lightscratch/users/$GASPAR \
  -e HF_API_KEY_FILE_AT=/lightscratch/users/$GASPAR/keys/hf_key.txt \
  -e WANDB_API_KEY_FILE_AT=/lightscratch/users/$GASPAR/keys/wandb_key.txt \
  -e GITCONFIG_AT=/lightscratch/users/$GASPAR/.gitconfig \
  -e GIT_CREDENTIALS_AT=/lightscratch/users/$GASPAR/.git-credentials \
  -e VSCODE_CONFIG_AT=/lightscratch/users/$GASPAR/.vscode-server \
  --backoff-limit 0 \
  --run-as-gid 84257 \
  --node-pool h100 \
  --gpu 2 \
  -- sleep infinity

echo
echo "Submitted. Check status with: runai describe job $JOB_NAME"
echo "Attach with:                  runai bash $JOB_NAME"
