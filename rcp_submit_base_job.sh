#!/usr/bin/env bash
# Delete then resubmit base-job on RCP (light-lafont project, h100 node-pool).
# Usage: bash rcp_submit_base_job.sh
set -euo pipefail

runai delete job base-job 2>/dev/null || true

runai submit \
  --name base-job \
  --image registry.rcp.epfl.ch/multiturn-eval-harness/lafont/basic:amd64-cuda-lafont-latest \
  --pvc light-scratch:/lightscratch \
  --large-shm \
  -e NAS_HOME=/lightscratch/users/lafont \
  -e HF_API_KEY_FILE_AT=/lightscratch/users/lafont/keys/hf_key.txt \
  -e WANDB_API_KEY_FILE_AT=/lightscratch/users/lafont/keys/wandb_key.txt \
  -e GITCONFIG_AT=/lightscratch/users/lafont/.gitconfig \
  -e GIT_CREDENTIALS_AT=/lightscratch/users/lafont/.git-credentials \
  -e VSCODE_CONFIG_AT=/lightscratch/users/lafont/.vscode-server \
  --backoff-limit 0 \
  --run-as-gid 84257 \
  --node-pool h100 \
  --gpu 1 \
  -- sleep infinity

echo
echo "Submitted. Check status with: runai describe job base-job"
echo "Attach with:                  runai bash base-job"
