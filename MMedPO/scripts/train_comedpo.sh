#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
# Set NUM_PROCESSES=8 to match the paper's GPU count.
LAUNCH_FLAGS=(--num_processes "${NUM_PROCESSES:-1}"
    --num_machines 1 --mixed_precision bf16 --dynamo_backend no)
if [[ "${NUM_PROCESSES:-1}" -gt 1 ]]; then
    LAUNCH_FLAGS+=(--multi_gpu)
fi
accelerate launch "${LAUNCH_FLAGS[@]}" -m MMedPO.comedpo.train "$@"
