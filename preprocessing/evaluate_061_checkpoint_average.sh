#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/root/autodl-tmp/workspace/AMSF-Net-main/preprocessing}"
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/workspace/AMSF-Net-main}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
GPU="${GPU:-0}"
RUN_DIR="${RUN_DIR:-/root/autodl-tmp/runs/AMSF-Net/061_seed42_058_main_spectral_anchor}"

cd "${WORKDIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "RUN_DIR=${RUN_DIR}"

exec "${PYTHON_BIN}" evaluate_checkpoint_average.py \
    --run-dir "${RUN_DIR}" \
    --weight-a 0.5 \
    "$@"
