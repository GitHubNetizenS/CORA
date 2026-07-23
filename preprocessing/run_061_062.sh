#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_CONFIG="${SCRIPT_DIR}/config.yaml"
QUEUE_FILE="${SCRIPT_DIR}/experiment_queues/queue_061_062.yaml"
GENERATED_DIR="${SCRIPT_DIR}/experiment_queues/generated_configs_061_062"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
WAIT_PID="${WAIT_PID:-}"
GPU_INDEX="${GPU_INDEX:-}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --wait-pid)
      if [[ $# -lt 2 ]]; then
        echo "--wait-pid需要一个进程PID。" >&2
        exit 1
      fi
      WAIT_PID="$2"
      shift
      ;;
    --gpu)
      if [[ $# -lt 2 ]]; then
        echo "--gpu需要一个GPU编号。" >&2
        exit 1
      fi
      GPU_INDEX="$2"
      shift
      ;;
    *)
      echo "未知参数: $1" >&2
      echo "用法: bash run_061_062.sh [--dry-run] [--wait-pid PID] [--gpu INDEX]" >&2
      exit 1
      ;;
  esac
  shift
done

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "缺少基础配置: ${BASE_CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${QUEUE_FILE}" ]]; then
  echo "缺少实验队列: ${QUEUE_FILE}" >&2
  exit 1
fi

mapfile -t EXPERIMENT_LINES < <(
  "${PYTHON_BIN}" - "${BASE_CONFIG}" "${QUEUE_FILE}" "${GENERATED_DIR}" "${DRY_RUN}" <<'PY'
import copy
import sys
from pathlib import Path

import yaml

base_config = Path(sys.argv[1])
queue_file = Path(sys.argv[2])
generated_dir = Path(sys.argv[3])
dry_run = sys.argv[4] == "1"

with base_config.open("r", encoding="utf-8") as f:
    base = yaml.safe_load(f)

with queue_file.open("r", encoding="utf-8") as f:
    queue = yaml.safe_load(f)

experiments = queue.get("experiments", [])
if dry_run:
    for experiment in experiments:
        print(experiment["name"])
    raise SystemExit(0)

generated_dir.mkdir(parents=True, exist_ok=True)
for experiment in experiments:
    config = copy.deepcopy(base)
    train_config = config.setdefault("CAVE", {}).setdefault("train", {})
    train_config.update(experiment.get("train", {}))
    experiment_name = experiment["name"]
    train_config["experiment_name"] = experiment_name
    config_path = generated_dir / f"{experiment_name}.yaml"
    with config_path.open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    run_root = train_config.get("run_root", "/root/autodl-tmp/runs/AMSF-Net")
    print(f"{experiment_name}\t{config_path}\t{run_root}")
PY
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%s\n' "${EXPERIMENT_LINES[@]}"
  exit 0
fi

if [[ -n "${WAIT_PID}" ]]; then
  if [[ ! "${WAIT_PID}" =~ ^[0-9]+$ ]]; then
    echo "无效的PID: ${WAIT_PID}" >&2
    exit 1
  fi
  echo "实验配置已生成，等待当前训练进程PID=${WAIT_PID}结束。"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 60
  done
  echo "进程${WAIT_PID}已结束，开始执行实验061。"
fi

LOG_ROOT="/root/autodl-tmp/runs/AMSF-Net/batch_061_062_logs"
mkdir -p "${LOG_ROOT}"

for line in "${EXPERIMENT_LINES[@]}"; do
  IFS=$'\t' read -r EXPERIMENT_NAME CONFIG_PATH RUN_ROOT <<< "${line}"
  RECORD_CSV="${RUN_ROOT}/${EXPERIMENT_NAME}/records/${EXPERIMENT_NAME}_cave_record.csv"

  if "${PYTHON_BIN}" - "${RECORD_CSV}" <<'PY'
import csv
import sys
from pathlib import Path

record_path = Path(sys.argv[1])
if not record_path.exists():
    raise SystemExit(1)

max_epoch = 0
with record_path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        if not row:
            continue
        try:
            max_epoch = max(max_epoch, int(float(row[0])))
        except ValueError:
            continue

raise SystemExit(0 if max_epoch >= 1000 else 1)
PY
  then
    echo "[跳过] ${EXPERIMENT_NAME}已经完成1000轮。"
    continue
  fi

  echo "[启动] ${EXPERIMENT_NAME}"
  if [[ -n "${GPU_INDEX}" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_INDEX}" \
      "${PYTHON_BIN}" "${SCRIPT_DIR}/preprocessing_train_cave.py" \
      --config "${CONFIG_PATH}" 2>&1 | tee "${LOG_ROOT}/${EXPERIMENT_NAME}.log"
  else
    "${PYTHON_BIN}" "${SCRIPT_DIR}/preprocessing_train_cave.py" \
      --config "${CONFIG_PATH}" 2>&1 | tee "${LOG_ROOT}/${EXPERIMENT_NAME}.log"
  fi
done

echo "实验061和062均已完成。"
