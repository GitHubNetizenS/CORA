#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_CONFIG="${SCRIPT_DIR}/config.yaml"
QUEUE_FILE="${SCRIPT_DIR}/experiment_queues/queue_031_036.yaml"
GENERATED_DIR="${SCRIPT_DIR}/experiment_queues/generated_configs"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Missing base config: ${BASE_CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${QUEUE_FILE}" ]]; then
  echo "Missing queue file: ${QUEUE_FILE}" >&2
  exit 1
fi

mapfile -t EXPERIMENT_LINES < <("${PYTHON_BIN}" - "${BASE_CONFIG}" "${QUEUE_FILE}" "${GENERATED_DIR}" "${DRY_RUN}" <<'PY'
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
    for exp in experiments:
        print(exp["name"])
    raise SystemExit(0)

generated_dir.mkdir(parents=True, exist_ok=True)
for exp in experiments:
    cfg = copy.deepcopy(base)
    train_cfg = cfg.setdefault("CAVE", {}).setdefault("train", {})
    train_cfg.update(exp.get("train", {}))
    name = exp["name"]
    train_cfg["experiment_name"] = name
    config_path = generated_dir / f"{name}.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    run_root = train_cfg.get("run_root", "/root/autodl-tmp/runs/AMSF-Net")
    print(f"{name}\t{config_path}\t{run_root}")
PY
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%s\n' "${EXPERIMENT_LINES[@]}"
  exit 0
fi

LOG_ROOT="/root/autodl-tmp/runs/AMSF-Net/batch_031_036_logs"
mkdir -p "${LOG_ROOT}"

for line in "${EXPERIMENT_LINES[@]}"; do
  IFS=$'\t' read -r EXP_NAME CONFIG_PATH RUN_ROOT <<< "${line}"
  RECORD_CSV="${RUN_ROOT}/${EXP_NAME}/records/${EXP_NAME}_cave_record.csv"

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
            epoch = int(float(row[0]))
        except ValueError:
            continue
        if epoch > max_epoch:
            max_epoch = epoch

raise SystemExit(0 if max_epoch >= 1000 else 1)
PY
  then
    echo "[SKIP] ${EXP_NAME} already has epoch 1000."
    continue
  fi

  echo "[RUN] ${EXP_NAME}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/preprocessing_train_cave.py" --config "${CONFIG_PATH}" 2>&1 | tee "${LOG_ROOT}/${EXP_NAME}.log"
done

echo "All queued experiments finished."
