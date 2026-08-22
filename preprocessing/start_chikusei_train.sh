#!/bin/bash
set -euo pipefail

SESSION_PREFIX="${1:-chikusei}"
WORKDIR="/root/autodl-tmp/workspace/AMSF-Net-main/preprocessing"
PROJECT_ROOT="/root/autodl-tmp/workspace/AMSF-Net-main"
PYTHON_BIN="/root/miniconda3/bin/python"
CONFIG_PATH="${WORKDIR}/config.yaml"
DATA_ROOT="/root/autodl-tmp/datasets/lrtn/Chikusei"
SRF_PATH="${WORKDIR}/resources/chikusei/chikusei_worldview2_srf_8x128.mat"

# 可通过环境变量调整空闲 GPU 判定阈值，或直接指定 GPU：
# FORCE_GPU=1 bash start_chikusei_train.sh chikusei
MAX_MEMORY_MB="${MAX_MEMORY_MB:-1000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-10}"
FORCE_GPU="${FORCE_GPU:-}"

for required_path in \
    "${WORKDIR}" \
    "${PROJECT_ROOT}" \
    "${DATA_ROOT}/Train" \
    "${DATA_ROOT}/Test"; do
    if [[ ! -d "${required_path}" ]]; then
        echo "错误：目录不存在：${required_path}"
        exit 1
    fi
done

for required_file in "${CONFIG_PATH}" "${SRF_PATH}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "错误：文件不存在：${required_file}"
        exit 1
    fi
done

if ! find "${DATA_ROOT}/Train" -maxdepth 1 -type f \
        \( -iname '*.h5' -o -iname '*.hdf5' \) -print -quit | grep -q .; then
    echo "错误：${DATA_ROOT}/Train 中未找到 H5 文件。"
    exit 1
fi

if ! find "${DATA_ROOT}/Test" -maxdepth 1 -type f \
        \( -iname '*.h5' -o -iname '*.hdf5' \) -print -quit | grep -q .; then
    echo "错误：${DATA_ROOT}/Test 中未找到 H5 文件。"
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "错误：Python 解释器不可用：${PYTHON_BIN}"
    exit 1
fi

for command_name in tmux nvidia-smi; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "错误：未找到 ${command_name}。"
        exit 1
    fi
done

readarray -t CONFIG_VALUES < <(
    "${PYTHON_BIN}" -c \
        "import yaml; c=yaml.safe_load(open('${CONFIG_PATH}', encoding='utf-8'))['Chikusei']['train']; print(c['experiment_name']); print(c['end_epoch']); print(c['n_bands']); print(c['n_select_bands'])"
)
EXPERIMENT_NAME="${CONFIG_VALUES[0]}"
END_EPOCH="${CONFIG_VALUES[1]}"
N_BANDS="${CONFIG_VALUES[2]}"
N_SELECT_BANDS="${CONFIG_VALUES[3]}"

echo "当前 GPU 状态："
nvidia-smi \
    --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits

if [[ -n "${FORCE_GPU}" ]]; then
    if ! nvidia-smi -i "${FORCE_GPU}" >/dev/null 2>&1; then
        echo "错误：GPU ${FORCE_GPU} 不存在。"
        exit 1
    fi
    GPU_INDEX="${FORCE_GPU}"
    GPU_MEMORY=$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -cd '0-9')
    GPU_UTIL=$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -cd '0-9')
else
    GPU_INDEX=""
    GPU_MEMORY=""
    GPU_UTIL=""

    while IFS=',' read -r current_gpu current_memory current_util; do
        current_gpu=$(printf '%s' "${current_gpu}" | tr -cd '0-9')
        current_memory=$(printf '%s' "${current_memory}" | tr -cd '0-9')
        current_util=$(printf '%s' "${current_util}" | tr -cd '0-9')

        if [[ -z "${current_gpu}" || -z "${current_memory}" || -z "${current_util}" ]]; then
            continue
        fi

        if (( current_memory <= MAX_MEMORY_MB && current_util <= MAX_GPU_UTIL )); then
            if [[ -z "${GPU_INDEX}" ]] ||
               (( current_memory < GPU_MEMORY )) ||
               (( current_memory == GPU_MEMORY && current_util < GPU_UTIL )); then
                GPU_INDEX="${current_gpu}"
                GPU_MEMORY="${current_memory}"
                GPU_UTIL="${current_util}"
            fi
        fi
    done < <(
        nvidia-smi \
            --query-gpu=index,memory.used,utilization.gpu \
            --format=csv,noheader,nounits
    )

    if [[ -z "${GPU_INDEX}" ]]; then
        echo "没有找到符合条件的空闲 GPU。"
        echo "要求：显存 <= ${MAX_MEMORY_MB} MiB，利用率 <= ${MAX_GPU_UTIL}%"
        echo "可手动指定，例如：FORCE_GPU=1 bash start_chikusei_train.sh chikusei"
        exit 1
    fi
fi

SESSION_PREFIX_SAFE=$(printf '%s' "${SESSION_PREFIX}" | sed 's/[^[:alnum:]_.-]/_/g')
EXPERIMENT_NAME_SAFE=$(printf '%s' "${EXPERIMENT_NAME}" | sed 's/[^[:alnum:]_.-]/_/g')
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
SESSION_NAME="${SESSION_PREFIX_SAFE}_${EXPERIMENT_NAME_SAFE}_gpu${GPU_INDEX}_${RUN_ID}"

while tmux has-session -t "${SESSION_NAME}" 2>/dev/null; do
    RUN_ID="$(date +%Y%m%d_%H%M%S)_$$_${RANDOM}"
    SESSION_NAME="${SESSION_PREFIX_SAFE}_${EXPERIMENT_NAME_SAFE}_gpu${GPU_INDEX}_${RUN_ID}"
done

echo
echo "数据集：Chikusei"
echo "实验名称：${EXPERIMENT_NAME}"
echo "训练轮数：${END_EPOCH}"
echo "通道配置：${N_SELECT_BANDS} MSI / ${N_BANDS} HSI"
echo "SRF：${SRF_PATH}"
echo "选择 GPU：${GPU_INDEX}（显存 ${GPU_MEMORY} MiB，利用率 ${GPU_UTIL}%）"
echo "tmux 会话：${SESSION_NAME}"
echo

tmux new-session -d -s "${SESSION_NAME}"
tmux send-keys -t "${SESSION_NAME}" \
    "cd '${WORKDIR}' && \
     export PYTHONPATH='${PROJECT_ROOT}':\${PYTHONPATH:-} && \
     export CUDA_VISIBLE_DEVICES='${GPU_INDEX}' && \
     echo 'CUDA_VISIBLE_DEVICES=${GPU_INDEX}' && \
     echo 'DATASET=Chikusei' && \
     echo 'EXPERIMENT_NAME=${EXPERIMENT_NAME}' && \
     exec '${PYTHON_BIN}' -u preprocessing_train_cave.py --config '${CONFIG_PATH}' --dataset Chikusei" \
    C-m

sleep 2
if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "训练会话启动失败，请检查 Python 报错。"
    exit 1
fi

echo "训练已启动。"
echo "查看会话：tmux ls"
echo "重新进入：tmux attach -t '${SESSION_NAME}'"
echo "后台查看：tmux capture-pane -pt '${SESSION_NAME}' | tail -n 30"
echo

tmux attach-session -t "${SESSION_NAME}"
