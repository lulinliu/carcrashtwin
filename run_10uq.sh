#!/usr/bin/env bash
# 用法：由 run_10nodes.slurm 通过 srun 直接调用
set -euo pipefail

# 读取 Slurm 环境
: "${SLURM_PROCID:?SLURM_PROCID not set}"
: "${VIDEOS_MANIFEST:?VIDEOS_MANIFEST not set}"
: "${PROMPTS_MANIFEST:?PROMPTS_MANIFEST not set}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT not set}"
: "${MODEL_SIZE:?MODEL_SIZE not set}"
: "${COND_FRAMES:?COND_FRAMES not set}"
: "${PROJECT_DIR:?PROJECT_DIR not set}"
: "${CONDA_ENV:=}"

TASK_IDX="${SLURM_PROCID}"

# 激活 conda（尽量稳妥）

cd "${PROJECT_DIR}"

# 取出本任务对应的 “视频列表 txt” 与 “prompt 列表 txt”
LINE_NO=$((TASK_IDX + 1))

if [[ ! -f "${VIDEOS_MANIFEST}" ]]; then
  echo "[ERROR] VIDEOS_MANIFEST not found: ${VIDEOS_MANIFEST}" >&2
  exit 1
fi
if [[ ! -f "${PROMPTS_MANIFEST}" ]]; then
  echo "[ERROR] PROMPTS_MANIFEST not found: ${PROMPTS_MANIFEST}" >&2
  exit 1
fi

VIDEOS_LIST_FILE="$(sed -n "${LINE_NO}p" "${VIDEOS_MANIFEST}" | tr -d '\r')"
PROMPTS_LIST_FILE="$(sed -n "${LINE_NO}p" "${PROMPTS_MANIFEST}" | tr -d '\r')"

if [[ -z "${VIDEOS_LIST_FILE}" || ! -f "${VIDEOS_LIST_FILE}" ]]; then
  echo "[ERROR] For task ${TASK_IDX}, invalid videos list file: '${VIDEOS_LIST_FILE}'" >&2
  exit 1
fi
if [[ -z "${PROMPTS_LIST_FILE}" || ! -f "${PROMPTS_LIST_FILE}" ]]; then
  echo "[ERROR] For task ${TASK_IDX}, invalid prompts list file: '${PROMPTS_LIST_FILE}'" >&2
  exit 1
fi

echo "[INFO][task ${TASK_IDX}] videos list : ${VIDEOS_LIST_FILE}"
echo "[INFO][task ${TASK_IDX}] prompts list: ${PROMPTS_LIST_FILE}"

# 保证一一对应：取两边行数的最小值
N_V=$(grep -cve '^\s*$' "${VIDEOS_LIST_FILE}" || true)
N_P=$(grep -cve '^\s*$' "${PROMPTS_LIST_FILE}" || true)
if [[ "${N_V}" -eq 0 || "${N_P}" -eq 0 ]]; then
  echo "[WARN][task ${TASK_IDX}] empty list detected (videos=${N_V}, prompts=${N_P}), nothing to do."
  exit 0
fi

N=$(( N_V < N_P ? N_V : N_P ))
echo "[INFO][task ${TASK_IDX}] total pairs to run: ${N}"

# 本任务输出目录
OUT_DIR="${OUTPUT_ROOT}/task_${TASK_IDX}"
mkdir -p "${OUT_DIR}"

# 限制线程，避免在单 GPU 上乱抢核
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# 逐行配对运行
for i in $(seq 1 "${N}"); do
  VIDEO_PATH="$(sed -n "${i}p" "${VIDEOS_LIST_FILE}"   | tr -d '\r')"
  PROMPT_TXT="$(sed -n "${i}p" "${PROMPTS_LIST_FILE}" | tr -d '\r')"

  if [[ -z "${VIDEO_PATH}" || ! -f "${VIDEO_PATH}" ]]; then
    echo "[WARN][task ${TASK_IDX}] skip idx=${i}: invalid video '${VIDEO_PATH}'"
    continue
  fi
  if [[ -z "${PROMPT_TXT}" ]]; then
    echo "[WARN][task ${TASK_IDX}] skip idx=${i}: empty prompt"
    continue
  fi

  STEM="$(basename "${VIDEO_PATH}")"
  STEM="${STEM%.*}"
  SAVE_PATH="${OUT_DIR}/${STEM}_v2w_${MODEL_SIZE}.mp4"

  echo "[INFO][task ${TASK_IDX}] (${i}/${N}) -> ${SAVE_PATH}"
  # 如需使用你提供的固定 PROMPT_，把 --prompt 改为 "${PROMPT_}" 即可
  python -m examples.video2world \
      --model_size "${MODEL_SIZE}" \
      --input_path "${VIDEO_PATH}" \
      --num_conditional_frames "${COND_FRAMES}" \
      --prompt "${PROMPT_TXT}" \
      --save_path "${SAVE_PATH}" \
      --disable_guardrail \
      --disable_prompt_refiner
done

echo "[INFO][task ${TASK_IDX}] done. Outputs at ${OUT_DIR}"
