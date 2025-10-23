#!/bin/bash
#SBATCH --job-name=video2world_10n
#SBATCH -A CGAI24022
#SBATCH -N 10                 # 10 个节点
#SBATCH -p gh
#SBATCH -t 6:00:00
#SBATCH --ntasks-per-node=1   # 每节点 1 个 task
#SBATCH -o /scratch/10102/hh29499/carcrashtwin/%x_%j.out
#SBATCH -e /scratch/10102/hh29499/carcrashtwin/%x_%j.err


set -euo pipefail

# ========== 必填/可配环境变量 ==========
VIDEOS_MANIFEST="${VIDEOS_MANIFEST:-tasks_videos_manifest.txt}"     # 10 行，每行一个“视频列表txt”的路径
PROMPTS_MANIFEST="${PROMPTS_MANIFEST:-tasks_prompts_manifest.txt}"  # 10 行，每行一个“prompt列表txt”的路径
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"                               # 输出根目录

MODEL_SIZE="${MODEL_SIZE:-2B}"             # --model_size
COND_FRAMES="${COND_FRAMES:-5}"            # --num_conditional_frames
EXTRA_ARGS="${EXTRA_ARGS:-}"               # 追加给 python -m examples.video2world 的其他参数（可空）

# ========== 基本检查 ==========
mkdir -p slurm_logs "$OUTPUT_ROOT"


# 读 10 行
mapfile -t VIDEO_LISTS < <(sed -e 's/^[[:space:]]*//; s/[[:space:]]*$//' "$VIDEOS_MANIFEST" | grep -v '^[[:space:]]*$')
mapfile -t PROMPT_LISTS < <(sed -e 's/^[[:space:]]*//; s/[[:space:]]*$//' "$PROMPTS_MANIFEST" | grep -v '^[[:space:]]*$')

if [[ "${#VIDEO_LISTS[@]}" -ne 10 ]]; then
  echo "[ERROR] $VIDEOS_MANIFEST must have exactly 10 non-empty lines." >&2; exit 1
fi
if [[ "${#PROMPT_LISTS[@]}" -ne 10 ]]; then
  echo "[ERROR] $PROMPTS_MANIFEST must have exactly 10 non-empty lines." >&2; exit 1
fi

echo "[INFO] Using ${#VIDEO_LISTS[@]} video lists and ${#PROMPT_LISTS[@]} prompt lists."
echo "[INFO] MODEL_SIZE=$MODEL_SIZE  COND_FRAMES=$COND_FRAMES"
echo "[INFO] OUTPUT_ROOT=$OUTPUT_ROOT"

# ========== 并发启动 10 个 task（每 task 绑定 1 节点 / 1GPU）==========
pids=()
for IDX in $(seq 0 9); do
  VLIST="${VIDEO_LISTS[$IDX]}"
  PLIST="${PROMPT_LISTS[$IDX]}"

  if [[ ! -f "$VLIST" ]]; then
    echo "[ERROR] Video list for task $IDX not found: $VLIST" >&2; exit 1
  fi
  if [[ ! -f "$PLIST" ]]; then
    echo "[ERROR] Prompt list for task $IDX not found: $PLIST" >&2; exit 1
  fi

  echo "[LAUNCH] task=$IDX  videos=$VLIST  prompts=$PLIST"

  srun --nodes=1 --ntasks=1 --gres=gpu:1 --exclusive \
    bash ./run_10uq.sh \
      --videos-list "$VLIST" \
      --prompts-list "$PLIST" \
      --out "$OUTPUT_ROOT" \
      --model "$MODEL_SIZE" \
      --cond "$COND_FRAMES" \
      --extra "$EXTRA_ARGS" \
      --task-idx "$IDX" &

  pids+=( $! )
done

# 等所有子任务结束
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "[DONE] All 10 tasks finished."
