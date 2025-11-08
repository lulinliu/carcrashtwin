#!/usr/bin/env bash
set -euo pipefail

#################### 路径配置 ####################
REPO_ROOT="/scratch/10102/hh29499/carcrashtwin"

# 输入
FIRST_DIR="${REPO_ROOT}/hot_3d_infer/videos"
PROMPT_DIR="${REPO_ROOT}/hot_3d_infer/captions"

# 输出
OUT_ROOT="${REPO_ROOT}/cosmos_hot3d_van14b"
mkdir -p "$OUT_ROOT"

# 如果你这个纯推理也要指定权重，就在这里写；如果 examples/video2world 自己会load默认model，可以不写
# DIT_PATH="/scratch/10102/hh29499/carcrashtwin/real_world_3k/checkpoints/model/iter_000003000.pt"

#################### 分布式相关（slurm 传进来） ####################
WORLD_SIZE="${SLURM_NTASKS:-${WORLD_SIZE:-1}}"
WORKER_RANK="${SLURM_PROCID:-${WORKER_RANK:-0}}"
GPU_ID="${SLURM_LOCALID:-${GPU_ID:-0}}"

#################### 分布式相关 ####################

cd "$REPO_ROOT"

# 收集所有 prompt
mapfile -t PROMPTS < <(ls "$PROMPT_DIR"/*.txt | sort)
TOTAL=${#PROMPTS[@]}
echo "[INFO][rank ${WORKER_RANK}/${WORLD_SIZE}] total samples: $TOTAL"

for ((i=0; i<TOTAL; i++)); do
  # 轮询分配
  if (( i % WORLD_SIZE != WORKER_RANK )); then
    continue
  fi

  prompt_file="${PROMPTS[$i]}"
  base="$(basename "$prompt_file" .txt)"

  # 找图
  img_file="${FIRST_DIR}/${base}.png"
  if [ ! -f "$img_file" ]; then
    img_file="${FIRST_DIR}/${base}.jpg"
  fi
  if [ ! -f "$img_file" ]; then
    echo "[rank $WORKER_RANK] [WARN] no image for $base, skip."
    continue
  fi

  # 输出
  outdir="${OUT_ROOT}/${base}"
  mkdir -p "$outdir"
  outmp4="${outdir}/${base}.mp4"

  # 已有就跳
  if [ -f "$outmp4" ]; then
    echo "[rank $WORKER_RANK] [SKIP] $base already done."
    continue
  fi

  # 只拿txt第一行
  first_line="$(head -n 1 "$prompt_file")"

  echo "[rank $WORKER_RANK] run $base -> $outmp4"

  # ====== 不带LoRA的版本，用你贴的那条 ======
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  python -m examples.video2world \
    --model_size 14B \
    --input_path "$img_file" \
    --num_conditional_frames 1 \
    --prompt "$first_line" \
    --save_path "$outmp4" \
    --disable_prompt_refiner \
    --disable_guardrail
    # 如果这个脚本也支持手动指定权重，就加：
    # --dit_path "$DIT_PATH"
  # ==========================================

  echo "[rank $WORKER_RANK] done $base"
done

echo "[rank $WORKER_RANK] all assigned samples done."