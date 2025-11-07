#!/usr/bin/env bash
set -euo pipefail

#################### 路径配置 ####################
REPO_ROOT="/scratch/10102/hh29499/carcrashtwin"

# 原来的图像 & prompt
FIRST_DIR="${REPO_ROOT}/metric300/test_set_first"
PROMPT_DIR="${REPO_ROOT}/metric300/txt1027_test_set"

# 只跑缺的：这个是你刚才生成的文件
MISSING_LIST="${REPO_ROOT}/missing_videos.txt"

# 新的输出目录（和原来的分开）
OUT_ROOT="${REPO_ROOT}/cosmos_out_van14b_missing"
mkdir -p "$OUT_ROOT"

# 如果要手动指定权重，在这里开
# DIT_PATH="${REPO_ROOT}/real_world_3k/checkpoints/model/iter_000003000.pt"

#################### 分布式相关（slurm 传进来） ####################
WORLD_SIZE="${SLURM_NTASKS:-${WORLD_SIZE:-1}}"
WORKER_RANK="${SLURM_PROCID:-${WORKER_RANK:-0}}"
GPU_ID="${SLURM_LOCALID:-${GPU_ID:-0}}"

cd "$REPO_ROOT"

echo "[INFO][rank ${WORKER_RANK}/${WORLD_SIZE}] run missing list: $MISSING_LIST"

# 一行一行读 missing_videos.txt
i=0
while IFS= read -r base; do
  # 空行跳过
  [ -z "$base" ] && continue

  # 做分配：多机多卡时按行号分
  if (( i % WORLD_SIZE != WORKER_RANK )); then
    i=$((i+1))
    continue
  fi

  # 找图：先 png 再 jpg
  img_file="${FIRST_DIR}/${base}.png"
  if [ ! -f "$img_file" ]; then
    img_file="${FIRST_DIR}/${base}.jpg"
  fi
  if [ ! -f "$img_file" ]; then
    echo "[rank $WORKER_RANK] [WARN] image not found for $base, skip."
    i=$((i+1))
    continue
  fi

  # 找 prompt
  prompt_file="${PROMPT_DIR}/${base}.txt"
  if [ ! -f "$prompt_file" ]; then
    echo "[rank $WORKER_RANK] [WARN] prompt not found for $base, skip."
    i=$((i+1))
    continue
  fi

  # 输出放新的目录里，一个子目录一个视频
  outdir="${OUT_ROOT}/${base}"
  mkdir -p "$outdir"
  outmp4="${outdir}/${base}.mp4"

  # 已经生成过就跳
  if [ -f "$outmp4" ]; then
    echo "[rank $WORKER_RANK] [SKIP] $base already exists in missing out."
    i=$((i+1))
    continue
  fi

  # 只拿第一行 prompt
  first_line="$(head -n 1 "$prompt_file")"

  echo "[rank $WORKER_RANK] run $base -> $outmp4"

  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  python -m examples.video2world \
    --model_size 14B \
    --input_path "$img_file" \
    --num_conditional_frames 1 \
    --prompt "$first_line" \
    --save_path "$outmp4" \
    --disable_prompt_refiner \
    --disable_guardrail
    # 如果要手动指定权重就把上面的命令改成：
    # --dit_path "$DIT_PATH"

  echo "[rank $WORKER_RANK] done $base"

  i=$((i+1))
done < "$MISSING_LIST"

echo "[rank $WORKER_RANK] all missing samples done."
