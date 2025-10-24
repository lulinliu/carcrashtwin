#!/bin/bash
#SBATCH -J cosmos_8n1g
#SBATCH -A CGAI24022    
#SBATCH --partition=gh
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 16:00:00
#SBATCH -o /scratch/10102/hh29499/carcrashtwin/%x_%j.out
#SBATCH -e /scratch/10102/hh29499/carcrashtwin/%x_%j.err


set -euo pipefail

#################### 可按需修改的变量 ####################
# 两个 manifest：各有 10 行，逐行给出“每个节点要用的 txt 列表文件”的路径
VIDEOS_MANIFEST="tasks_videos_manifest.txt"
PROMPTS_MANIFEST="tasks_prompts_manifest.txt"

# 每个节点对应的列表 txt 里：逐行给出单个视频路径 / 单个 prompt，一一对应
OUTPUT_ROOT="/scratch/10102/hh29499/carcrashtwin/output_video2world_$(date +%Y%m%d_%H%M%S)"

MODEL_SIZE="2B"
COND_FRAMES="1"

# 你的代码根目录（含 examples/video2world 模块）
PROJECT_DIR="/scratch/10102/hh29499/carcrashtwin"
CONDA_ENV="cosmos-predict2"
########################################################

mkdir -p slurm_logs
mkdir -p "${OUTPUT_ROOT}"

echo "[INFO] Job starts at: $(date)"
echo "[INFO] Output root   : ${OUTPUT_ROOT}"

module load gcc/13.2.0
module load cuda/12.6

# 激活环境（两种方式择其一都写上，避免登录壳差异）
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source ~/.bashrc || true
  conda activate "${CONDA_ENV}" || true
fi

# 也允许使用 conda run（worker 里仍会检测一次）
export PROJECT_DIR OUTPUT_ROOT MODEL_SIZE COND_FRAMES \
       VIDEOS_MANIFEST PROMPTS_MANIFEST CONDA_ENV

cd "${PROJECT_DIR}"

# 每个任务各跑一个 worker；worker 会用 SLURM_PROCID 选取自己的那一行
srun --ntasks=${SLURM_NTASKS} bash ./run_worker.sh
echo "[INFO] Job ends at: $(date)"
