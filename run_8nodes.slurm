#!/bin/bash
#SBATCH -J cosmos_lora_8n
#SBATCH -A CGAI24022
#SBATCH -p gh
#SBATCH -N 8
#SBATCH -t 16:00:00
#SBATCH -o /scratch/10102/hh29499/carcrashtwin/%x_%j.out
#SBATCH -e /scratch/10102/hh29499/carcrashtwin/%x_%j.err
#SBATCH --ntasks-per-node=1

set -euo pipefail

module purge
module load gcc/13.2.0
module load cuda/12.6

# ---------- Your paths ----------
export CUDA_HOME=/home1/apps/nvidia/Linux_aarch64/24.9/cuda/12.6
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export CUDNN_PATH=/scratch/10102/hh29499/anaconda/envs/cosmos-predict2/lib/python3.10/site-packages/nvidia/cudnn
export CUDNN_INCLUDE_DIR="$CUDNN_PATH/include"
export CUDNN_LIBRARY="$CUDNN_PATH/lib/libcudnn.so"
export CPATH="$CUDNN_INCLUDE_DIR:${CPATH:-}"
export LIBRARY_PATH="$CUDNN_PATH/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDNN_PATH/lib:${LD_LIBRARY_PATH:-}"

export MATHLIBS=/home1/apps/nvidia/Linux_aarch64/24.9/math_libs/12.6
export CPATH="$MATHLIBS/include:${CPATH:-}"
export LIBRARY_PATH="$MATHLIBS/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$MATHLIBS/lib:${LD_LIBRARY_PATH:-}"

PROJECT_DIR=/scratch/10102/hh29499/carcrashtwin
cd "$PROJECT_DIR"

# caches (optional)
export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$HF_HOME/torch}"

# NCCL/CPU env
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=${MASTER_PORT:-12341}
NNODES=${SLURM_JOB_NUM_NODES}

srun --ntasks=${NNODES} --ntasks-per-node=1 --export=ALL --kill-on-bad-exit=1 bash -lc '
  module load gcc/13.2.0
  module load cuda/12.6

  # Re-apply these inside srun to be extra safe on each node
  export CUDA_HOME=/home1/apps/nvidia/Linux_aarch64/24.9/cuda/12.6
  export PATH="$CUDA_HOME/bin:${PATH:-}"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

  export CUDNN_PATH=/scratch/10102/hh29499/anaconda/envs/cosmos-predict2/lib/python3.10/site-packages/nvidia/cudnn
  export CUDNN_INCLUDE_DIR="$CUDNN_PATH/include"
  export CUDNN_LIBRARY="$CUDNN_PATH/lib/libcudnn.so"
  export CPATH="$CUDNN_INCLUDE_DIR:${CPATH:-}"
  export LIBRARY_PATH="$CUDNN_PATH/lib:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="$CUDNN_PATH/lib:${LD_LIBRARY_PATH:-}"

  export MATHLIBS=/home1/apps/nvidia/Linux_aarch64/24.9/math_libs/12.6
  export CPATH="$MATHLIBS/include:${CPATH:-}"
  export LIBRARY_PATH="$MATHLIBS/lib:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="$MATHLIBS/lib:${LD_LIBRARY_PATH:-}"

  # conda
  set +u
  source /scratch/10102/hh29499/anaconda/etc/profile.d/conda.sh
  conda activate cosmos-predict2
  set -u

  # avoid local shadowing
  unset PYTHONPATH

  # sanity checks
  python - << "PY"
import sys, os
import tokenizers
print("tokenizers from:", tokenizers.__file__)
assert "site-packages/tokenizers" in tokenizers.__file__, "Still shadowed by local package!"

import decord, pathlib
from decord import VideoReader, cpu
so = pathlib.Path(decord.__file__).with_name("libdecord.so")
print("decord py:", decord.__file__)
print("libdecord.so exists:", so.exists(), "->", so)
assert so.exists(), "libdecord.so not found next to decord; reinstall decord properly."
print("decord OK:", VideoReader, cpu)
PY

  export NODE_RANK=${SLURM_PROCID}
  cd /scratch/10102/hh29499/carcrashtwin

  torchrun \
    --nproc_per_node=1 \
    --nnodes='"${NNODES}"' \
    --node_rank=${NODE_RANK} \
    --master_addr='"${MASTER_ADDR}"' \
    --master_port='"${MASTER_PORT}"' \
    -m scripts.train \
    --config=cosmos_predict2/configs/base/config.py \
    -- experiment=predict2_video2world_lora_training_2b_1019nuo_full
'
