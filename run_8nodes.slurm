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

PROJECT_DIR=/scratch/10102/hh29499/carcrashtwin
cd "$PROJECT_DIR"

# ---- SAFE defaults under `set -u` ----
: "${PYTHONPATH:=}"
: "${HF_HOME:=$SCRATCH/.cache/huggingface}"
: "${TRANSFORMERS_CACHE:=$HF_HOME/hub}"
: "${TORCH_HOME:=$HF_HOME/torch}"

# ✅ 只加入你自己的包目录，避免遮蔽第三方库（不要加仓库根）
export PYTHONPATH="$PROJECT_DIR/cosmos_predict2:$PROJECT_DIR/imaginaire:${PYTHONPATH}"

export HF_HOME TRANSFORMERS_CACHE TORCH_HOME

# NCCL/CPU env
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
# 如遇网卡冲突/超时再启用：
# export NCCL_SOCKET_IFNAME=^lo,docker0
export OMP_NUM_THREADS=8

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=${MASTER_PORT:-12341}
NNODES=${SLURM_JOB_NUM_NODES}

srun --ntasks=${NNODES} --ntasks-per-node=1 --export=ALL --kill-on-bad-exit=1 bash -lc '
  module load gcc/13.2.0
  module load cuda/12.6   # ✅ 在 srun 中也要加载

  # 选用更新的 libstdc++（修复 GLIBCXX_3.4.32 之类）
  GCL=$(gcc --print-file-name=libstdc++.so.6)
  export LD_LIBRARY_PATH="$(dirname "$GCL"):${LD_LIBRARY_PATH:-}"

  # Conda activate 的 hook 可能用到未定义变量，避开 set -u
  set +u
  source /scratch/10102/hh29499/anaconda/etc/profile.d/conda.sh
  conda activate cosmos-predict2
  set -u

  echo "Node=$(hostname)"
  echo "libstdc++ -> $(readlink -f "$(dirname "$GCL")/libstdc++.so.6")"

  # ---- Sanity checks：在多节点启动前快速失败 ----
  python - <<'"'"'PY'"'"'
import importlib.util as u, sys
# 1) transformers & tokenizers 不应被项目同名包遮蔽
spec_tok = u.find_spec("tokenizers")
print("tokenizers spec:", spec_tok)
from tokenizers import Tokenizer
print("Tokenizer OK:", Tokenizer)

# 2) decord 要能导入 VideoReader
from decord import VideoReader, cpu
print("decord OK:", VideoReader, cpu)
PY
  # ---- Sanity checks end ----

  export NODE_RANK=${SLURM_PROCID}
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
