#!/bin/bash
#SBATCH --job-name=cosmos
#SBATCH --account=CGAI24022
#SBATCH --nodes=8
#SBATCH --time=16:00:00
#SBATCH --partition=gh
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --ntasks-per-node=8               # 1 task per GPU
#SBATCH --gpus-per-node=8                 # 8 GPUs per GH node
#SBATCH --cpus-per-task=8                 # tweak if needed
#SBATCH --mem=0                           # take full node memory

# -------- User config --------
export EXP=predict2_video2world_lora_training_2b_1019nuo_full
CONFIG="cosmos_predict2/configs/base/config.py"
CONDA_ENV="cosmos-predict2"

set -euo pipefail
mkdir -p slurm_logs

# -------- (Optional) Environment setup --------
module purge
module load cuda/12.6
module load gcc/13.2.0

# Conda env
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
fi

# -------- NCCL / Distributed hints (Grace Hopper) --------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NCCL_DEBUG=warn
export TORCH_DISTRIBUTED_DEBUG=OFF
# Infiniband/UCX environments vary; these are safe starters:
export NCCL_IB_HCA=${NCCL_IB_HCA:-"^docker,lo"}       # ignore docker, loopback
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-"^lo,docker0"}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-0}
export NCCL_IB_TC=${NCCL_IB_TC:-"106"}                # ECN friendly
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-"NVL"}        # prefer NVLink when available
# For GH200 (can help on some fabrics):
export NCCL_CROSS_NIC=1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple

# -------- Master setup --------
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR
export MASTER_PORT=12341

# Derive nproc per node from SLURM
NPROC_PER_NODE="${SLURM_NTASKS_PER_NODE:-8}"
if [[ -n "${SLURM_GPUS_ON_NODE-}" ]]; then
  NPROC_PER_NODE="$SLURM_GPUS_ON_NODE"
fi

echo "Job        : $SLURM_JOB_ID"
echo "Nodes      : $SLURM_NNODES"
echo "GPUs/node  : $NPROC_PER_NODE"
echo "MASTER     : $MASTER_ADDR:$MASTER_PORT"
echo "Experiment : $EXP"
echo "Config     : $CONFIG"
echo "Start time : $(date)"

# -------- Launch --------
# torchrun handles multi-node when MASTER_* and --nnodes are set.
torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  -m scripts.train \
  --config="$CONFIG" \
  -- experiment="${EXP}"
