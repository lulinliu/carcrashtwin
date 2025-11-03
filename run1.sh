export NCCL_DEBUG=DEBUG
: "${ARNOLD_ID:=0}"                 # Set 0 for single node; cluster will set per node
: "${ARNOLD_WORKER_NUM:=1}"         # Set 1 for single node; cluster sets >1
: "${METIS_WORKER_0_HOST:=127.0.0.1}"  # Default to localhost for single node

MASTER_ADDR=${MASTER_ADDR:-$METIS_WORKER_0_HOST}    # [Required] Master node IP for multi-GPU training
MASTER_PORT=${MASTER_PORT:-22223}                   # Use fixed port (cluster-compatible)

# ======================
# Slurm auto-configuration (overrides defaults when available)
# ======================
if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
fi

NODE_RANK=${NODE_RANK:-${SLURM_PROCID:-$ARNOLD_ID}}
NNODES=${NNODES:-${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-$ARNOLD_WORKER_NUM}}}

torchrun \
  --nproc_per_node=1 \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m scripts.train \
  --config=cosmos_predict2/configs/base/config.py \
  -- experiment=predict2_video2world_lora_training_14b_1030nuo_14b    model.config.train_architecture=lora
