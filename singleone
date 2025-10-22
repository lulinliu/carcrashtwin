torchrun \
  --nproc_per_node=1 \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m scripts.train \
  --config=cosmos_predict2/configs/base/config.py \
  -- experiment=predict2_video2world_lora_training_2b_1019nuo_full

  
