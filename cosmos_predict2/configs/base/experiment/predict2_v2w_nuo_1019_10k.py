# 导入必要的库
from hydra.core.config_store import ConfigStore
# from omegaconf import MISSING
# from lazy_loader import LazyLoader
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2.data.dataset_video import Dataset
# WARNING:
# 
# EXP="predict2_video2world_lora_training_2b_custom_data"
# Your EXP correspond to the configs setting
# 

def get_sampler(dataset) -> DistributedSampler:
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )
cs = ConfigStore.instance()

from imaginaire.lazy_config import LazyCall as L

print("[DEBUG] loaded experiment: predict2_video2world_lora_training_2b_custom_data")

custom_video_lora_dataset = L(Dataset)(
    dataset_dir="data_fsyn/",  # <--- 修改这里！改成你的数据集路径
    num_frames=93,                                          # 视频帧数
    # video_size=(480,832),                                 # resolution (H * W)
    video_size=(704,1280) 
)

dataloader_video_train_lora = L(DataLoader)(
    dataset=custom_video_lora_dataset,
    sampler=L(get_sampler)(dataset=custom_video_lora_dataset),
    batch_size=1,          #  batch size
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)


# =================================================================================
# 2. 为 2B 模型创建 LoRA 训练配置
# =================================================================================
predict2_video2world_lora_training_2b_custom_data = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_2b"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world_lora",
        name="2b_custom_data_nuo_1019",  # Name the experiment --> affect the output directory path
    ),
    model=dict(
        config=dict(
            train_architecture="lora",  # 启用 LoRA
            lora_rank=24, # default 16
            lora_alpha=24,  # default 16
            # lora_target_modules="q_proj,k_proj,v_proj",
            lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2", # default
            init_lora_weights=True,
            # ---------------------
            pipe_config=dict(
                ema=dict(enabled=True),
                prompt_refiner_config=dict(enabled=False),
                guardrail_config=dict(enabled=False),
            ),
        )
    ),

    model_parallel=dict(
        context_parallel_size=4,
    ),
    dataloader_train=dataloader_video_train_lora,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(iter_speed=dict(hit_thres=10)),
        max_iter=10000,                      # LoRA 训练迭代次数
    ),
    
    checkpoint=dict(
        save_iter=500,                      # 2k 保存一次
    ),
    optimizer=dict(
        # lr=2 ** (-12),
        # lr = 1e-5                      # LoRA 使用较高的学习率
        lr=2 ** (-9.5),
    ),
    scheduler=dict(        # 老师 
        warm_up_steps=[0],
        cycle_lengths=[10_000],     # 最大迭代数
        f_max=[0.6],
        f_min=[0.0],
    ),
)


# # =================================================================================
# # 3. 为 14B 模型创建 LoRA 训练配置 (可选)
# # =================================================================================
# predict2_video2world_lora_training_14b_custom_data = dict(
#     defaults=[
#         {"override /model": "predict2_video2world_fsdp_14b"},
#         {"override /optimizer": "fusedadamw"},
#         {"override /scheduler": "lambdalinear"},
#         {"override /ckpt_type": "standard"},
#         {"override /dataloader_val": "mock"},
#         "_self_",
#     ],
#     job=dict(
#         project="posttraining",
#         group="video2world_lora",
#         name="14b_my_custom_data", # <--- 你可以给你的实验起一个名字
#     ),
#     model=dict(
#         config=dict(
#             train_architecture="lora",  # 启用 LoRA
#             # --- LoRA 关键参数 (为大模型调整) ---
#             lora_rank=32,
#             lora_alpha=32,
#             lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
#             init_lora_weights=True,
#             # --------------------------------
#             pipe_config=dict(
#                 ema=dict(enabled=True),
#                 prompt_refiner_config=dict(enabled=False),
#                 guardrail_config=dict(enabled=False),
#             ),
#         )
#     ),
#     model_parallel=dict(
#         context_parallel_size=4,
#     ),
#     dataloader_train=dataloader_video_train_lora, # 同样使用上面定义好的数据加载器
#     trainer=dict(
#         distributed_parallelism="fsdp",
#         callbacks=dict(iter_speed=dict(hit_thres=10)),
#         max_iter=1500,
#     ),
#     checkpoint=dict(
#         save_iter=300,
#     ),
#     optimizer=dict(
#         lr=2 ** (-11),
#     ),
#     scheduler=dict(
#         warm_up_steps=[0],
#         cycle_lengths=[1_500],
#         f_max=[0.6],
#         f_min=[0.0],
#     ),
# )



for _item in [
    predict2_video2world_lora_training_2b_custom_data,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    cs.store(
        group="experiment",
        package="_global_",
        name="predict2_video2world_lora_training_2b_1019nuo",
        node=_item,
    )
