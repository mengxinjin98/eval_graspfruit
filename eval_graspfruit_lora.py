import torch
import numpy as np
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from peft import PeftConfig, PeftModel
from lerobot.policies.factory import make_pre_post_processors
from mujoco_env.y_env import SimulationEnv

"""
lerobot-train \
    --policy.path="lerobot/smolvla_base" \
    --dataset.repo_id="graspfruit_demo" \
    --dataset.root="./graspfruit_demo" \
    --output_dir="./outputs/train/my_smolvla_batch8_lora" \
    --job_name="my_smolvla_training" \
    --rename_map="{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2", "observation.images.top": "observation.images.camera3"}" \
    --batch_size=8 \
    --steps=100000 \
    --save_freq=5000 \
    --policy.optimizer_lr=1e-3 \
    --policy.scheduler_decay_lr=1e-4 \
    --peft.method_type=LORA \
    --peft.r=64 \
    --peft.lora_alpha=64 \
    --wandb.enable=false \
    --policy.push_to_hub=false \
    --policy.device=cuda
"""

# TASK_NAME = "Put the apple in the blue bin."
TASK_NAME = "Put the fruits in the bins: 1. put the apple in the blue bin; 2. put the banana in the green bin; 3. put the pear in the green bin."
NUM_DEMO = 10
SEED = 10


def eval_policy():
    # 1. 配置评估参数
    base_model_id = "C:\\Users\\Mengxin Jin\\.cache\\huggingface\\hub\\models--lerobot--smolvla_base\\snapshots\\c83c3163b8ca9b7e67c509fffd9121e66cb96205"
    lora_adapter_path = (
        "./outputs/train/my_smolvla_batch8_lora/checkpoints/100000/pretrained_model"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. 加载策略模型
    print(f"正在从 '{lora_adapter_path}' 加载策略...")
    config = SmolVLAConfig.from_pretrained(lora_adapter_path)
    policy = SmolVLAPolicy.from_pretrained(base_model_id, config=config)
    # peft_model = PeftModel.from_pretrained(policy, lora_adapter_path)
    # policy.model = peft_model.base_model.model.model
    peft_config = PeftConfig.from_pretrained(lora_adapter_path)
    policy = PeftModel.from_pretrained(policy, lora_adapter_path, config=peft_config)
    # print("=== 检查实际模块名 ===")
    # for name, _ in policy.model.named_modules():
    #     if any(
    #         k in name
    #         for k in [
    #             "q_proj",
    #             "v_proj",
    #             "state_proj",
    #             "action_in_proj",
    #             "action_out_proj",
    #             "action_time_mlp",
    #         ]
    #     ):
    #         print(name)
    # print("=== 检查结束 ===")
    policy.eval()
    policy.to(device)
    print("策略加载完成!")

    # 3. 创建预处理和后处理管道
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=lora_adapter_path
    )

    episode_idx = 0
    GraspFruitEnv = SimulationEnv(name="GraspFruitEnv", seed=SEED)
    while GraspFruitEnv.env.is_viewer_alive() and episode_idx < NUM_DEMO:
        if GraspFruitEnv.env.loop_every(Hz=30):
            _, reset, done = GraspFruitEnv.teleop_robot()
            if done:
                print(f"[Episode {episode_idx}]: finish evaluation!")
                episode_idx += 1
                if episode_idx == NUM_DEMO:
                    break
                GraspFruitEnv.reset(seed=SEED + episode_idx)
                policy.reset()
            if reset:
                print(f"[Episode {episode_idx}]: reset environment!")
                GraspFruitEnv.reset(seed=SEED + episode_idx)
                policy.reset()

            front_image, wrist_image, top_image = GraspFruitEnv.grab_image()
            state = GraspFruitEnv.env.data.ctrl[:7]
            obs_frame = {
                "observation.images.front": front_image,
                "observation.images.wrist": wrist_image,
                "observation.images.top": top_image,
                "observation.state": state.astype(np.float32),
            }

            for name in obs_frame:
                tensor = torch.from_numpy(obs_frame[name].copy())
                if "image" in name:
                    if tensor.dtype == torch.uint8:
                        tensor = tensor.type(torch.float32) / 255
                    tensor = tensor.permute(2, 0, 1).contiguous()
                obs_frame[name] = tensor

            obs_frame["task"] = TASK_NAME
            obs = preprocessor(obs_frame)
            action = policy.select_action(obs)
            action = postprocessor(action)
            action = action.squeeze(0).numpy()
            GraspFruitEnv.render()
            GraspFruitEnv.step(action)
        else:
            action = GraspFruitEnv.env.data.ctrl[:7]
            # GraspFruitEnv.grab_image()
            # GraspFruitEnv.render()
            GraspFruitEnv.step(action)

    GraspFruitEnv.env.close_viewer()


if __name__ == "__main__":
    eval_policy()
