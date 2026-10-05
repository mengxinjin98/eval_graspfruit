import torch
import numpy as np
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from mujoco_env.y_env import SimulationEnv

"""
lerobot-train \
    --policy.type=act \
    --dataset.repo_id="graspfruit_demo" \
    --dataset.root="./graspfruit_demo" \
    --output_dir="./outputs/train/my_act_batch8" \
    --job_name="my_act_training" \
    --batch_size=8 \
    --steps=100000 \
    --save_freq=5000 \
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
    model_id = "./outputs/train/my_act_batch8/checkpoints/100000/pretrained_model"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. 加载策略模型
    print(f"正在从 '{model_id}' 加载策略...")
    policy = ACTPolicy.from_pretrained(model_id)
    policy.eval()
    policy.to(device)
    print("策略加载完成!")

    # 3. 创建预处理和后处理管道
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=model_id
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
