import os
import time
import shutil
import numpy as np
from PIL import Image
from mujoco_env.y_env import SimulationEnv
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# REPO_NAME = "grasp_apple_dataset"
# REPO_PATH = "./grasp_apple_dataset"
# TASK_NAME = "Put the apple in the blue bin."
REPO_NAME = "graspfruit_demo"
REPO_PATH = "./graspfruit_demo"
TASK_NAME = "Put the fruits in the bins: 1. put the apple in the blue bin; 2. put the banana in the green bin; 3. put the pear in the green bin."
NUM_DEMO = 10
SEED = 50


def main():
    create_new = True
    if os.path.exists(REPO_PATH):
        print(f"Directory {REPO_PATH} already exists!")
        ans = input("Do you want to delete it? (y/n) ")
        if ans == "y":
            shutil.rmtree(REPO_PATH)
        else:
            create_new = False

    if create_new:
        print(f"A new dataset named {REPO_NAME} will be created!")
        dataset = LeRobotDataset.create(
            repo_id=REPO_NAME,
            fps=30,  # 30 frames per second
            features={
                "observation.images.front": {
                    "dtype": "video",
                    "shape": (512, 512, 3),
                    "names": ["height", "width", "channels"],
                },
                "observation.images.wrist": {
                    "dtype": "video",
                    "shape": (512, 512, 3),
                    "names": ["height", "width", "channels"],
                },
                "observation.images.top": {
                    "dtype": "video",
                    "shape": (512, 512, 3),
                    "names": ["height", "width", "channels"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["state"],  # 6 joint angles and 1 gripper
                },
                "action": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["action"],  # 6 joint angles and 1 gripper
                },
            },
            root=REPO_PATH,
            robot_type="ur5e",
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=8,
            video_backend="pyav",
        )
    else:
        print("Load from previous dataset!")
        dataset = LeRobotDataset.resume(
            repo_id=REPO_NAME, root=REPO_PATH, video_backend="pyav"
        )

    episode_idx = 0
    record_flag = False
    # GraspFruitEnv = SimulationEnv(name="GraspFruitEnv", seed=SEED)
    GraspFruitEnv = SimulationEnv(name="GraspFruitEnv", seed=SEED + episode_idx)
    while GraspFruitEnv.env.is_viewer_alive() and episode_idx < NUM_DEMO:
        time.sleep(0.002)
        if GraspFruitEnv.env.loop_every(Hz=30):
            # print(f"len = {dataset.writer.episode_buffer.get("size", 0)}")
            action, reset, done = GraspFruitEnv.teleop_robot()
            if done:
                print(f"[Episode {episode_idx}]: finish recording!")
                dataset.save_episode()
                episode_idx += 1
                if episode_idx == NUM_DEMO:
                    break
                GraspFruitEnv.reset(seed=SEED + episode_idx)
                record_flag = False
            if reset:
                print(f"[Episode {episode_idx}]: reset environment!")
                dataset.clear_episode_buffer()
                GraspFruitEnv.reset(seed=SEED + episode_idx)
                record_flag = False
            if not record_flag and any(action != GraspFruitEnv.env.data.ctrl[:7]):
                print(f"[Episode {episode_idx}]: start recording...")
                record_flag = True
            front_image, wrist_image, top_image = GraspFruitEnv.grab_image()
            # # resize to 512x512
            # front_image = Image.fromarray(front_image)
            # wrist_image = Image.fromarray(wrist_image)
            # top_image = Image.fromarray(top_image)
            # front_image = front_image.resize((512, 512))
            # wrist_image = wrist_image.resize((512, 512))
            # top_image = top_image.resize((512, 512))
            # front_image = np.array(front_image)
            # wrist_image = np.array(wrist_image)
            # top_image = np.array(top_image)
            state = GraspFruitEnv.env.data.ctrl[:7]
            if record_flag:
                # add the frame to the dataset
                dataset.add_frame(
                    {
                        "observation.images.front": front_image,
                        "observation.images.wrist": wrist_image,
                        "observation.images.top": top_image,
                        "observation.state": state.astype(np.float32),
                        "action": action.astype(np.float32),
                        "task": TASK_NAME,
                    }
                )
            GraspFruitEnv.render()
            GraspFruitEnv.step(action)
        else:
            action = GraspFruitEnv.env.data.ctrl[:7]
            # GraspFruitEnv.grab_image()
            # GraspFruitEnv.render()
            GraspFruitEnv.step(action)

    dataset.finalize()
    GraspFruitEnv.env.close_viewer()
    shutil.rmtree(REPO_PATH + "/images")


if __name__ == "__main__":
    main()
