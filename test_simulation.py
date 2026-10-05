import time
from mujoco_env.y_env import SimulationEnv

SEED = 10


def main():
    GraspFruitEnv = SimulationEnv(name="GraspFruitEnv", seed=SEED)
    while GraspFruitEnv.env.is_viewer_alive():
        time.sleep(0.002)
        if GraspFruitEnv.env.loop_every(Hz=30):
            action, _, _ = GraspFruitEnv.teleop_robot()
            GraspFruitEnv.grab_image()
            GraspFruitEnv.render()
            GraspFruitEnv.step(action)
        else:
            action = GraspFruitEnv.env.data.ctrl[:7]
            # GraspFruitEnv.grab_image()
            # GraspFruitEnv.render()
            GraspFruitEnv.step(action)


if __name__ == "__main__":
    main()
