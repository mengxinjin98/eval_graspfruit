import glfw
import mujoco
import numpy as np
from mujoco_env.mujoco_parser import MuJoCoParserClass
from mujoco_env.utils import add_title_to_img

HOME_Q = np.asarray([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 1.5708])
# FRUIT_SPECS = (
#     ("apple_0", "apple", 1.05, 0.075),
#     ("apple_1", "apple", 0.95, 0.065),
#     ("apple_2", "apple", 1.0, 0.055),
#     ("banana_0", "banana", 1.05, 0.055),
#     ("banana_1", "banana", 0.95, 0.065),
#     ("banana_2", "banana", 1.0, 0.075),
#     ("pear_0", "pear", 1.05, 0.065),
#     ("pear_1", "pear", 0.95, 0.075),
#     ("pear_2", "pear", 1.0, 0.055),
# )
FRUIT_SPECS = (
    ("apple_0", "apple", 1.0, 0.075),
    ("banana_0", "banana", 1.0, 0.075),
    ("pear_0", "pear", 1.0, 0.075),
)
# FRUIT_SPECS = (("apple_0", "apple", 1.0, 0.075),)
FRUIT_BODIES = [item[0] for item in FRUIT_SPECS]
FRUIT_SPAWN_Z = {item[0]: item[3] for item in FRUIT_SPECS}


class SimulationEnv:
    def __init__(self, name, seed=10):
        """
        args:
            name: str, name of the environment
            seed: int, seed for random number generator
        """
        self.env = MuJoCoParserClass(name=name)
        self.env.init_viewer(
            distance=2.0,
            elevation=-30,
            transparent=False,
            black_sky=True,
            use_rgb_overlay=False,
            loc_rgb_overlay="top right",
        )
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.arm_joint_ids = [
            mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        ]
        self.arm_qpos = np.asarray(
            [self.env.model.jnt_qposadr[i] for i in self.arm_joint_ids]
        )
        self.arm_dofs = np.asarray(
            [self.env.model.jnt_dofadr[i] for i in self.arm_joint_ids]
        )
        self.arm_ranges = self.env.model.jnt_range[self.arm_joint_ids]
        self.fruit_ids = [
            mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in FRUIT_BODIES
        ]
        self.fruit_joint_ids = [self.env.model.body_jntadr[i] for i in self.fruit_ids]
        self.site_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_pinch"
        )
        self._planning_data = mujoco.MjData(self.env.model)
        self.reset(seed)

    def reset(self, seed: int):
        self.env.reset()
        self.env.data.ctrl[:7] = np.r_[HOME_Q, 0]
        rng = np.random.default_rng(seed)
        # Randomly distribute classes over a centred, generously spaced 3x3
        # array. Per-fruit yaw stays fully random and positions retain
        # millimetre-scale jitter, so the scene remains natural rather than
        # looking deliberately grouped by class.
        slots = np.asarray(
            [[x, y] for x in (0.1, 0.3, 0.5) for y in (-0.2, 0.0, 0.2)],
            dtype=float,
        )
        slots = slots[rng.permutation(len(slots))]
        slots = slots[:3]
        # slots = slots[:1]
        for slot, (body_name, joint_id) in zip(
            slots, zip(FRUIT_BODIES, self.fruit_joint_ids, strict=True), strict=True
        ):
            qadr = self.env.model.jnt_qposadr[joint_id]
            candidate = np.r_[
                slot + rng.uniform(-0.01, 0.01, size=2),
                FRUIT_SPAWN_Z[body_name] + 0.025,
            ]
            yaw = rng.uniform(-np.pi, np.pi)
            self.env.data.qpos[qadr : qadr + 7] = [
                *candidate,
                np.cos(yaw / 2),
                0,
                0,
                np.sin(yaw / 2),
            ]
        for _ in range(500):
            self.env.step()
        # Reject catastrophic spawn/settling drift. This is a simulation validity guard only;
        # poses are never exposed to the perception or planning pipeline.
        for body_id in self.fruit_ids:
            position = self.env.data.xpos[body_id]
            if not (
                0.0 < position[0] < 0.65
                and -0.35 < position[1] < 0.35
                and position[2] > 0.025
            ):
                name = mujoco.mj_id2name(
                    self.env.model, mujoco.mjtObj.mjOBJ_BODY, body_id
                )
                raise RuntimeError(
                    f"invalid randomized scene: {name} -> ({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}) left workspace"
                )

    def step(self, action):
        """
        Take a step in the environment
        args:
            action: np.array of shape (7,), action to take
        """
        self.env.step(action)

    def grab_image(self):
        """
        grab images from the environment
        returns:
            rgb_front: np.array, rgb image from the front view
            rgb_wrist: np.array, rgb image from the wrist view
            rgb_top: np.array, rgb image from the top view
        """
        self.rgb_front = self.env.get_fixed_cam_rgb(cam_name="front_camera")
        self.rgb_wrist = self.env.get_fixed_cam_rgb(cam_name="wrist_camera")
        self.rgb_top = self.env.get_fixed_cam_rgb(cam_name="top_camera")
        return self.rgb_front, self.rgb_wrist, self.rgb_top

    def render(self, teleop=False):
        """
        Render the environment
        """
        self.env.plot_time()

        rgb_front_view = add_title_to_img(
            self.rgb_front, text="Front View", shape=(512, 512)
        )
        rgb_wrist_view = add_title_to_img(
            self.rgb_wrist, text="Wrist View", shape=(512, 512)
        )
        rgb_top_view = add_title_to_img(self.rgb_top, text="Top View", shape=(512, 512))

        self.env.viewer_rgb_overlay(rgb_front_view, loc="top right")
        self.env.viewer_rgb_overlay(rgb_wrist_view, loc="bottom right")
        self.env.viewer_rgb_overlay(rgb_top_view, loc="top left")

        if teleop:
            self.env.viewer_text_overlay(
                text1="Key Pressed", text2="%s" % (self.env.get_key_pressed_list())
            )
            self.env.viewer_text_overlay(
                text1="Key Repeated", text2="%s" % (self.env.get_key_repeated_list())
            )

        self.env.render()

    def teleop_robot(self):
        """
        Teleoperate the robot using keyboard
        returns:
            action: np.array, action to take
            reset: bool, True if the user wants to reset the episode
            done: bool, True if the user wants to finish the episode
        keycodes:
            ---------      ----------------------
               W       ->         backward
            A  S  D        left   forward   right
            ---------      ----------------------
            In x, y plane

            ---------
            R: moving up
            F: moving down
            ---------
            In z axis

            ---------
            Q: tilt left
            E: tilt right
            Up: look upward
            Down: look downward
            Right: turn right
            Left: turn left
            ---------
            For rotation

            ---------
            Backspace: reset
            Enter: done
            Space: gripper open/close
            ---------
        """
        is_teleop_gripper = False
        target = np.eye(4)
        target[:3, 3] = self.env.data.site_xpos[self.site_id]
        target[:3, :3] = self.env.data.site_xmat[self.site_id].reshape(3, 3)
        target_gripper = self.env.data.ctrl[6]
        pos_step = 0.03
        rot_step = np.pi / 180 * 10
        if self.env.is_key_pressed_once(key=glfw.KEY_ENTER):
            return np.r_[HOME_Q, 0], False, True
        elif self.env.is_key_pressed_once(key=glfw.KEY_BACKSPACE):
            return np.r_[HOME_Q, 0], True, False
        elif self.env.is_key_pressed_repeat(key=glfw.KEY_S):
            target[0, 3] += pos_step
        elif self.env.is_key_pressed_repeat(key=glfw.KEY_W):
            target[0, 3] -= pos_step
        elif self.env.is_key_pressed_repeat(key=glfw.KEY_D):
            target[1, 3] += pos_step
        elif self.env.is_key_pressed_repeat(key=glfw.KEY_A):
            target[1, 3] -= pos_step
        elif self.env.is_key_pressed_repeat(key=glfw.KEY_R):
            target[2, 3] += pos_step
        elif self.env.is_key_pressed_repeat(key=glfw.KEY_F):
            target[2, 3] -= pos_step
        elif self.env.is_key_pressed_repeat(key=glfw.KEY_UP):
            rot_matrix = np.array(
                [
                    [np.cos(rot_step), 0, np.sin(rot_step)],
                    [0, 1, 0],
                    [-np.sin(rot_step), 0, np.cos(rot_step)],
                ]
            )
            target[:3, :3] = rot_matrix @ target[:3, :3]
        elif self.env.is_key_pressed_repeat(key=glfw.KEY_DOWN):
            rot_matrix = np.array(
                [
                    [np.cos(-rot_step), 0, np.sin(-rot_step)],
                    [0, 1, 0],
                    [-np.sin(-rot_step), 0, np.cos(-rot_step)],
                ]
            )
            target[:3, :3] = rot_matrix @ target[:3, :3]
        elif self.env.is_key_pressed_repeat(key=glfw.KEY_LEFT):
            rot_matrix = np.array(
                [
                    [np.cos(-rot_step), -np.sin(-rot_step), 0],
                    [np.sin(-rot_step), np.cos(-rot_step), 0],
                    [0, 0, 1],
                ]
            )
            target[:3, :3] = rot_matrix @ target[:3, :3]
        elif self.env.is_key_pressed_repeat(key=glfw.KEY_RIGHT):
            rot_matrix = np.array(
                [
                    [np.cos(rot_step), -np.sin(rot_step), 0],
                    [np.sin(rot_step), np.cos(rot_step), 0],
                    [0, 0, 1],
                ]
            )
            target[:3, :3] = rot_matrix @ target[:3, :3]
        elif self.env.is_key_pressed_repeat(key=glfw.KEY_Q):
            rot_matrix = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(-rot_step), -np.sin(-rot_step)],
                    [0, np.sin(-rot_step), np.cos(-rot_step)],
                ]
            )
            target[:3, :3] = rot_matrix @ target[:3, :3]
        elif self.env.is_key_pressed_repeat(key=glfw.KEY_E):
            rot_matrix = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(rot_step), -np.sin(rot_step)],
                    [0, np.sin(rot_step), np.cos(rot_step)],
                ]
            )
            target[:3, :3] = rot_matrix @ target[:3, :3]
        elif self.env.is_key_pressed_once(key=glfw.KEY_SPACE):
            is_teleop_gripper = True
            if target_gripper == 0:
                target_gripper = 255
            else:
                target_gripper = 0
        else:
            return self.env.data.ctrl[:7], False, False

        if is_teleop_gripper:
            target_q = self.env.data.ctrl[:6]
        else:
            target_q = self._solve_from_current(target)
        action = np.r_[target_q, target_gripper]
        return action, False, False

    def _solve_from_current(
        self, pose: np.ndarray, max_iterations: int = 100
    ) -> np.ndarray | None:
        """Solve on scratch state so the viewer can never display IK iterates."""
        planning = self._planning_data
        planning.qpos[:] = self.env.data.qpos
        planning.qvel[:] = self.env.data.qvel
        planning.ctrl[:] = self.env.data.ctrl
        mujoco.mj_forward(self.env.model, planning)
        return self._ik(target=pose, max_iterations=max_iterations, data=planning)

    def _ik(
        self,
        target: np.ndarray,
        max_iterations: int = 100,
        data=None,
    ) -> np.ndarray | None:
        data = self.env.data if data is None else data
        original = data.qpos.copy()
        best = data.qpos[self.arm_qpos].copy()
        best_error = float("inf")
        for _ in range(max_iterations):
            mujoco.mj_forward(self.env.model, data)
            current_pos = data.site_xpos[self.site_id]
            current_rot = data.site_xmat[self.site_id].reshape(3, 3)
            position_error = target[:3, 3] - current_pos
            rotation_error = 0.5 * (
                np.cross(current_rot[:, 0], target[:3, 0])
                + np.cross(current_rot[:, 1], target[:3, 1])
                + np.cross(current_rot[:, 2], target[:3, 2])
            )
            error = np.r_[position_error, 0.25 * rotation_error]
            norm = float(np.linalg.norm(error))
            if norm < best_error:
                best_error, best = norm, data.qpos[self.arm_qpos].copy()
            if (
                np.linalg.norm(position_error) < 0.0001
                and np.linalg.norm(rotation_error) < 0.0001
            ):
                result = data.qpos[self.arm_qpos].copy()
                data.qpos[:] = original
                mujoco.mj_forward(self.env.model, data)
                return result
            jacp = np.zeros((3, self.env.model.nv))
            jacr = np.zeros((3, self.env.model.nv))
            mujoco.mj_jacSite(self.env.model, data, jacp, jacr, self.site_id)
            jacobian = np.vstack(
                (jacp[:, self.arm_dofs], 0.25 * jacr[:, self.arm_dofs])
            )
            damping = 0.035
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping**2 * np.eye(6), error
            )
            data.qpos[self.arm_qpos] += np.clip(delta, -0.1, 0.1)
            data.qpos[self.arm_qpos] = np.clip(
                data.qpos[self.arm_qpos], self.arm_ranges[:, 0], self.arm_ranges[:, 1]
            )
        data.qpos[:] = original
        mujoco.mj_forward(self.env.model, data)
        return best if best_error < 0.001 else None
