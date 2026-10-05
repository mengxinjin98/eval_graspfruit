from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig
from .models import BinName, GraspCandidate, Observation, PipelineError, PipelineState
from .scene_builder import FRUIT_SPECS, build_official_scene

ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
FRUIT_BODIES = [item[0] for item in FRUIT_SPECS]
FRUIT_SPAWN_Z = {item[0]: item[3] for item in FRUIT_SPECS}
BIN_CENTERS = {
    BinName.BLUE: np.asarray([0.27, -0.50, 0.25]),
    BinName.GREEN: np.asarray([0.27, 0.50, 0.25]),
}
BIN_DROP_OFFSETS = {
    BinName.BLUE: ((0.0, 0.0), (0.055, 0.0), (-0.055, 0.0)),
    BinName.GREEN: (
        (0.0, 0.0),
        (0.065, 0.0),
        (-0.065, 0.0),
        (0.0, 0.055),
        (0.0, -0.055),
        (0.065, 0.055),
    ),
}
HOME_Q = np.asarray([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

# GraspNet uses local +X as the approach direction.  The official Robotiq
# model uses local +Z at its ``pinch`` TCP.  This proper rotation maps a
# GraspNet grasp frame to the physical TCP without changing the grasp centre.
GRASPNET_TO_TCP = np.asarray(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float
)


@dataclass
class MujocoFruitEnv:
    config: SimulationConfig
    headless: bool = False

    def __post_init__(self) -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("MuJoCo is not installed; install the sim extra") from exc
        self.mj = mujoco
        self.model = build_official_scene(self.config.timestep)
        self.data = mujoco.MjData(self.model)
        # IK feasibility checks must never mutate the data displayed by the
        # viewer or used by execution. The passive viewer reads self.data from
        # another thread and can otherwise expose every numerical IK iterate as
        # seemingly random robot motion during PLAN_GRASP.
        self._planning_data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(
            self.model, height=self.config.image_height, width=self.config.image_width
        )
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
        self.presentation_camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "presentation"
        )
        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_pinch"
        )
        self.arm_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS
        ]
        self.arm_qpos = np.asarray([self.model.jnt_qposadr[i] for i in self.arm_joint_ids])
        self.arm_dofs = np.asarray([self.model.jnt_dofadr[i] for i in self.arm_joint_ids])
        self.arm_ranges = self.model.jnt_range[self.arm_joint_ids].copy()
        self.last_reachability_failure = ""
        self.fruit_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in FRUIT_BODIES
        ]
        self.fruit_joint_ids = [self.model.body_jntadr[i] for i in self.fruit_ids]
        self.fruit_collision_geoms = [
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_collision"
            )
            for name in FRUIT_BODIES
        ]
        self.finger_pad_geoms = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in (
                "gripper_right_pad1",
                "gripper_right_pad2",
                "gripper_left_pad1",
                "gripper_left_pad2",
            )
        }
        self.viewer = None
        if not self.headless:
            try:
                import mujoco.viewer

                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception:  # noqa: BLE001 - viewer is optional; headless simulation remains valid
                self.viewer = None
        self._held_joint: int | None = None
        self._held_offset = np.eye(4)
        self._frame_callback: Callable[[np.ndarray], None] | None = None
        self._presentation_frame_callback: Callable[[np.ndarray], None] | None = None
        self._record_fps = 30
        self._next_record_time = 0.0
        self.reset(7)

    def enable_recording(
        self,
        callback: Callable[[np.ndarray], None],
        fps: int = 30,
        presentation_callback: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        """Record synchronised decision and presentation camera streams."""
        self._frame_callback = callback
        self._presentation_frame_callback = presentation_callback
        self._record_fps = fps
        self._next_record_time = float(self.data.time)
        self._capture_motion_frame()
        self._next_record_time += 1.0 / self._record_fps

    def _capture_motion_frame(self) -> None:
        if self._frame_callback is None:
            return
        self.renderer.update_scene(self.data, camera=self.camera_id)
        self._frame_callback(self.renderer.render())
        if self._presentation_frame_callback is not None:
            self.renderer.update_scene(self.data, camera=self.presentation_camera_id)
            self._presentation_frame_callback(self.renderer.render())

    def reset(self, seed: int) -> None:
        self.mj.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qpos] = HOME_Q
        self.data.ctrl[:6] = HOME_Q
        self.data.ctrl[6] = 0.0
        self._held_joint = None
        self._bin_drop_counts = {BinName.BLUE: 0, BinName.GREEN: 0}
        rng = np.random.default_rng(seed)
        # Randomly distribute classes over a centred, generously spaced 3x3
        # array. Per-fruit yaw stays fully random and positions retain
        # millimetre-scale jitter, so the scene remains natural rather than
        # looking deliberately grouped by class.
        slots = np.asarray(
            [[x, y] for x in (0.22, 0.41, 0.60) for y in (-0.29, 0.0, 0.29)],
            dtype=float,
        )
        slots = slots[rng.permutation(len(slots))]
        for slot, (body_name, joint_id) in zip(
            slots, zip(FRUIT_BODIES, self.fruit_joint_ids, strict=True), strict=True
        ):
            qadr = self.model.jnt_qposadr[joint_id]
            candidate = np.r_[slot + rng.uniform(-0.012, 0.012, size=2), FRUIT_SPAWN_Z[body_name] + 0.045]
            yaw = rng.uniform(-np.pi, np.pi)
            self.data.qpos[qadr : qadr + 7] = [*candidate, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.mj.mj_forward(self.model, self.data)
        self._step(self.config.settle_steps)
        # Reject catastrophic spawn/settling drift. This is a simulation validity guard only;
        # poses are never exposed to the perception or planning pipeline.
        for body_id in self.fruit_ids:
            position = self.data.xpos[body_id]
            if not (0.15 < position[0] < 0.75 and -0.40 < position[1] < 0.40 and position[2] > 0.025):
                raise RuntimeError(f"invalid randomized scene: fruit body {body_id} left workspace")

    def _step(self, count: int, held: bool = True) -> None:
        for _ in range(count):
            wall_started = time.perf_counter()
            self.mj.mj_step(self.model, self.data)
            if held and self._held_joint is not None:
                self._follow_tcp()
            if (
                self._frame_callback is not None
                and self.data.time + self.model.opt.timestep * 0.5 >= self._next_record_time
            ):
                self._capture_motion_frame()
                while self._next_record_time <= self.data.time:
                    self._next_record_time += 1.0 / self._record_fps
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()
                remaining = self.model.opt.timestep - (time.perf_counter() - wall_started)
                if remaining > 0:
                    time.sleep(remaining)

    def _follow_tcp(self) -> None:
        tcp = np.eye(4)
        tcp[:3, :3] = self.data.site_xmat[self.site_id].reshape(3, 3)
        tcp[:3, 3] = self.data.site_xpos[self.site_id]
        pose = tcp @ self._held_offset
        qadr = self.model.jnt_qposadr[self._held_joint]
        self.data.qpos[qadr : qadr + 3] = pose[:3, 3]
        quat = np.empty(4)
        self.mj.mju_mat2Quat(quat, pose[:3, :3].reshape(-1))
        self.data.qpos[qadr + 3 : qadr + 7] = quat
        dof = self.model.jnt_dofadr[self._held_joint]
        self.data.qvel[dof : dof + 6] = 0
        self.mj.mj_forward(self.model, self.data)

    def _camera_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        fovy = np.deg2rad(self.model.cam_fovy[self.camera_id])
        fy = self.config.image_height / (2 * np.tan(fovy / 2))
        fx = fy
        intrinsics = np.asarray(
            [[fx, 0, self.config.image_width / 2], [0, fy, self.config.image_height / 2], [0, 0, 1]],
            dtype=float,
        )
        camera_body_rotation = self.data.cam_xmat[self.camera_id].reshape(3, 3)
        # MuJoCo camera looks along -Z with +Y up; convert to OpenCV +Z forward, +Y down.
        cv_to_mujoco = np.diag([1.0, -1.0, -1.0])
        camera_to_world = np.eye(4)
        camera_to_world[:3, :3] = camera_body_rotation @ cv_to_mujoco
        camera_to_world[:3, 3] = self.data.cam_xpos[self.camera_id]
        return intrinsics, camera_to_world

    def observe(self) -> Observation:
        self.renderer.update_scene(self.data, camera=self.camera_id)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.camera_id)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        intrinsics, camera_to_world = self._camera_matrices()
        return Observation(
            rgb=rgb,
            depth_m=depth.astype(np.float32),
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            timestamp=time.time(),
        )

    def _ik(
        self,
        target: np.ndarray,
        max_iterations: int = 180,
        data=None,
    ) -> np.ndarray | None:
        data = self.data if data is None else data
        original = data.qpos.copy()
        best = data.qpos[self.arm_qpos].copy()
        best_error = float("inf")
        for _ in range(max_iterations):
            self.mj.mj_forward(self.model, data)
            current_pos = data.site_xpos[self.site_id].copy()
            current_rot = data.site_xmat[self.site_id].reshape(3, 3).copy()
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
            if np.linalg.norm(position_error) < 0.008 and np.linalg.norm(rotation_error) < 0.15:
                result = data.qpos[self.arm_qpos].copy()
                data.qpos[:] = original
                self.mj.mj_forward(self.model, data)
                return result
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            self.mj.mj_jacSite(self.model, data, jacp, jacr, self.site_id)
            jacobian = np.vstack((jacp[:, self.arm_dofs], 0.25 * jacr[:, self.arm_dofs]))
            damping = 0.035
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping**2 * np.eye(6), error
            )
            data.qpos[self.arm_qpos] += np.clip(delta, -0.08, 0.08)
            data.qpos[self.arm_qpos] = np.clip(
                data.qpos[self.arm_qpos], self.arm_ranges[:, 0], self.arm_ranges[:, 1]
            )
        data.qpos[:] = original
        self.mj.mj_forward(self.model, data)
        return best if best_error < 0.04 else None

    def _solve_from_current(self, pose: np.ndarray, max_iterations: int = 180) -> np.ndarray | None:
        """Solve on scratch state so the viewer can never display IK iterates."""
        planning = self._planning_data
        planning.qpos[:] = self.data.qpos
        planning.qvel[:] = self.data.qvel
        planning.ctrl[:] = self.data.ctrl
        self.mj.mj_forward(self.model, planning)
        return self._ik(pose, max_iterations=max_iterations, data=planning)

    def _move_pose(self, pose: np.ndarray, stage: PipelineState = PipelineState.PICK) -> None:
        target_q = self._solve_from_current(pose)
        if target_q is None:
            raise PipelineError(stage, "IK failed during execution")
        start = self.data.qpos[self.arm_qpos].copy()
        for alpha in np.linspace(0, 1, self.config.control_steps):
            smooth = 3 * alpha**2 - 2 * alpha**3
            self.data.ctrl[:6] = (1 - smooth) * start + smooth * target_q
            self._step(1)
        # The last interpolation sample used to apply the target for only one
        # 2 ms physics step. Hold the command until the position-controlled arm
        # has actually converged before advancing to close/lift/place.
        converged = False
        for _ in range(max(240, self.config.control_steps * 2)):
            self.data.ctrl[:6] = target_q
            self._step(1)
            joint_error = float(
                np.max(np.abs(self.data.qpos[self.arm_qpos] - target_q))
            )
            position_error = float(
                np.linalg.norm(self.data.site_xpos[self.site_id] - pose[:3, 3])
            )
            if joint_error < 0.025 and position_error < 0.018:
                converged = True
                break
        # Loaded position actuators have a small steady-state error.  Apply a
        # bounded joint-error compensation and re-measure the TCP instead of
        # accepting a loose Cartesian threshold.
        for _ in range(3):
            if converged:
                break
            joint_residual = target_q - self.data.qpos[self.arm_qpos]
            compensated = np.clip(
                target_q + joint_residual,
                self.model.actuator_ctrlrange[:6, 0],
                self.model.actuator_ctrlrange[:6, 1],
            )
            self.data.ctrl[:6] = compensated
            self._step(240)
            joint_error = float(
                np.max(np.abs(self.data.qpos[self.arm_qpos] - target_q))
            )
            position_error = float(
                np.linalg.norm(self.data.site_xpos[self.site_id] - pose[:3, 3])
            )
            converged = joint_error < 0.025 and position_error < 0.018
        if not converged:
            position_error = float(
                np.linalg.norm(self.data.site_xpos[self.site_id] - pose[:3, 3])
            )
            raise PipelineError(
                stage,
                f"motion did not converge (TCP position error={position_error:.3f} m)",
            )

    def _set_gripper(self, closed: bool) -> None:
        self.data.ctrl[6] = 255.0 if closed else 0.0
        self._step(280)

    def _attach_nearest(self) -> bool:
        tcp = self.data.site_xpos[self.site_id]
        distances = [np.linalg.norm(self.data.xpos[body_id] - tcp) for body_id in self.fruit_ids]
        index = int(np.argmin(distances))
        if distances[index] > 0.10:
            return False
        fruit_geom = self.fruit_collision_geoms[index]
        pad_contacts = 0
        for contact in self.data.contact[: self.data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if fruit_geom in pair and pair.intersection(self.finger_pad_geoms):
                pad_contacts += 1
        # The assist stabilises a grasp only after the physical gripper has
        # genuinely touched the selected scanned mesh.  A near miss is never
        # converted into a successful pick.
        if pad_contacts == 0:
            return False
        joint_id = self.fruit_joint_ids[index]
        fruit = np.eye(4)
        fruit[:3, :3] = self.data.xmat[self.fruit_ids[index]].reshape(3, 3)
        fruit[:3, 3] = self.data.xpos[self.fruit_ids[index]]
        tcp_pose = np.eye(4)
        tcp_pose[:3, :3] = self.data.site_xmat[self.site_id].reshape(3, 3)
        tcp_pose[:3, 3] = tcp
        self._held_offset = np.linalg.inv(tcp_pose) @ fruit
        self._held_joint = joint_id
        return True

    def execute_pick(self, candidate: GraspCandidate) -> None:
        pose = self._graspnet_pose_to_tcp(candidate.pose_world)
        approach = pose[:3, 2]
        pregrasp = pose.copy()
        pregrasp[:3, 3] -= approach * 0.11
        lift = pose.copy()
        lift[:3, 3] += np.asarray([0, 0, 0.18])
        self._set_gripper(False)
        self._move_pose(pregrasp)
        self._move_pose(pose)
        self._set_gripper(True)
        if not self._attach_nearest():
            raise PipelineError(PipelineState.PICK, "gripper closed without acquiring a fruit")
        self._move_pose(lift)

    def execute_place(self, destination: BinName) -> None:
        # Use a canonical top-down release pose rather than carrying the arbitrary GraspNet
        # wrist roll into the bin approach, which can make an otherwise reachable position fail.
        release = np.eye(4)
        # TCP local +Z points down, local +Y is the finger closing axis.
        # Keep the closest collision-free wrist roll.  A fixed roll can force
        # a large wrist flip even though the same top-down position has many
        # equivalent solutions.
        current_rotation = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        best_rotation = None
        best_distance = float("inf")
        offsets = BIN_DROP_OFFSETS[destination]
        drop_offset = offsets[self._bin_drop_counts[destination] % len(offsets)]
        drop_position = BIN_CENTERS[destination].copy()
        drop_position[:2] += np.asarray(drop_offset)
        for degrees in range(-180, 181, 30):
            angle = np.deg2rad(degrees)
            rotation = np.asarray(
                [
                    [np.cos(angle), np.sin(angle), 0.0],
                    [np.sin(angle), -np.cos(angle), 0.0],
                    [0.0, 0.0, -1.0],
                ]
            )
            test = release.copy()
            test[:3, :3] = rotation
            test[:3, 3] = drop_position
            test[2, 3] = 0.35
            target_q = self._solve_from_current(test, max_iterations=260)
            if target_q is None:
                continue
            distance = float(
                np.linalg.norm(target_q - self.data.qpos[self.arm_qpos])
                + 0.05 * np.linalg.norm(rotation - current_rotation)
            )
            if distance < best_distance:
                best_distance, best_rotation = distance, rotation
        if best_rotation is None:
            raise PipelineError(PipelineState.PLACE, "no reachable top-down bin pose")
        release[:3, :3] = best_rotation
        release[:3, 3] = drop_position
        release[2, 3] = 0.20
        above = release.copy()
        above[2, 3] = 0.35
        self._move_pose(above, PipelineState.PLACE)
        self._move_pose(release, PipelineState.PLACE)
        self._set_gripper(False)
        self._held_joint = None
        self._bin_drop_counts[destination] += 1
        self._step(120, held=False)
        self._move_pose(above, PipelineState.PLACE)
        self.return_safe()

    def return_safe(self) -> None:
        start = self.data.ctrl[:6].copy()
        self._held_joint = None
        self.data.ctrl[6] = 0
        for alpha in np.linspace(0, 1, self.config.control_steps):
            self.data.ctrl[:6] = (1 - alpha) * start + alpha * HOME_Q
            self._step(1, held=False)

    def grasp_is_reachable(self, pose_world: np.ndarray) -> bool:
        planning = self._planning_data
        planning.qpos[:] = self.data.qpos
        planning.qvel[:] = self.data.qvel
        self.mj.mj_forward(self.model, planning)
        tcp_pose = self._graspnet_pose_to_tcp(pose_world)
        return self._ik(tcp_pose, max_iterations=100, data=planning) is not None

    def grasp_path_is_reachable(self, pose_world: np.ndarray) -> bool:
        """Check pregrasp, grasp and lift sequentially without changing simulation state."""
        planning = self._planning_data
        planning.qpos[:] = self.data.qpos
        planning.qvel[:] = self.data.qvel
        self.mj.mj_forward(self.model, planning)
        tcp_pose = self._graspnet_pose_to_tcp(pose_world)
        approach = tcp_pose[:3, 2]
        pregrasp = tcp_pose.copy()
        pregrasp[:3, 3] -= approach * 0.11
        lift = tcp_pose.copy()
        lift[:3, 3] += np.asarray([0.0, 0.0, 0.18])
        self.last_reachability_failure = ""
        for stage, target in (("pregrasp", pregrasp), ("grasp", tcp_pose), ("lift", lift)):
            target_q = self._ik(target, max_iterations=140, data=planning)
            if target_q is None:
                self.last_reachability_failure = stage
                return False
            planning.qpos[self.arm_qpos] = target_q
            self.mj.mj_forward(self.model, planning)
        return True

    def path_is_clear(self, pose_world: np.ndarray) -> bool:
        point = pose_world[:3, 3]
        return 0.035 < point[2] < 0.65 and -0.62 < point[1] < 0.62 and -0.1 < point[0] < 0.88

    @staticmethod
    def _graspnet_pose_to_tcp(grasp_pose: np.ndarray) -> np.ndarray:
        tcp_pose = grasp_pose.copy()
        tcp_pose[:3, :3] = grasp_pose[:3, :3] @ GRASPNET_TO_TCP
        return tcp_pose

    def close(self) -> None:
        self.renderer.close()
        if self.viewer is not None:
            self.viewer.close()
