from __future__ import annotations

import importlib
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from .config import GraspConfig
from .geometry import depth_to_points, pose_from_translation_rotation, rotation_angle_degrees
from .models import GraspCandidate, Observation, PipelineError, PipelineState, SegmentationResult
from .protocols import RobotEnv


def sample_points(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) == 0:
        raise ValueError("cannot sample an empty point cloud")
    indices = rng.choice(len(points), count, replace=len(points) < count)
    return points[indices].astype(np.float32, copy=False)


def eroded_mask(mask: np.ndarray, pixels: int = 3) -> np.ndarray:
    kernel = np.ones((pixels, pixels), np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def candidate_is_excluded(
    pose: np.ndarray,
    excluded: Sequence[np.ndarray],
    translation_threshold: float = 0.035,
    angle_threshold_deg: float = 20,
) -> bool:
    for previous in excluded:
        distance = float(np.linalg.norm(pose[:3, 3] - previous[:3, 3]))
        if distance < translation_threshold and rotation_angle_degrees(pose, previous) < angle_threshold_deg:
            return True
    return False


@dataclass
class GraspNetPlanner:
    config: GraspConfig
    robot: RobotEnv
    seed: int = 7

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        root = self.config.third_party_root.resolve()
        for folder in (root / "models", root / "dataset", root / "utils", root / "pointnet2"):
            if str(folder) not in sys.path:
                sys.path.insert(0, str(folder))
        if not root.exists():
            raise RuntimeError("GraspNet source is missing; run scripts/setup_windows.ps1")
        if not self.config.checkpoint.exists():
            raise RuntimeError(f"GraspNet checkpoint not found: {self.config.checkpoint}")
        self._load_model()

    def _load_model(self) -> None:
        import torch

        try:
            module = importlib.import_module("graspnet")
            self.GraspNet = module.GraspNet
            self.pred_decode = module.pred_decode
            self.GraspGroup = importlib.import_module("graspnetAPI").GraspGroup
            self.ModelFreeCollisionDetector = importlib.import_module(
                "collision_detector"
            ).ModelFreeCollisionDetector
        except ImportError as exc:
            raise RuntimeError(
                f"GraspNet/GraspNetAPI imports failed: {type(exc).__name__}: {exc}"
            ) from exc
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = self.GraspNet(
            input_feature_dim=0,
            num_view=300,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        ).to(self.device)
        checkpoint = torch.load(self.config.checkpoint, map_location=self.device, weights_only=False)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.net.eval()

    def plan(
        self,
        observation: Observation,
        segmentation: SegmentationResult,
        excluded_poses: Sequence[np.ndarray],
    ) -> list[GraspCandidate]:
        import torch

        organized = depth_to_points(observation.depth_m, observation.intrinsics)
        valid = np.isfinite(observation.depth_m) & (observation.depth_m > 0.05) & (observation.depth_m < 2.5)
        target_mask = valid & segmentation.mask
        target_points = organized[target_mask]
        if len(target_points) < 100:
            raise PipelineError(PipelineState.PLAN_GRASP, "too few target point-cloud samples")
        sampled = sample_points(target_points, self.config.num_points, self.rng)
        end_points = {"point_clouds": torch.from_numpy(sampled[None]).to(self.device)}
        with torch.inference_mode():
            predictions = self.pred_decode(self.net(end_points))[0].detach().cpu().numpy()
        group = self.GraspGroup(predictions)
        filter_counts = {"network": len(group)}
        full_cloud = organized[valid]
        detector = self.ModelFreeCollisionDetector(full_cloud, voxel_size=0.01)
        collisions = detector.detect(
            group, approach_dist=0.05, collision_thresh=self.config.collision_threshold
        )
        group = group[~collisions]
        filter_counts["collision"] = len(group)
        group = group.nms()
        filter_counts["nms"] = len(group)
        group.sort_by_score()

        inner = eroded_mask(segmentation.mask, 5)
        world_down = np.asarray([0.0, 0.0, -1.0])
        candidates: list[GraspCandidate] = []
        filter_counts.update(
            width=0, mask=0, approach=0, excluded=0,
            ik_pregrasp_fail=0, ik_grasp_fail=0, ik_lift_fail=0, ik_path=0, path=0,
        )
        world_to_camera = np.linalg.inv(observation.camera_to_world)
        for source_index, grasp in enumerate(group):
            camera_pose = pose_from_translation_rotation(grasp.translation, grasp.rotation_matrix)
            world_pose = observation.camera_to_world @ camera_pose
            width = float(grasp.width)
            if not (
                self.config.min_width
                <= width
                <= self.config.max_width + self.config.network_width_tolerance
            ):
                continue
            filter_counts["width"] += 1
            center = grasp.translation
            if center[2] <= 0:
                continue
            uv = observation.intrinsics @ center
            u, v = round(uv[0] / uv[2]), round(uv[1] / uv[2])
            if not (0 <= v < inner.shape[0] and 0 <= u < inner.shape[1] and inner[v, u]):
                continue
            filter_counts["mask"] += 1
            approach = world_pose[:3, 0]
            angle = math.degrees(math.acos(np.clip(np.dot(approach, world_down), -1, 1)))
            if angle > self.config.approach_angle_deg:
                continue
            filter_counts["approach"] += 1
            if candidate_is_excluded(world_pose, excluded_poses):
                continue
            filter_counts["excluded"] += 1
            path_reachable = getattr(self.robot, "grasp_path_is_reachable", None)
            reachable = (
                path_reachable(world_pose)
                if path_reachable is not None
                else self.robot.grasp_is_reachable(world_pose)
            )
            if not reachable:
                failed_stage = getattr(self.robot, "last_reachability_failure", "")
                key = f"ik_{failed_stage}_fail"
                if key in filter_counts:
                    filter_counts[key] += 1
                continue
            filter_counts["ik_path"] += 1
            if not self.robot.path_is_clear(world_pose):
                continue
            filter_counts["path"] += 1
            # Store the exact camera pose reconstructed through the inverse transform to guard
            # against accidental convention drift in adapters.
            camera_pose = world_to_camera @ world_pose
            candidates.append(
                GraspCandidate(
                    pose_camera=camera_pose,
                    pose_world=world_pose,
                    width=min(width, self.config.max_width),
                    score=float(grasp.score),
                    source_index=source_index,
                    filters_passed=[
                        "collision", "mask", "width_with_network_tolerance",
                        "approach", "ik_path", "path",
                    ],
                )
            )
            if len(candidates) >= self.config.max_candidates:
                break
        # The decision camera is deliberately overhead, so the most reliable
        # grasp is the RGB-D centre of the segmented instance with a vertical
        # approach.  GraspNet candidates remain valuable alternatives for
        # unusual poses, but its highest-scoring surface pose can sit near a
        # curved fruit edge and let the fingers close above the object.
        top_down = self._top_down_fallback(
            observation, organized, target_mask, excluded_poses, len(group)
        )
        candidates = top_down + candidates
        if not candidates:
            summary = ", ".join(f"{name}={count}" for name, count in filter_counts.items())
            raise PipelineError(
                PipelineState.PLAN_GRASP,
                f"no collision-free reachable grasps ({summary}; top_down_fallback=0)",
            )
        target_center_world = (
            observation.camera_to_world @ np.r_[np.median(target_points, axis=0), 1.0]
        )[:3]
        # Network score alone can favour a high-confidence surface-edge grasp.
        # Prefer candidates close to the SAM RGB-D centre while retaining the
        # score as a secondary term; this reduces gripper closure near misses.
        return sorted(
            candidates,
            key=lambda candidate: (
                0 if "rgbd_top_down_primary" in candidate.filters_passed else 1,
                float(np.linalg.norm(candidate.pose_world[:3, 3] - target_center_world))
                - 0.015 * candidate.score
            ),
        )

    def _top_down_fallback(
        self,
        observation: Observation,
        organized_points_camera: np.ndarray,
        target_mask: np.ndarray,
        excluded_poses: Sequence[np.ndarray],
        source_index: int,
    ) -> list[GraspCandidate]:
        """Build a perception-only top-down candidate after GraspNet exhaustion."""
        # A curved banana's point-cloud median can lie in the empty space
        # inside its arc. Select the mask's most interior pixel instead.
        distance = cv2.distanceTransform(target_mask.astype(np.uint8), cv2.DIST_L2, 5)
        center_v, center_u = np.unravel_index(np.argmax(distance), distance.shape)
        center_camera = organized_points_camera[center_v, center_u]
        if not np.isfinite(center_camera).all() or center_camera[2] <= 0:
            center_camera = np.median(organized_points_camera[target_mask], axis=0)
        center_world = (observation.camera_to_world @ np.r_[center_camera, 1.0])[:3]
        # RGB-D observes the upper surface, while the Robotiq pinch site must
        # descend into the fruit's mid-section before closing.  This offset is
        # relative to the perceived surface and does not use simulator truth.
        center_world[2] = max(0.065, center_world[2] - 0.025)
        base_rotation = np.asarray(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],
            dtype=float,
        )
        world_to_camera = np.linalg.inv(observation.camera_to_world)
        for yaw_deg in (0, -30, 30, -60, 60, 90):
            yaw = math.radians(yaw_deg)
            yaw_rotation = np.asarray(
                [
                    [math.cos(yaw), -math.sin(yaw), 0.0],
                    [math.sin(yaw), math.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            world_pose = np.eye(4)
            world_pose[:3, :3] = yaw_rotation @ base_rotation
            world_pose[:3, 3] = center_world
            if candidate_is_excluded(world_pose, excluded_poses):
                continue
            path_reachable = getattr(self.robot, "grasp_path_is_reachable", None)
            reachable = (
                path_reachable(world_pose)
                if path_reachable is not None
                else self.robot.grasp_is_reachable(world_pose)
            )
            if not reachable or not self.robot.path_is_clear(world_pose):
                continue
            return [
                GraspCandidate(
                    pose_camera=world_to_camera @ world_pose,
                    pose_world=world_pose,
                    width=0.06,
                    score=-1.0,
                    source_index=source_index,
                    filters_passed=[
                        "rgbd_top_down_primary",
                        "ik_path",
                        "path",
                    ],
                )
            ]
        return []


class CentroidGraspPlanner:
    """Development-only planner that produces a top-down grasp at the mask centroid."""

    def __init__(self, robot: RobotEnv):
        self.robot = robot

    def plan(self, observation, segmentation, excluded_poses):
        points = depth_to_points(observation.depth_m, observation.intrinsics)[segmentation.mask]
        points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0)]
        if len(points) == 0:
            raise PipelineError(PipelineState.PLAN_GRASP, "mock planner has no valid points")
        center = np.median(points, axis=0)
        # GraspNet convention: local x is approach. Point it along world down.
        rotation_world = np.asarray([[0, 1, 0], [0, 0, 1], [-1, 0, 0]], dtype=float)
        world_pose = np.eye(4)
        world_pose[:3, :3] = rotation_world
        world_pose[:3, 3] = (observation.camera_to_world @ np.r_[center, 1])[:3]
        if candidate_is_excluded(world_pose, excluded_poses):
            world_pose[0, 3] += 0.02
        camera_pose = np.linalg.inv(observation.camera_to_world) @ world_pose
        return [
            GraspCandidate(
                pose_camera=camera_pose,
                pose_world=world_pose,
                width=0.06,
                score=0.5,
                source_index=0,
                filters_passed=["development_heuristic"],
            )
        ]
