from __future__ import annotations

import numpy as np


def depth_to_points(depth_m: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Back-project an HxW metric depth image into an organized HxWx3 point cloud."""
    height, width = depth_m.shape
    ys, xs = np.indices((height, width), dtype=np.float32)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    z = depth_m.astype(np.float32, copy=False)
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack((x, y, z), axis=-1)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    original_shape = points.shape
    flat = points.reshape(-1, 3)
    homogeneous = np.concatenate((flat, np.ones((flat.shape[0], 1))), axis=1)
    result = (transform @ homogeneous.T).T[:, :3]
    return result.reshape(original_shape)


def pose_from_translation_rotation(translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def project_point(point_camera: np.ndarray, intrinsics: np.ndarray) -> tuple[float, float] | None:
    if point_camera[2] <= 0:
        return None
    uvw = intrinsics @ point_camera
    return float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2])


def rotation_angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    delta = a[:3, :3].T @ b[:3, :3]
    value = np.clip((np.trace(delta) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))

