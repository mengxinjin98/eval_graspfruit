from __future__ import annotations

import cv2
import numpy as np

from .geometry import project_point
from .models import BinName, VerificationEvidence

BIN_WORLD_CENTERS = {
    BinName.BLUE: np.asarray([0.27, -0.50, 0.095]),
    BinName.GREEN: np.asarray([0.27, 0.50, 0.095]),
}

BIN_HALF_FOOTPRINTS = {
    BinName.BLUE: (0.12, 0.10),
    BinName.GREEN: (0.16, 0.13),
}


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 0.0


class VisualVerifier:
    def verify_pick(self, before, after, initial, tracked):
        gray_before = cv2.cvtColor(before.rgb, cv2.COLOR_RGB2GRAY)
        gray_after = cv2.cvtColor(after.rgb, cv2.COLOR_RGB2GRAY)
        change = cv2.absdiff(gray_before, gray_after)
        changed_ratio = float((change[initial.mask] > 20).mean()) if initial.mask.any() else 0.0
        rgb_distance = np.linalg.norm(
            before.rgb.astype(np.float32) - after.rgb.astype(np.float32), axis=2
        )
        rgb_changed_ratio = (
            float((rgb_distance[initial.mask] > 40).mean()) if initial.mask.any() else 0.0
        )
        tracked_moved = False
        movement = 0.0
        object_radius = float(np.sqrt(max(int(initial.mask.sum()), 1) / np.pi))
        movement_threshold = max(5.0, 0.60 * object_radius)
        if tracked is not None:
            movement = float(np.linalg.norm(np.asarray(tracked.center_xy) - np.asarray(initial.center_xy)))
            tracked_moved = movement > movement_threshold
        # Optical flow may lock onto newly exposed table texture after the
        # fruit is lifted and falsely report a stationary target. Accept the
        # independent disappearance evidence only when both grayscale and
        # full-RGB changes cover most of the original object mask.
        region_cleared = changed_ratio > 0.35 and rgb_changed_ratio > 0.55
        success = changed_ratio > 0.35 and (
            tracked is None or tracked_moved or region_cleared
        )
        confidence = min(
            1.0,
            0.45 * changed_ratio
            + 0.35 * rgb_changed_ratio
            + (0.20 if tracked_moved or region_cleared else 0.05),
        )
        return VerificationEvidence(
            success=success,
            confidence=confidence,
            reason="target left its original image region" if success else "target appears to remain at pickup",
            measurements={
                "changed_ratio": changed_ratio,
                "rgb_changed_ratio": rgb_changed_ratio,
                "original_region_cleared": region_cleared,
                "tracked_motion_px": movement,
                "movement_threshold_px": movement_threshold,
                "initial_object_radius_px": object_radius,
            },
        )

    def verify_place(self, before, observation, destination, tracked):
        # Fixed-camera depth differencing inside the physical bin footprint is
        # robust to optical-flow loss during a
        # large pick-to-place motion.  It remains a runtime visual test: no
        # fruit identity or MuJoCo body pose is consulted.
        center = BIN_WORLD_CENTERS[destination]
        half_x, half_y = BIN_HALF_FOOTPRINTS[destination]
        world_to_camera = np.linalg.inv(observation.camera_to_world)
        corners = []
        for dx, dy in ((-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y)):
            corner_cam = (world_to_camera @ np.r_[center + [dx, dy, 0.0], 1])[:3]
            projected = project_point(corner_cam, observation.intrinsics)
            if projected is None:
                return VerificationEvidence(
                    success=False, confidence=0, reason="destination is outside camera"
                )
            corners.append(projected)
        bin_mask = np.zeros(observation.depth_m.shape, dtype=np.uint8)
        cv2.fillConvexPoly(bin_mask, np.round(corners).astype(np.int32), 1)
        valid = (
            bin_mask.astype(bool)
            & np.isfinite(before.depth_m)
            & np.isfinite(observation.depth_m)
            & (before.depth_m > 0)
            & (observation.depth_m > 0)
        )
        changed = valid & (np.abs(before.depth_m - observation.depth_m) > 0.01)
        count, _, stats, _ = cv2.connectedComponentsWithStats(changed.astype(np.uint8), 8)
        largest_change = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
        depth_success = largest_change >= 45
        if depth_success:
            return VerificationEvidence(
                success=True,
                confidence=float(np.clip(0.55 + largest_change / 800, 0, 1)),
                reason="new object is stationary inside destination bin",
                measurements={
                    "bin_depth_change_pixels": int(changed.sum()),
                    "largest_bin_change_component": largest_change,
                },
            )
        if tracked is None:
            return VerificationEvidence(
                success=False, confidence=0.15, reason="target could not be tracked after release"
            )
        center_cam = (world_to_camera @ np.r_[center, 1])[:3]
        projected = project_point(center_cam, observation.intrinsics)
        if projected is None:
            return VerificationEvidence(success=False, confidence=0, reason="destination is outside camera")
        distance = float(np.linalg.norm(np.asarray(tracked.center_xy) - np.asarray(projected)))
        success = distance < 105
        return VerificationEvidence(
            success=success,
            confidence=float(np.clip(1 - distance / 180, 0, 1)),
            reason="target is inside destination image region" if success else "target is outside destination",
            measurements={
                "distance_to_bin_px": distance,
                "largest_bin_change_component": largest_change,
            },
        )
