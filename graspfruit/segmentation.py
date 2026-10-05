from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import SamConfig
from .models import Observation, PipelineError, PipelineState, SegmentationResult


def score_masks(
    masks: np.ndarray,
    model_scores: np.ndarray,
    bbox: tuple[int, int, int, int],
    depth_m: np.ndarray,
) -> list[tuple[float, int, float, int]]:
    x1, y1, x2, y2 = bbox
    box = np.zeros(depth_m.shape, dtype=bool)
    box[y1:y2, x1:x2] = True
    scores: list[tuple[float, int, float, int]] = []
    for index, raw in enumerate(masks):
        mask = raw.astype(bool)
        area = int(mask.sum())
        intersection = int((mask & box).sum())
        coverage = intersection / max(area, 1)
        valid_depth = int((mask & np.isfinite(depth_m) & (depth_m > 0)).sum())
        valid_ratio = valid_depth / max(area, 1)
        count, _ = cv2.connectedComponents(mask.astype(np.uint8))
        component_penalty = min(max(count - 2, 0), 4) * 0.08
        combined = 0.5 * float(model_scores[index]) + 0.3 * coverage + 0.2 * valid_ratio
        scores.append((combined - component_penalty, index, valid_ratio, valid_depth))
    return scores


def build_result(
    mask: np.ndarray, quality: float, depth_m: np.ndarray, min_pixels: int, min_ratio: float
) -> SegmentationResult:
    mask = mask.astype(bool)
    area = int(mask.sum())
    if area < min_pixels:
        raise PipelineError(PipelineState.SEGMENT, f"mask too small: {area} pixels")
    valid = mask & np.isfinite(depth_m) & (depth_m > 0)
    valid_points = int(valid.sum())
    valid_ratio = valid_points / area
    if valid_ratio < min_ratio:
        raise PipelineError(PipelineState.SEGMENT, f"valid depth ratio too low: {valid_ratio:.3f}")
    moments = cv2.moments(mask.astype(np.uint8))
    center = (round(moments["m10"] / moments["m00"]), round(moments["m01"] / moments["m00"]))
    return SegmentationResult(
        mask=mask,
        quality=float(np.clip(quality, 0, 1)),
        center_xy=center,
        valid_depth_points=valid_points,
        valid_depth_ratio=valid_ratio,
    )


@dataclass
class SAM2Segmenter:
    config: SamConfig
    device: str = "cuda"

    def __post_init__(self) -> None:
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise RuntimeError("SAM2 is not installed; run scripts/setup_windows.ps1") from exc
        self.predictor = SAM2ImagePredictor.from_pretrained(self.config.model_id, device=self.device)

    def segment(self, observation: Observation, bbox_xyxy: tuple[int, int, int, int]) -> SegmentationResult:
        import torch

        x1, y1, x2, y2 = bbox_xyxy
        padding = self.config.box_padding_pixels
        height, width = observation.depth_m.shape
        prompt_box = (
            max(0, x1 - padding), max(0, y1 - padding),
            min(width, x2 + padding), min(height, y2 + padding),
        )
        self.predictor.set_image(observation.rgb)
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            masks, scores, _ = self.predictor.predict(
                box=np.asarray(prompt_box, dtype=np.float32), multimask_output=True
            )
        ranked = score_masks(masks, scores, prompt_box, observation.depth_m)
        quality, index, _, _ = max(ranked, key=lambda item: item[0])
        return build_result(
            masks[index], quality, observation.depth_m,
            self.config.min_mask_pixels, self.config.min_valid_depth_ratio,
        )

    def track(
        self, before: Observation, after: Observation, initial: SegmentationResult
    ) -> SegmentationResult | None:
        # Sparse optical flow proposes a point in the new keyframe; SAM2 then refines the object.
        # This avoids retaining a full video state while GraspNet occupies the 8 GB GPU.
        ys, xs = np.nonzero(initial.mask)
        if len(xs) < 8:
            return None
        stride = max(1, len(xs) // 80)
        points = np.c_[xs[::stride], ys[::stride]].astype(np.float32).reshape(-1, 1, 2)
        before_gray = cv2.cvtColor(before.rgb, cv2.COLOR_RGB2GRAY)
        after_gray = cv2.cvtColor(after.rgb, cv2.COLOR_RGB2GRAY)
        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            before_gray, after_gray, points, None, winSize=(31, 31), maxLevel=4
        )
        if moved is None or status is None or int(status.sum()) < 5:
            return None
        point = np.median(moved[status.ravel() == 1].reshape(-1, 2), axis=0)
        height, width = after.depth_m.shape
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            return None
        try:
            import torch

            self.predictor.set_image(after.rgb)
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                masks, scores, _ = self.predictor.predict(
                    point_coords=np.asarray([point], dtype=np.float32),
                    point_labels=np.asarray([1], dtype=np.int32),
                    multimask_output=True,
                )
            px, py = int(point[0]), int(point[1])
            candidates = [i for i, mask in enumerate(masks) if mask[py, px]]
            if not candidates:
                return None
            index = max(candidates, key=lambda i: float(scores[i]))
            return build_result(
                masks[index], float(scores[index]), after.depth_m,
                self.config.min_mask_pixels, max(0.25, self.config.min_valid_depth_ratio / 2),
            )
        except (PipelineError, RuntimeError, ValueError):
            return None


class BoxSegmenter:
    """Development-only depth/color region segmenter."""

    def __init__(self, config: SamConfig):
        self.config = config

    def segment(self, observation: Observation, bbox_xyxy: tuple[int, int, int, int]) -> SegmentationResult:
        x1, y1, x2, y2 = bbox_xyxy
        mask = np.zeros(observation.depth_m.shape, dtype=bool)
        crop = observation.depth_m[y1:y2, x1:x2]
        valid = crop[np.isfinite(crop) & (crop > 0)]
        if valid.size:
            median = float(np.median(valid))
            mask[y1:y2, x1:x2] = np.abs(crop - median) < 0.08
        else:
            mask[y1:y2, x1:x2] = True
        return build_result(mask, 0.5, observation.depth_m, 20, 0.1)

    def track(self, before, after, initial):
        pixels = before.rgb[initial.mask]
        if len(pixels) == 0:
            return None
        reference = np.median(pixels.astype(float), axis=0)
        distance = np.linalg.norm(after.rgb.astype(float) - reference, axis=2)
        raw = (distance < 80).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
        components = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 20]
        if not components:
            return None
        index = max(components, key=lambda i: stats[i, cv2.CC_STAT_AREA])
        mask = labels == index
        try:
            return build_result(mask, 0.45, after.depth_m, 20, 0.1)
        except PipelineError:
            return None
