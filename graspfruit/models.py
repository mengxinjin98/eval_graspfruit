from __future__ import annotations

from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BinName(StrEnum):
    BLUE = "blue_bin"
    GREEN = "green_bin"


class PipelineState(StrEnum):
    OBSERVE = "OBSERVE"
    GROUND = "GROUND"
    SEGMENT = "SEGMENT"
    PLAN_GRASP = "PLAN_GRASP"
    PICK = "PICK"
    VERIFY_PICK = "VERIFY_PICK"
    PLACE = "PLACE"
    VERIFY_PLACE = "VERIFY_PLACE"
    RETRY = "RETRY"
    DONE = "DONE"
    FAILED = "FAILED"


class Observation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    timestamp: float

    @model_validator(mode="after")
    def validate_shapes(self) -> Observation:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError("rgb must have shape HxWx3")
        if self.depth_m.shape != self.rgb.shape[:2]:
            raise ValueError("depth and rgb resolutions differ")
        if self.intrinsics.shape != (3, 3):
            raise ValueError("intrinsics must be 3x3")
        if self.camera_to_world.shape != (4, 4):
            raise ValueError("camera_to_world must be 4x4")
        return self


class GroundingDecision(BaseModel):
    target_name: str = Field(min_length=1)
    attributes: list[str] = Field(default_factory=list)
    relation: str = ""
    reasoning: str = Field(min_length=1)
    bbox_norm: tuple[float, float, float, float]
    destination: BinName
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_class_route(self) -> GroundingDecision:
        routes = {
            "apple": BinName.BLUE,
            "banana": BinName.GREEN,
            "pear": BinName.GREEN,
        }
        normalized = self.target_name.strip().lower()
        if normalized not in routes:
            raise ValueError(f"unsupported fruit class: {self.target_name}")
        if self.destination != routes[normalized]:
            raise ValueError(
                f"invalid route: {normalized} must go to {routes[normalized].value}"
            )
        self.target_name = normalized
        return self

    @field_validator("bbox_norm")
    @classmethod
    def validate_bbox(cls, value: tuple[float, float, float, float]):
        x1, y1, x2, y2 = value
        if not all(0.0 <= v <= 1.0 for v in value):
            raise ValueError("normalized bbox values must be in [0, 1]")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must have positive area")
        return value

    def bbox_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.bbox_norm
        return (
            max(0, min(width - 1, round(x1 * width))),
            max(0, min(height - 1, round(y1 * height))),
            max(1, min(width, round(x2 * width))),
            max(1, min(height, round(y2 * height))),
        )


class SegmentationResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mask: np.ndarray
    quality: float = Field(ge=0.0, le=1.0)
    center_xy: tuple[int, int]
    valid_depth_points: int = Field(ge=0)
    valid_depth_ratio: float = Field(ge=0.0, le=1.0)

    @field_validator("mask")
    @classmethod
    def validate_mask(cls, value: np.ndarray) -> np.ndarray:
        if value.ndim != 2:
            raise ValueError("mask must be two-dimensional")
        return value.astype(bool, copy=False)


class GraspCandidate(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pose_camera: np.ndarray
    pose_world: np.ndarray
    width: float = Field(gt=0)
    score: float
    source_index: int = Field(ge=0)
    filters_passed: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_poses(self) -> GraspCandidate:
        if self.pose_camera.shape != (4, 4) or self.pose_world.shape != (4, 4):
            raise ValueError("grasp poses must be 4x4")
        return self


class VerificationEvidence(BaseModel):
    success: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    measurements: dict[str, float | int | str | bool] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    status: PipelineState
    command: str
    attempts: int = Field(ge=0)
    failure_stage: PipelineState | None = None
    reason: str = ""
    evidence: list[VerificationEvidence] = Field(default_factory=list)
    run_directory: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class PipelineError(RuntimeError):
    def __init__(self, stage: PipelineState, message: str, retryable: bool = True):
        super().__init__(message)
        self.stage = stage
        self.retryable = retryable
