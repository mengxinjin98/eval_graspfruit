from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .models import (
    BinName,
    GraspCandidate,
    GroundingDecision,
    Observation,
    SegmentationResult,
    VerificationEvidence,
)


class VLMProvider(Protocol):
    def ground(self, command: str, rgb: np.ndarray) -> GroundingDecision: ...


class Segmenter(Protocol):
    def segment(
        self, observation: Observation, bbox_xyxy: tuple[int, int, int, int]
    ) -> SegmentationResult: ...

    def track(
        self, before: Observation, after: Observation, initial: SegmentationResult
    ) -> SegmentationResult | None: ...


class GraspPlanner(Protocol):
    def plan(
        self,
        observation: Observation,
        segmentation: SegmentationResult,
        excluded_poses: Sequence[np.ndarray],
    ) -> list[GraspCandidate]: ...


class RobotEnv(Protocol):
    def reset(self, seed: int) -> None: ...

    def observe(self) -> Observation: ...

    def execute_pick(self, candidate: GraspCandidate) -> None: ...

    def execute_place(self, destination: BinName) -> None: ...

    def return_safe(self) -> None: ...

    def grasp_is_reachable(self, pose_world: np.ndarray) -> bool: ...

    def grasp_path_is_reachable(self, pose_world: np.ndarray) -> bool: ...

    def path_is_clear(self, pose_world: np.ndarray) -> bool: ...

    def close(self) -> None: ...


class Verifier(Protocol):
    def verify_pick(
        self,
        before: Observation,
        after: Observation,
        initial: SegmentationResult,
        tracked: SegmentationResult | None,
    ) -> VerificationEvidence: ...

    def verify_place(
        self,
        before: Observation,
        observation: Observation,
        destination: BinName,
        tracked: SegmentationResult | None,
    ) -> VerificationEvidence: ...
