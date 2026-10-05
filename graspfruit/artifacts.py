from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import GroundingDecision, Observation, PipelineState, SegmentationResult


class RunArtifacts:
    def __init__(self, root: Path, command: str, seed: int, record_video: bool = False):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = root / f"{stamp}-seed{seed}"
        self.path.mkdir(parents=True, exist_ok=False)
        self.events_path = self.path / "events.jsonl"
        self.summary_frames: list[np.ndarray] = []
        self._motion_writer = None
        self._presentation_writer = None
        self.motion_frame_count = 0
        self.presentation_frame_count = 0
        self.video_fps = 30
        self.record_video = record_video
        self.write_json("run.json", {"command": command, "seed": seed, "created_at": time.time()})

    def write_json(self, name: str, value: Any) -> None:
        def convert(obj: Any):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, Path):
                return str(obj)
            if hasattr(obj, "model_dump"):
                return obj.model_dump(mode="json")
            raise TypeError(type(obj).__name__)

        (self.path / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=convert), encoding="utf-8"
        )

    def event(self, state: PipelineState, attempt: int, message: str, **details: Any) -> None:
        payload = {
            "timestamp": time.time(), "state": state.value, "attempt": attempt,
            "message": message, "details": details,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def observation(self, name: str, observation: Observation) -> None:
        cv2.imwrite(str(self.path / f"{name}_rgb.png"), cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2BGR))
        np.save(self.path / f"{name}_depth.npy", observation.depth_m)
        if self.record_video:
            self.summary_frames.append(observation.rgb.copy())

    def motion_frame(self, rgb: np.ndarray) -> None:
        """Receive a frame sampled from MuJoCo simulation time."""
        if self.record_video:
            self._motion_writer = self._append_stream_frame(
                self._motion_writer, "replay.mp4", rgb
            )
            self.motion_frame_count += 1

    def presentation_motion_frame(self, rgb: np.ndarray) -> None:
        """Receive the synchronised front-view presentation frame."""
        if self.record_video:
            self._presentation_writer = self._append_stream_frame(
                self._presentation_writer, "presentation_replay.mp4", rgb
            )
            self.presentation_frame_count += 1

    def _append_stream_frame(self, writer, name: str, rgb: np.ndarray):
        if writer is None:
            import imageio.v2 as imageio

            writer = imageio.get_writer(
                self.path / name,
                fps=self.video_fps,
                codec="libx264",
                quality=8,
                macro_block_size=None,
            )
        writer.append_data(rgb)
        return writer

    def grounding(self, decision: GroundingDecision, raw: str | None = None) -> None:
        self.write_json("grounding.json", decision)
        if raw:
            (self.path / "vlm_raw.txt").write_text(raw, encoding="utf-8")

    def segmentation(self, name: str, result: SegmentationResult) -> None:
        cv2.imwrite(str(self.path / f"{name}_mask.png"), result.mask.astype(np.uint8) * 255)

    def overlay(
        self,
        name: str,
        observation: Observation,
        state: PipelineState,
        decision: GroundingDecision | None = None,
        segmentation: SegmentationResult | None = None,
    ) -> None:
        image = cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2BGR)
        if segmentation is not None:
            tint = np.zeros_like(image)
            tint[..., 1] = segmentation.mask.astype(np.uint8) * 180
            image = cv2.addWeighted(image, 1.0, tint, 0.4, 0)
        if decision is not None:
            x1, y1, x2, y2 = decision.bbox_pixels(image.shape[1], image.shape[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(image, f"{decision.target_name}->{decision.destination.value}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(image, state.value, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(self.path / f"{name}_overlay.png"), image)
        if self.record_video:
            self.summary_frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def finalize(self) -> None:
        if not self.record_video:
            return
        try:
            import imageio.v2 as imageio

            if self._motion_writer is not None:
                self._motion_writer.close()
                self._motion_writer = None
            if self._presentation_writer is not None:
                self._presentation_writer.close()
                self._presentation_writer = None
            # Keep the old sparse diagnostic images under an honest name.
            if self.summary_frames:
                with imageio.get_writer(
                    self.path / "summary.mp4", fps=4, codec="libx264"
                ) as writer:
                    for frame in self.summary_frames:
                        writer.append_data(frame)
        except Exception as exc:  # noqa: BLE001 - recording failure must not fail robot cleanup
            self.event(PipelineState.FAILED, 0, "video export failed", error=str(exc))

    def scope(self, prefix: str) -> ScopedRunArtifacts:
        """Namespace one item while sharing events and continuous video."""
        return ScopedRunArtifacts(self, prefix)


class ScopedRunArtifacts:
    """Per-item artifact names backed by one top-level sorting run."""

    def __init__(self, parent: RunArtifacts, prefix: str):
        self.parent = parent
        self.prefix = prefix
        self.path = parent.path

    def _name(self, name: str) -> str:
        return f"{self.prefix}_{name}"

    def write_json(self, name: str, value: Any) -> None:
        self.parent.write_json(self._name(name), value)

    def event(self, state: PipelineState, attempt: int, message: str, **details: Any) -> None:
        self.parent.event(state, attempt, message, item=self.prefix, **details)

    def observation(self, name: str, observation: Observation) -> None:
        self.parent.observation(self._name(name), observation)

    def grounding(self, decision: GroundingDecision, raw: str | None = None) -> None:
        self.parent.write_json(self._name("grounding.json"), decision)
        if raw:
            (self.path / self._name("vlm_raw.txt")).write_text(raw, encoding="utf-8")

    def segmentation(self, name: str, result: SegmentationResult) -> None:
        self.parent.segmentation(self._name(name), result)

    def overlay(self, name: str, *args, **kwargs) -> None:
        self.parent.overlay(self._name(name), *args, **kwargs)

    def finalize(self) -> None:
        pass
