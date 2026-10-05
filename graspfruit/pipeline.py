from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .artifacts import RunArtifacts
from .models import ExecutionResult, PipelineError, PipelineState, VerificationEvidence
from .protocols import GraspPlanner, RobotEnv, Segmenter, Verifier, VLMProvider


@dataclass
class ClosedLoopPipeline:
    vlm: VLMProvider
    segmenter: Segmenter
    planner: GraspPlanner
    robot: RobotEnv
    verifier: Verifier
    artifacts: RunArtifacts
    max_attempts: int = 3
    finalize_artifacts: bool = True

    def run(self, command: str) -> ExecutionResult:
        started = time.perf_counter()
        evidence: list[VerificationEvidence] = []
        excluded: list[np.ndarray] = []
        last_stage: PipelineState | None = None
        last_reason = ""
        attempts_used = 0
        decision = None
        robot_moved = False
        try:
            for attempt in range(1, self.max_attempts + 1):
                attempts_used = attempt
                selected = None
                try:
                    self._event(PipelineState.OBSERVE, attempt, "capturing RGB-D")
                    stage_started = time.perf_counter()
                    before = self.robot.observe()
                    self.artifacts.observation(f"attempt_{attempt}_before", before)
                    self._timing(PipelineState.OBSERVE, attempt, stage_started)

                    if decision is None:
                        self._event(PipelineState.GROUND, attempt, "grounding command")
                        stage_started = time.perf_counter()
                        decision = self.vlm.ground(command, before.rgb)
                        raw = getattr(self.vlm, "last_raw_response", None)
                        self.artifacts.grounding(decision, raw)
                        self._timing(PipelineState.GROUND, attempt, stage_started)
                    else:
                        self._event(
                            PipelineState.GROUND,
                            attempt,
                            "reusing target grounding; refreshing RGB-D and mask",
                        )

                    self._event(PipelineState.SEGMENT, attempt, "segmenting selected target")
                    stage_started = time.perf_counter()
                    bbox = decision.bbox_pixels(before.rgb.shape[1], before.rgb.shape[0])
                    segmentation = self.segmenter.segment(before, bbox)
                    self.artifacts.segmentation(f"attempt_{attempt}", segmentation)
                    self.artifacts.overlay(
                        f"attempt_{attempt}", before, PipelineState.SEGMENT, decision, segmentation
                    )
                    self._timing(PipelineState.SEGMENT, attempt, stage_started)

                    self._event(PipelineState.PLAN_GRASP, attempt, "planning filtered 6D grasps")
                    stage_started = time.perf_counter()
                    candidates = self.planner.plan(before, segmentation, excluded)
                    self.artifacts.write_json(
                        f"attempt_{attempt}_grasps.json",
                        [candidate.model_dump() for candidate in candidates],
                    )
                    self._timing(PipelineState.PLAN_GRASP, attempt, stage_started)
                    selected = candidates[0]

                    self._event(PipelineState.PICK, attempt, "executing pregrasp-descend-close-lift")
                    robot_moved = True
                    self.robot.execute_pick(selected)
                    after_pick = self.robot.observe()
                    self.artifacts.observation(f"attempt_{attempt}_after_pick", after_pick)

                    self._event(PipelineState.VERIFY_PICK, attempt, "visual pick verification")
                    pick_track = self.segmenter.track(before, after_pick, segmentation)
                    pick_evidence = self.verifier.verify_pick(
                        before, after_pick, segmentation, pick_track
                    )
                    evidence.append(pick_evidence)
                    if not pick_evidence.success:
                        raise PipelineError(
                            PipelineState.VERIFY_PICK, pick_evidence.reason, retryable=True
                        )

                    self._event(PipelineState.PLACE, attempt, f"placing into {decision.destination.value}")
                    self.robot.execute_place(decision.destination)
                    after_place = self.robot.observe()
                    self.artifacts.observation(f"attempt_{attempt}_after_place", after_place)

                    self._event(PipelineState.VERIFY_PLACE, attempt, "visual place verification")
                    place_track = self.segmenter.track(after_pick, after_place, pick_track or segmentation)
                    place_evidence = self.verifier.verify_place(
                        after_pick, after_place, decision.destination, place_track
                    )
                    evidence.append(place_evidence)
                    if not place_evidence.success:
                        raise PipelineError(
                            PipelineState.VERIFY_PLACE, place_evidence.reason, retryable=False
                        )

                    elapsed = time.perf_counter() - started
                    self._event(PipelineState.DONE, attempt, "task completed")
                    result = ExecutionResult(
                        status=PipelineState.DONE,
                        command=command,
                        attempts=attempt,
                        evidence=evidence,
                        run_directory=str(self.artifacts.path),
                        metrics={"elapsed_seconds": elapsed, "mock_mode": self._mock_mode()},
                    )
                    self.artifacts.write_json("result.json", result)
                    return result
                except PipelineError as exc:
                    last_stage, last_reason = exc.stage, str(exc)
                    if exc.stage in (PipelineState.PICK, PipelineState.VERIFY_PICK):
                        if selected is not None:
                            excluded.append(selected.pose_world.copy())
                        # Execution may have disturbed the object, so a cached
                        # image-space box is no longer safe after robot motion.
                        decision = None
                    self._event(exc.stage, attempt, "attempt failed", error=str(exc))
                    if robot_moved:
                        self.robot.return_safe()
                        robot_moved = False
                    if not exc.retryable or attempt == self.max_attempts:
                        break
                    self._event(PipelineState.RETRY, attempt, "re-observing for next attempt")
            elapsed = time.perf_counter() - started
            result = ExecutionResult(
                status=PipelineState.FAILED,
                command=command,
                attempts=attempts_used,
                failure_stage=last_stage,
                reason=last_reason or "retry limit reached",
                evidence=evidence,
                run_directory=str(self.artifacts.path),
                metrics={"elapsed_seconds": elapsed, "mock_mode": self._mock_mode()},
            )
            self.artifacts.write_json("result.json", result)
            return result
        except Exception as exc:  # noqa: BLE001 - top-level robot safety boundary
            if robot_moved:
                try:
                    self.robot.return_safe()
                except Exception as safe_exc:  # noqa: BLE001 - preserve the original failure
                    self.artifacts.event(
                        PipelineState.FAILED,
                        attempts_used,
                        "safe return also failed",
                        error=str(safe_exc),
                    )
            result = ExecutionResult(
                status=PipelineState.FAILED,
                command=command,
                attempts=attempts_used,
                failure_stage=last_stage or PipelineState.FAILED,
                reason=f"unexpected error: {exc}",
                evidence=evidence,
                run_directory=str(self.artifacts.path),
                metrics={"elapsed_seconds": time.perf_counter() - started, "mock_mode": self._mock_mode()},
            )
            self.artifacts.write_json("result.json", result)
            return result
        finally:
            if self.finalize_artifacts:
                self.artifacts.finalize()

    def _event(self, state: PipelineState, attempt: int, message: str, **details) -> None:
        print(f"[{state.value}] {message}")
        self.artifacts.event(state, attempt, message, **details)

    def _timing(self, state: PipelineState, attempt: int, started: float) -> None:
        elapsed = time.perf_counter() - started
        self._event(state, attempt, f"completed in {elapsed:.2f}s", elapsed_seconds=elapsed)

    def _mock_mode(self) -> bool:
        return any(type(component).__name__.startswith(("Heuristic", "Box", "Centroid")) for component in (self.vlm, self.segmenter, self.planner))
