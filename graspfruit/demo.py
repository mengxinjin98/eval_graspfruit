from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifacts import RunArtifacts
from .config import load_config
from .grasping import CentroidGraspPlanner, GraspNetPlanner
from .models import PipelineState
from .pipeline import ClosedLoopPipeline
from .segmentation import BoxSegmenter, SAM2Segmenter
from .simulation import MujocoFruitEnv
from .sorting import SortAllCoordinator
from .verification import VisualVerifier
from .vlm import HeuristicVLM, OpenAICompatibleVLM


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Closed-loop VLM fruit sorting in MuJoCo")
    result.add_argument("--command", help="Chinese natural-language pick-and-place instruction")
    result.add_argument(
        "--sort-all",
        action="store_true",
        help="sort all nine fruit: apples to blue, bananas and pears to green",
    )
    result.add_argument("--seed", type=int, default=7)
    result.add_argument("--config", type=Path)
    result.add_argument("--headless", action="store_true")
    result.add_argument("--record", action="store_true")
    result.add_argument(
        "--mock-perception",
        action="store_true",
        help="explicit development mode without VLM, SAM2 or GraspNet; results are not experimental",
    )
    return result


def build_pipeline(args: argparse.Namespace):
    config = load_config(
        args.config,
        {
            "runtime": {
                "seed": args.seed,
                "headless": args.headless,
                "record": args.record,
            }
        },
    )
    if args.sort_all and args.command:
        raise SystemExit("--sort-all 与 --command 不能同时使用")
    command = "分拣桌面上的全部水果" if args.sort_all else args.command or input("请输入抓放指令：").strip()
    if not command:
        raise SystemExit("指令不能为空")
    artifacts = RunArtifacts(config.runtime.run_root, command, args.seed, args.record)
    robot = MujocoFruitEnv(config.simulation, headless=config.runtime.headless)
    try:
        robot.reset(args.seed)
        if args.record:
            robot.enable_recording(
                artifacts.motion_frame,
                fps=artifacts.video_fps,
                presentation_callback=artifacts.presentation_motion_frame,
            )
        if args.mock_perception:
            print("WARNING: --mock-perception is development-only and is excluded from experiment metrics.")
            vlm = HeuristicVLM()
            segmenter = BoxSegmenter(config.sam)
            planner = CentroidGraspPlanner(robot)
        else:
            vlm = OpenAICompatibleVLM(config.vlm)
            segmenter = SAM2Segmenter(config.sam)
            planner = GraspNetPlanner(config.grasp, robot, seed=args.seed)
    except Exception:
        robot.close()
        raise
    def make_pipeline(run_artifacts, _prefix=""):
        return ClosedLoopPipeline(
            vlm=vlm,
            segmenter=segmenter,
            planner=planner,
            robot=robot,
            verifier=VisualVerifier(),
            artifacts=run_artifacts,
            max_attempts=config.runtime.max_attempts,
            finalize_artifacts=not args.sort_all,
        )

    operation = (
        SortAllCoordinator(make_pipeline, artifacts) if args.sort_all else make_pipeline(artifacts)
    )
    return command, robot, operation


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    robot = None
    try:
        command, robot, operation = build_pipeline(args)
        result = operation.run(command) if isinstance(operation, ClosedLoopPipeline) else operation.run()
        payload = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        status = result.status if hasattr(result, "status") else result["status"]
        return 0 if status == PipelineState.DONE else 2
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"Environment error: {exc}", file=sys.stderr)
        print("Run: python -m graspfruit.env_check --strict", file=sys.stderr)
        return 3
    finally:
        if robot is not None:
            robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
