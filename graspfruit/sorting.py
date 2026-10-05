from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .artifacts import RunArtifacts
from .models import ExecutionResult, PipelineState
from .pipeline import ClosedLoopPipeline

SORT_PLAN = (
    ("apple", "苹果", "蓝框"),
    ("banana", "香蕉", "绿框"),
    ("pear", "青绿色鸭梨", "绿框"),
)


@dataclass
class SortAllCoordinator:
    """Run one fresh visual closed loop per fruit in the shared scene."""

    pipeline_factory: Callable[[RunArtifacts, str], ClosedLoopPipeline]
    artifacts: RunArtifacts
    instances_per_class: int = 3

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        items: list[dict[str, Any]] = []
        try:
            item_number = 0
            for target_name, chinese_name, bin_name in SORT_PLAN:
                for class_index in range(1, self.instances_per_class + 1):
                    item_number += 1
                    command = (
                        f"从桌面待分拣区选择一只尚未分拣的{chinese_name}，"
                        "优先选择靠近左侧机械臂、位于水果阵列左下或中部且无遮挡的实例，"
                        f"放进{bin_name}。不要选择已经位于任何框内的水果。"
                    )
                    prefix = f"item_{item_number:02d}_{target_name}"
                    pipeline = self.pipeline_factory(self.artifacts.scope(prefix), prefix)
                    result = pipeline.run(command)
                    item = {
                        "index": item_number,
                        "class_index": class_index,
                        "target_name": target_name,
                        "command": command,
                        "result": result.model_dump(mode="json"),
                    }
                    items.append(item)
                    self.artifacts.write_json(f"{prefix}_summary.json", item)
                    if result.status != PipelineState.DONE:
                        return self._finish(items, started, result)
            return self._finish(items, started)
        finally:
            self.artifacts.finalize()

    def _finish(
        self,
        items: list[dict[str, Any]],
        started: float,
        failed: ExecutionResult | None = None,
    ) -> dict[str, Any]:
        completed = sum(item["result"]["status"] == PipelineState.DONE.value for item in items)
        expected = len(SORT_PLAN) * self.instances_per_class
        summary = {
            "status": PipelineState.DONE.value if failed is None and completed == expected else PipelineState.FAILED.value,
            "expected_items": expected,
            "completed_items": completed,
            "failed_item": items[-1]["index"] if failed is not None else None,
            "reason": failed.reason if failed is not None else "all fruit sorted and visually verified",
            "elapsed_seconds": time.perf_counter() - started,
            "items": items,
        }
        self.artifacts.write_json("sort_result.json", summary)
        return summary
