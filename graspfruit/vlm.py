from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass

import cv2
import numpy as np
from openai import OpenAI
from pydantic import ValidationError

from .config import VLMConfig
from .models import BinName, GroundingDecision, PipelineError, PipelineState

SYSTEM_PROMPT = """你是机器人抓取系统的视觉定位模块。根据单张图像和中文指令，选择唯一的水果和目标果篮。
图像坐标原点在左上角。只输出一个 JSON 对象，不要使用 Markdown：
{
  "target_name": "apple|banana|pear",
  "attributes": ["small", "rightmost"],
  "relation": "用于消歧的空间关系",
  "reasoning": "简短且可审计的选择依据",
  "bbox_norm": [x1, y1, x2, y2],
  "destination": "blue_bin|green_bin",
  "confidence": 0.0
}
bbox_norm 必须是 0 到 1 的归一化 xyxy。场景中只有红苹果、黄香蕉和青绿色鸭梨。
固定分类规则是苹果进入蓝框、香蕉和鸭梨进入绿框，不得改变映射。
只选择桌面待分拣区域内的水果，不要选择已经放在任一框内的水果。
先逐一比较同类实例的大小和空间关系，再给出紧贴目标且不包含相邻水果的框。
如果指令无法唯一消歧，将 confidence 设为 0 并解释歧义；不得猜测不存在的物体或果篮。"""


def _encode_jpeg(rgb: np.ndarray) -> str:
    ok, data = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("failed to encode RGB image")
    return base64.b64encode(data).decode("ascii")


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(text[start : end + 1])


@dataclass
class OpenAICompatibleVLM:
    config: VLMConfig
    minimum_confidence: float = 0.35

    def __post_init__(self) -> None:
        if not self.config.api_key:
            raise RuntimeError("GRASPFRUIT_VLM_API_KEY is not configured")
        kwargs = {"api_key": self.config.api_key, "timeout": self.config.timeout_seconds}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        self.client = OpenAI(**kwargs)
        self.last_raw_response = ""

    def _request(self, command: str, rgb: np.ndarray, repair: str | None = None) -> str:
        prompt = f"用户指令：{command}"
        if repair:
            prompt += f"\n上一次输出未通过校验：{repair}\n请重新查看原图并只输出合法 JSON。"
        completion = self.client.chat.completions.create(
            model=self.config.model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{_encode_jpeg(rgb)}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        )
        return completion.choices[0].message.content or ""

    def ground(self, command: str, rgb: np.ndarray) -> GroundingDecision:
        error = None
        for attempt in range(self.config.repair_attempts + 1):
            try:
                raw = self._request(command, rgb, repair=error if attempt else None)
                self.last_raw_response = raw
                decision = GroundingDecision.model_validate(extract_json(raw))
                if decision.confidence < self.minimum_confidence:
                    raise ValueError(f"ambiguous target: {decision.reasoning}")
                return decision
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                error = str(exc)
            except Exception as exc:
                raise PipelineError(PipelineState.GROUND, f"VLM request failed: {exc}") from exc
        raise PipelineError(PipelineState.GROUND, f"VLM output invalid after repair: {error}")


class HeuristicVLM:
    """Development-only color/shape grounder; enabled only by --mock-perception."""

    def ground(self, command: str, rgb: np.ndarray) -> GroundingDecision:
        destination = BinName.GREEN if "绿" in command else BinName.BLUE
        if "香蕉" in command:
            name, color = "banana", (200, 170, 20)
        elif "梨" in command:
            name, color = "pear", (145, 170, 25)
        else:
            name, color = "apple", (190, 25, 25)
        delta = np.linalg.norm(rgb.astype(float) - np.asarray(color), axis=2)
        mask = (delta < 125).astype(np.uint8)
        count, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        parts = [tuple(stats[i]) for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 50]
        if not parts:
            raise PipelineError(PipelineState.GROUND, f"mock grounder found no {name}")
        if "最右" in command or "右边" in command:
            chosen = max(parts, key=lambda item: item[0] + item[2] / 2)
        elif "最左" in command or "左边" in command:
            chosen = min(parts, key=lambda item: item[0] + item[2] / 2)
        elif "较小" in command or "最小" in command:
            chosen = min(parts, key=lambda item: item[4])
        else:
            chosen = max(parts, key=lambda item: item[4])
        x, y, width_px, height_px, _ = chosen
        image_height, image_width = rgb.shape[:2]
        return GroundingDecision(
            target_name=name,
            attributes=["development_heuristic"],
            relation="color connected component",
            reasoning="explicit --mock-perception development mode",
            bbox_norm=(
                x / image_width,
                y / image_height,
                (x + width_px) / image_width,
                (y + height_px) / image_height,
            ),
            destination=destination,
            confidence=0.5,
        )
