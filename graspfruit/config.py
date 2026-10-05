from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class RuntimeConfig(BaseModel):
    max_attempts: int = Field(default=5, ge=1, le=10)
    seed: int = 7
    headless: bool = False
    record: bool = False
    run_root: Path = Path("runs")


class VLMConfig(BaseModel):
    model: str = "qwen3-vl-plus"
    base_url: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 45
    repair_attempts: int = 1


class SamConfig(BaseModel):
    model_id: str = "facebook/sam2.1-hiera-base-plus"
    min_mask_pixels: int = 250
    min_valid_depth_ratio: float = 0.65
    box_padding_pixels: int = Field(default=8, ge=0, le=64)


class GraspConfig(BaseModel):
    checkpoint: Path = Path("checkpoints/graspnet/checkpoint-rs.tar")
    third_party_root: Path = Path("third_party/graspnet-baseline")
    num_points: int = 12000
    max_candidates: int = 30
    collision_threshold: float = 0.01
    approach_angle_deg: float = 60
    min_width: float = 0.015
    max_width: float = 0.085
    network_width_tolerance: float = Field(default=0.015, ge=0.0, le=0.03)


class SimulationConfig(BaseModel):
    image_width: int = 640
    image_height: int = 480
    settle_steps: int = 500
    control_steps: int = 160
    timestep: float = 0.002


class AppConfig(BaseModel):
    runtime: RuntimeConfig = RuntimeConfig()
    vlm: VLMConfig = VLMConfig()
    sam: SamConfig = SamConfig()
    grasp: GraspConfig = GraspConfig()
    simulation: SimulationConfig = SimulationConfig()


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    default = Path(__file__).with_name("configs") / "default.yaml"
    with default.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if path:
        with path.open("r", encoding="utf-8") as handle:
            data = _merge(data, yaml.safe_load(handle) or {})
    if overrides:
        data = _merge(data, overrides)

    data.setdefault("vlm", {})
    data["vlm"]["api_key"] = os.getenv("GRASPFRUIT_VLM_API_KEY", data["vlm"].get("api_key"))
    data["vlm"]["base_url"] = os.getenv("GRASPFRUIT_VLM_BASE_URL", data["vlm"].get("base_url"))
    data["vlm"]["model"] = os.getenv("GRASPFRUIT_VLM_MODEL", data["vlm"].get("model"))
    return AppConfig.model_validate(data)
