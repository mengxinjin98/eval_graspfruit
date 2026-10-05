from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def run_checks(formal: bool = True) -> list[Check]:
    checks: list[Check] = []
    checks.append(
        Check(
            "Python 3.11",
            sys.version_info[:2] == (3, 11),
            sys.version.split()[0],
            formal,
        )
    )
    for module, required in (
        ("numpy", True), ("pydantic", True), ("cv2", True), ("mujoco", True),
        ("torch", True), ("openai", formal), ("huggingface_hub", formal),
        ("sam2", formal), ("graspnetAPI", formal),
    ):
        found = importlib.util.find_spec(module) is not None
        checks.append(Check(f"Python module {module}", found, "installed" if found else "missing", required))
    try:
        import torch

        checks.append(Check("CUDA available", torch.cuda.is_available(), torch.version.cuda or "none", formal))
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            checks.append(Check("GPU sm_120", capability >= (12, 0), f"{torch.cuda.get_device_name(0)} sm_{capability[0]}{capability[1]}", formal))
    except Exception as exc:  # noqa: BLE001 - diagnostic tool reports arbitrary driver failures
        checks.append(Check("PyTorch CUDA", False, str(exc), formal))
    extension = False
    try:
        extension = importlib.util.find_spec("pointnet2._ext") is not None
    except ModuleNotFoundError:
        pass
    checks.append(Check("PointNet++ CUDA extension", extension, "import pointnet2._ext", formal))
    # cl/nvcc are build-time requirements. Once the extension imports, they
    # need not remain on PATH for normal inference runs.
    build_tools_required = formal and not extension
    nvcc = shutil.which("nvcc")
    checks.append(Check("CUDA Toolkit nvcc", nvcc is not None, nvcc or "build-time only; not on PATH", build_tools_required))
    cl = shutil.which("cl")
    checks.append(Check("MSVC cl.exe", cl is not None, cl or "build-time only; open a VS developer shell", build_tools_required))
    checkpoint = Path("checkpoints/graspnet/checkpoint-rs.tar")
    checks.append(Check("GraspNet checkpoint", checkpoint.exists(), str(checkpoint), formal))
    key = os.getenv("GRASPFRUIT_VLM_API_KEY")
    checks.append(Check("VLM API key", bool(key), "configured" if key else "GRASPFRUIT_VLM_API_KEY missing", formal))
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="require all formal inference dependencies")
    args = parser.parse_args(argv)
    checks = run_checks(formal=args.strict)
    for check in checks:
        icon = "OK" if check.ok else ("FAIL" if check.required else "WARN")
        print(f"[{icon:4}] {check.name:30} {check.detail}")
    return 1 if any(check.required and not check.ok for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
