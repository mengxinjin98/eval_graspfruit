from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CASES = [
    "把最左边的香蕉放进绿色篮子",
    "把较小的苹果放进蓝色篮子",
    "把离香蕉最近的苹果放进蓝色篮子",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the fixed-seed regression protocol")
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--mock-perception", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/batch_summary.json"))
    args = parser.parse_args(argv)
    results = []
    for seed in range(args.seeds):
        command = CASES[seed % len(CASES)]
        call = [
            sys.executable, "-m", "graspfruit.demo", "--headless",
            "--seed", str(seed), "--command", command,
        ]
        if args.mock_perception:
            call.append("--mock-perception")
        completed = subprocess.run(call, check=False)
        run_files = sorted(Path("runs").glob("*-seed*/result.json"), key=lambda p: p.stat().st_mtime)
        payload = json.loads(run_files[-1].read_text(encoding="utf-8")) if run_files else None
        results.append(
            {
                "seed": seed,
                "command": command,
                "exit_code": completed.returncode,
                "status": payload.get("status") if payload else "NO_RESULT",
                "attempts": payload.get("attempts") if payload else None,
                "failure_stage": payload.get("failure_stage") if payload else "NO_RESULT",
                "mock_mode": payload.get("metrics", {}).get("mock_mode") if payload else None,
            }
        )
    success = sum(item["status"] == "DONE" for item in results)
    rate = success / max(len(results), 1)
    formal = not any(item["mock_mode"] for item in results)
    summary = {
        "formal_experiment": formal,
        "successes": success,
        "episodes": len(results),
        "success_rate": rate,
        "acceptance_threshold": 0.8,
        "accepted": formal and rate >= 0.8,
        "episodes_detail": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"success={success}/{len(results)} ({rate:.1%}); "
        f"formal={'yes' if formal else 'no (mock results excluded)'}"
    )
    return 0 if summary["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
