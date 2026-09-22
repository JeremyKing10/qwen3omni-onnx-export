from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from qwen3_omni_thinking_components import THINKING_COMPONENTS

WORKSPACE = Path(__file__).resolve().parent
PACKAGE = WORKSPACE / "Qwen3-Omni-30B-A3B-Thinking-ONNX"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键执行本地 tiny 全流程（无需下载权重）")
    parser.add_argument(
        "--force",
        action="store_true",
        help="允许用 tiny 端到端报告覆盖已有的 real 端到端报告（默认拒绝，防止证据被降级覆盖）",
    )
    return parser.parse_args()


def run(*arguments: str) -> None:
    command = [sys.executable, *arguments]
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, cwd=WORKSPACE, check=True)


def operator_report_name(component: str) -> str:
    return component.replace("_encoder", "") + ".json"


def main() -> None:
    args = parse_args()
    run(
        "export_thinking_onnx.py",
        "--mode",
        "tiny",
        "--component",
        "all",
        "--package-dir",
        str(PACKAGE),
        "--force",
    )
    for component in THINKING_COMPONENTS:
        model_path = PACKAGE / "onnx" / component / "model.onnx"
        report_path = PACKAGE / "operators" / operator_report_name(component)
        run("validate_onnx.py", "--model", str(model_path))
        run(
            "inspect_onnx.py",
            "--model",
            str(model_path),
            "--output",
            str(report_path),
            "--fail-on-custom-domain",
        )
    pipeline_arguments = ["validate_thinking_pipeline.py", "--package-dir", str(PACKAGE)]
    if args.force:
        pipeline_arguments.append("--force")
    run(*pipeline_arguments)
    run("aggregate_operators.py", "--package-dir", str(PACKAGE))
    run("build_thinking_package.py", "--package-dir", str(PACKAGE), "--offline")
    print(f"\n[OK] local Thinking pipeline completed: {PACKAGE}")


if __name__ == "__main__":
    main()
