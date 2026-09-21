from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from qwen3_omni_thinking_components import THINKING_COMPONENTS

WORKSPACE = Path(__file__).resolve().parent
PACKAGE = WORKSPACE / "Qwen3-Omni-30B-A3B-Thinking-ONNX"


def run(*arguments: str) -> None:
    command = [sys.executable, *arguments]
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, cwd=WORKSPACE, check=True)


def operator_report_name(component: str) -> str:
    return component.replace("_encoder", "") + ".json"


def main() -> None:
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
    run("validate_thinking_pipeline.py", "--package-dir", str(PACKAGE))
    run("aggregate_operators.py", "--package-dir", str(PACKAGE))
    run("build_thinking_package.py", "--package-dir", str(PACKAGE), "--offline")
    print(f"\n[OK] local Thinking pipeline completed: {PACKAGE}")


if __name__ == "__main__":
    main()
