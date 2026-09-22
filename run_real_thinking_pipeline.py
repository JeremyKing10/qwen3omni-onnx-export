from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from qwen3_omni_thinking_components import THINKING_COMPONENTS

WORKSPACE = Path(__file__).resolve().parent
DEFAULT_PACKAGE = WORKSPACE / "Qwen3-Omni-30B-A3B-Thinking-ONNX"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在大内存 Linux 上导出并验证官方 Thinking 四组件")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--minimum-memory-gib", type=float, default=128.0)
    parser.add_argument(
        "--run-end-to-end",
        action="store_true",
        help="额外加载官方 PyTorch 模型做端到端三步 Decode；建议至少 192 GiB RAM",
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
    package = args.package_dir.expanduser().resolve()
    common = (
        "--mode",
        "real",
        "--model-path",
        str(args.model_path.expanduser().resolve()),
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--minimum-memory-gib",
        str(args.minimum_memory_gib),
        "--package-dir",
        str(package),
    )
    run("export_thinking_onnx.py", *common, "--component", "all", "--force")
    for component in THINKING_COMPONENTS:
        model_path = package / "onnx" / component / "model.onnx"
        report_path = package / "operators" / operator_report_name(component)
        run("validate_onnx.py", "--model", str(model_path), "--provider", args.provider)
        run(
            "inspect_onnx.py",
            "--model",
            str(model_path),
            "--output",
            str(report_path),
            "--fail-on-custom-domain",
        )
    if args.run_end_to_end:
        run(
            "validate_thinking_pipeline.py",
            *common,
            "--provider",
            args.provider,
            "--minimum-memory-gib",
            str(max(args.minimum_memory_gib, 192.0)),
        )
    run("aggregate_operators.py", "--package-dir", str(package))
    # 打包可能因为未达验收判据而失败退出，这里显式检查退出码，避免 traceback 掩盖真实原因
    command = [
        sys.executable,
        "build_thinking_package.py",
        "--package-dir",
        str(package),
        "--source-dir",
        str(args.model_path.expanduser().resolve()),
        "--offline",
    ]
    print("\n$", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=WORKSPACE, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    status = json.loads((package / "manifest.json").read_text(encoding="utf-8")).get("status")
    if status != "official-weight-components-validated":
        print(
            f"\n[FAIL] status={status}：未达到 README 第 9.3 节的官方权重验收判据",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"\n[OK] official-weight Thinking package completed: {package}")


if __name__ == "__main__":
    main()
