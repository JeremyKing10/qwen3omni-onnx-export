from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from build_thinking_package import resolve_package_sources
from onnx_artifact_utils import check_execution_resources, safe_path
from qwen3_omni_thinking_components import THINKING_COMPONENTS, inspect_checkpoint

WORKSPACE = Path(__file__).resolve().parent
DEFAULT_PACKAGE = WORKSPACE / "Qwen3-Omni-30B-A3B-Thinking-ONNX"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在大内存 Linux 上导出并验证官方 Thinking 四组件")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cpu", help="PyTorch 设备；与 ONNX Runtime provider 是两回事")
    parser.add_argument("--provider", default="CPUExecutionProvider", help="ONNX Runtime provider")
    parser.add_argument("--minimum-memory-gib", type=float, default=128.0)
    parser.add_argument("--source-dir", type=Path, help="非权重配置/Processor 的本地来源目录")
    parser.add_argument("--offline", action="store_true", help="只使用本地文件/缓存获取非权重资源")
    parser.add_argument(
        "--run-end-to-end",
        action="store_true",
        help="额外加载官方 PyTorch 模型做端到端三步 Decode；建议至少 192 GiB RAM",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只做前置检查（不加载 59 GiB 权重、不写产品目录）：checkpoint、内存、磁盘、device/provider、资源可用性",
    )
    return parser.parse_args()


def run(*arguments: str) -> None:
    command = [sys.executable, *arguments]
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, cwd=WORKSPACE, check=True)


def operator_report_name(component: str) -> str:
    return component.replace("_encoder", "") + ".json"


def preflight(args: argparse.Namespace) -> Path:
    package_dir = safe_path(WORKSPACE, args.package_dir)
    checkpoint = Path(args.model_path).expanduser().resolve()
    inspected = inspect_checkpoint(checkpoint)
    resources = check_execution_resources(inspected, package_dir, args.minimum_memory_gib,
                                          args.device, args.provider, end_to_end=args.run_end_to_end)
    source = (args.source_dir or checkpoint).expanduser().resolve()
    sources = resolve_package_sources(source_dir=source, offline=args.offline)
    if sources["config.json"].read_bytes() != (checkpoint / "config.json").read_bytes():
        raise ValueError("--source-dir 的 config.json 与权重 checkpoint 不一致")
    print(f"[OK] checkpoint headers/tensor mapping: {inspected['tensor_count']} tensors")
    print("[OK] resource preflight:", json.dumps(resources, ensure_ascii=False))
    print(f"[OK] non-weight source: {source}")
    return source


def main() -> None:
    args = parse_args()
    package = safe_path(WORKSPACE, args.package_dir)
    if args.preflight_only:
        preflight(args)
        print("\n[OK] preflight-only 完成：未加载官方权重，未写入产品目录")
        return
    source = preflight(args)
    common = (
        "--mode", "real", "--model-path", str(Path(args.model_path).expanduser().resolve()),
        "--dtype", args.dtype, "--device", args.device,
        "--minimum-memory-gib", str(args.minimum_memory_gib), "--package-dir", str(package),
    )
    run("export_thinking_onnx.py", *common, "--component", "all", "--force")
    for component in THINKING_COMPONENTS:
        model_path = package / "onnx" / component / "model.onnx"
        report_path = package / "operators" / operator_report_name(component)
        run("validate_onnx.py", "--model", str(model_path), "--provider", args.provider)
        run("inspect_onnx.py", "--model", str(model_path), "--output", str(report_path), "--fail-on-custom-domain")
    if args.run_end_to_end:
        run(
            "validate_thinking_pipeline.py", *common, "--provider", args.provider,
            "--minimum-memory-gib", str(max(args.minimum_memory_gib, 192.0)),
        )
    run("aggregate_operators.py", "--package-dir", str(package))
    build_arguments = ["build_thinking_package.py", "--package-dir", str(package),
                       "--source-dir", str(source)]
    if args.offline:
        build_arguments.append("--offline")
    command = [sys.executable, *build_arguments]
    print("\n$", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=WORKSPACE, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    status = json.loads((package / "manifest.json").read_text(encoding="utf-8")).get("status")
    if status != "official-weight-components-validated":
        print(f"\n[FAIL] status={status}：未达到官方权重验收判据（未启用 --run-end-to-end 时这是预期结果）",
              file=sys.stderr)
        raise SystemExit(1)
    print(f"\n[OK] official-weight Thinking package completed: {package}")


if __name__ == "__main__":
    main()
