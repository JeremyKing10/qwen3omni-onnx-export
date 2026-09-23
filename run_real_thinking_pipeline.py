from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from build_thinking_package import resolve_package_sources
from onnx_artifact_utils import check_providers, safe_path
from qwen3_omni_thinking_components import THINKING_COMPONENTS

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


def preflight(args: argparse.Namespace) -> None:
    if not (isinstance(args.minimum_memory_gib, float) and math.isfinite(args.minimum_memory_gib)
            and args.minimum_memory_gib > 0):
        raise ValueError(f"--minimum-memory-gib 必须为有限正数，实际为 {args.minimum_memory_gib}")
    checkpoint = Path(args.model_path).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"官方 checkpoint 目录不存在：{checkpoint}")
    for name in ("config.json", "model.safetensors.index.json"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"官方 checkpoint 缺少 {name}：{checkpoint}")
    check_providers([args.provider])
    if args.provider == "CUDAExecutionProvider" and not args.device.startswith("cuda"):
        print("[WARN] provider 使用 CUDA，但 PyTorch device 不是 cuda；两端是独立设置", file=sys.stderr)
    if args.run_end_to_end and args.minimum_memory_gib < 192.0:
        raise ValueError("启用 --run-end-to-end 时 --minimum-memory-gib 至少 192（当前为 "
                         f"{args.minimum_memory_gib}），否则会在导出完成后才发现无法验收")
    # 端到端验证会在同一进程内同时驻留 PyTorch 模型与 ORT 会话，需要额外的内存/显存余量。
    required_gib = args.minimum_memory_gib * (1.6 if args.run_end_to_end else 1.0)
    total, _, free = shutil.disk_usage(checkpoint)
    print(f"[OK] checkpoint={checkpoint}")
    print(f"[OK] disk_free={free / 1024**3:.1f} GiB / total={total / 1024**3:.1f} GiB")
    print(f"[OK] planned_memory_need>= {required_gib:.0f} GiB（含端到端时的 PyTorch+ORT 双份驻留估算）")
    resolve_package_sources(source_dir=args.source_dir, offline=args.offline)
    print("[OK] 非权重配置/Processor 资源可用")


def main() -> None:
    args = parse_args()
    package = safe_path(WORKSPACE, args.package_dir)
    if args.preflight_only:
        preflight(args)
        print("\n[OK] preflight-only 完成：未加载官方权重，未写入产品目录")
        return
    preflight(args)
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
                       "--source-dir", str(Path(args.model_path).expanduser().resolve())]
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
