from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from build_thinking_package import resolve_package_sources
from onnx_artifact_utils import safe_path
from qwen3_omni_thinking_components import THINKING_COMPONENTS

WORKSPACE = Path(__file__).resolve().parent
PACKAGE = WORKSPACE / "Qwen3-Omni-30B-A3B-Thinking-ONNX"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键执行本地 tiny 全流程（无需下载权重）")
    parser.add_argument("--package-dir", type=Path, default=PACKAGE)
    parser.add_argument(
        "--force",
        action="store_true",
        help="允许用 tiny 覆盖已有的 real 产品（默认在任何写入前拒绝，防止证据被降级覆盖）",
    )
    parser.add_argument("--source-dir", type=Path, help="非权重配置/Processor 的本地来源目录")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="只使用本地文件或已有缓存获取非权重资源（默认允许联网获取这些小文件）",
    )
    return parser.parse_args()


def run(*arguments: str) -> None:
    command = [sys.executable, *arguments]
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, cwd=WORKSPACE, check=True)


def operator_report_name(component: str) -> str:
    return component.replace("_encoder", "") + ".json"


def existing_real_evidence(package_dir: Path) -> list[str]:
    found: list[str] = []
    manifest = package_dir / "manifest.json"
    if manifest.is_file():
        try:
            payload: Any = json.loads(manifest.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            raise RuntimeError(f"manifest.json 无法解析，拒绝继续以免覆盖无法确认的 real 证据：{manifest}")
        if payload.get("official_weights_included") or payload.get("status") in {
            "official-weight-components-validated", "unverified-real-artifacts",
        }:
            found.append(str(manifest))
    for component in THINKING_COMPONENTS:
        path = package_dir / "onnx" / component / "export_metadata.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            raise RuntimeError(f"导出元数据无法解析，拒绝继续以免覆盖 real 产物：{path}")
        if payload.get("profile") == "real-fixed-shape" or payload.get("official_weights_included"):
            found.append(str(path))
    return found


def main() -> None:
    args = parse_args()
    package_dir = safe_path(WORKSPACE, args.package_dir)
    conflicts = existing_real_evidence(package_dir)
    if conflicts and not args.force:
        raise SystemExit(
            "检测到已有 real 产物/证据，默认拒绝用 tiny 覆盖："
            + ", ".join(conflicts)
            + "；确需覆盖请显式加 --force（会先备份旧产物与旧证据）"
        )
    # 导出之前先确认非权重资源可用，避免前面已完成导出后才发现打包资源缺失。
    resolve_package_sources(source_dir=args.source_dir, offline=args.offline)
    export_arguments = ["export_thinking_onnx.py", "--mode", "tiny", "--component", "all",
                        "--package-dir", str(package_dir), "--force"]
    run(*export_arguments)
    for component in THINKING_COMPONENTS:
        model_path = package_dir / "onnx" / component / "model.onnx"
        report_path = package_dir / "operators" / operator_report_name(component)
        run("validate_onnx.py", "--model", str(model_path))
        run("inspect_onnx.py", "--model", str(model_path), "--output", str(report_path), "--fail-on-custom-domain")
    pipeline_arguments = ["validate_thinking_pipeline.py", "--package-dir", str(package_dir)]
    if args.force:
        pipeline_arguments.append("--force")
    run(*pipeline_arguments)
    run("aggregate_operators.py", "--package-dir", str(package_dir))
    build_arguments = ["build_thinking_package.py", "--package-dir", str(package_dir)]
    if args.source_dir is not None:
        build_arguments += ["--source-dir", str(args.source_dir)]
    if args.offline:
        build_arguments.append("--offline")
    run(*build_arguments)
    print(f"\n[OK] local Thinking pipeline completed: {package_dir}")


if __name__ == "__main__":
    main()
