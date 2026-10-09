from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import psutil
import torch

from export_onnx import export_transaction
from onnx_artifact_utils import check_execution_resources, model_identity, safe_path, save_tensor_archive, source_snapshot
from qwen3_omni_onnx_cases import (
    DEFAULT_SEED,
    WORKSPACE,
    file_sha256,
    normalize_outputs,
    runtime_metadata,
    tensor_dict,
    write_json,
)
from qwen3_omni_thinking_components import (
    THINKING_COMPONENTS,
    ThinkingComponentCase,
    build_real_thinking_component,
    build_tiny_thinking_component,
    component_routing,
    inspect_checkpoint,
    load_real_thinking_model,
)

DEFAULT_PACKAGE = WORKSPACE / "Qwen3-Omni-30B-A3B-Thinking-ONNX"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 Thinking 四组件的正式接口 ONNX")
    parser.add_argument("--mode", choices=("tiny", "real"), default="tiny", help="本机 tiny 回归或官方权重导出")
    parser.add_argument(
        "--component",
        choices=(*THINKING_COMPONENTS, "all"),
        default="all",
        help="要导出的组件，默认全部",
    )
    parser.add_argument("--model-path", type=Path, help="real 模式下的官方 Thinking checkpoint 本地目录")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cpu", help="real 模式模型设备，例如 cpu 或 cuda")
    parser.add_argument("--minimum-memory-gib", type=float, default=128.0, help="real 模式最低物理内存保护线")
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE, help="产品根目录")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true", help="覆盖对应组件目录")
    parser.add_argument("--allow-real-overwrite", action="store_true", help="显式允许 tiny 覆盖 real 或身份未知的现有产物；还需 --force")
    return parser.parse_args()


def estimate_weight_bytes(checkpoint_path: Path) -> int:
    return inspect_checkpoint(checkpoint_path)["weight_bytes"]


def existing_real_evidence(package_dir: Path) -> list[str]:
    paths = [package_dir / "manifest.json", package_dir / "validation/end_to_end.json"]
    paths += [package_dir / "onnx" / name / "export_metadata.json" for name in THINKING_COMPONENTS]
    conflicts = []
    for candidate in paths:
        path = safe_path(WORKSPACE, candidate)
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("not an object")
                if path.name == "export_metadata.json":
                    tiny = (data.get("profile") == "tiny-fixed-shape" and data.get("case") == path.parent.name
                            and data.get("product_component") == path.parent.name
                            and data.get("official_weights_included") is False and data.get("randomly_initialized") is True
                            and not data.get("checkpoint_fingerprint"))
                elif path.name == "end_to_end.json":
                    tiny = data.get("profile") == "tiny-fixed-shape"
                else:
                    tiny = data.get("official_weights_included") is False and data.get("status") in {"tiny-interface-validation-only", "unverified-tiny-artifacts"}
                if not tiny:
                    conflicts.append(f"real 或无法确认 tiny 身份：{path}")
            except (ValueError, TypeError):
                conflicts.append(f"身份未知：{path}")
    for name in THINKING_COMPONENTS:
        root = package_dir / "onnx" / name
        if (root / "model.onnx").exists() and not (root / "export_metadata.json").is_file():
            conflicts.append(f"缺少元数据：{root}")
    return conflicts


def authorize_tiny_export(package_dir: Path, allow: bool) -> None:
    conflicts = existing_real_evidence(package_dir)
    if conflicts and not allow:
        raise RuntimeError("拒绝覆盖 real/未知产物，需明确降级授权：" + ", ".join(conflicts))


def invalidate_package_evidence(package_dir: Path) -> None:
    """删除已经与当前 ONNX 不再对应的全局证据（manifest / 端到端 / 算子汇总）。"""
    for stale in (
        package_dir / "validation" / "end_to_end.json",
        package_dir / "manifest.json",
        package_dir / "operators" / "summary.json",
        package_dir / "operators" / "all_operators.csv",
    ):
        if stale.is_file() and not stale.is_symlink():
            stale.unlink()


def resolve_package_dir(package_dir: Path) -> Path:
    package_dir = safe_path(WORKSPACE, package_dir)
    if package_dir == WORKSPACE:
        raise ValueError("产品目录不能是工作区根目录")
    for candidate in (package_dir, package_dir / "onnx", package_dir / "validation", package_dir / "operators"):
        if candidate.exists() and candidate.is_symlink():
            raise ValueError(f"拒绝符号链接路径：{candidate}")
    return package_dir


def check_component_dir(package_dir: Path, component: str, force: bool) -> Path:
    output_dir = safe_path(package_dir, package_dir / "onnx" / component)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"{output_dir} 非空；如需覆盖请添加 --force")
    return output_dir


def export_component(
    case: ThinkingComponentCase,
    output_dir: Path,
    opset: int,
    seed: int,
    profile: str,
    snapshot: dict[str, Any],
) -> dict:
    if profile.startswith("real-") and (not case.loading_verified or not case.checkpoint_fingerprint):
        raise RuntimeError("真实导出必须经过完整 checkpoint 加载检查")
    model_path = output_dir / "model.onnx"
    case.model.requires_grad_(False)
    test_vectors = []
    routing_hashes = set()

    for index, vector in enumerate(case.test_vectors):
        with torch.inference_mode():
            reference = normalize_outputs(case.model(*vector))
        input_name = f"inputs_{index}.npz"
        reference_name = f"reference_outputs_{index}.npz"
        input_dtypes = save_tensor_archive(output_dir / input_name, tensor_dict(case.input_names, vector))
        reference_dtypes = save_tensor_archive(
            output_dir / reference_name, tensor_dict(case.output_names, reference)
        )
        routing = component_routing(case.model, vector)
        if routing:
            routing_hashes.add(routing["sha256"])
        test_vectors.append(
            {
                "index": index,
                "input_file": input_name,
                "reference_output_file": reference_name,
                "input_sha256": file_sha256(output_dir / input_name),
                "reference_output_sha256": file_sha256(output_dir / reference_name),
                "input_dtypes": input_dtypes,
                "reference_output_dtypes": reference_dtypes,
                "routing": routing,
            }
        )

    if case.name in {"thinker_prefill", "thinker_decode"} and len(routing_hashes) < 2:
        raise RuntimeError(f"{case.name} 的两组输入未产生不同 MoE 路由")

    started = time.perf_counter()
    with torch.inference_mode():
        torch.onnx.export(
            case.model,
            case.export_args,
            model_path,
            input_names=list(case.input_names),
            output_names=list(case.output_names),
            opset_version=opset,
            dynamo=True,
            external_data=True,
            optimize=True,
            export_params=True,
            dynamic_shapes=case.dynamic_shapes,
        )
    elapsed = time.perf_counter() - started
    onnx.checker.check_model(str(model_path), full_check=True)
    graph = onnx.load(str(model_path), load_external_data=False)

    identity = model_identity(model_path)
    if source_snapshot()["files"] != snapshot["files"]:
        raise RuntimeError("导出期间工具源码发生变化，拒绝写入可能不对应的导出元数据")
    metadata = {
        "evidence_schema_version": 2,
        "case": case.name,
        "product_component": case.name,
        "model_identity": identity,
        "source_snapshot": snapshot,
        "profile": profile,
        "official_weights_included": profile.startswith("real-"),
        "randomly_initialized": profile.startswith("tiny-"),
        "description": case.description,
        "model_file": model_path.name,
        "model_sha256": file_sha256(model_path),
        "test_vectors": test_vectors,
        "input_names": list(case.input_names),
        "output_names": list(case.output_names),
        "interface": case.interface,
        "source_equivalence": case.source_equivalence,
        "opset_requested": opset,
        "opset_imports": {item.domain or "ai.onnx": item.version for item in graph.opset_import},
        "seed": seed,
        "fixed_shapes": case.dynamic_shapes is None,
        "dynamic_shape_contract": case.interface if case.dynamic_shapes is not None else None,
        "checkpoint_fingerprint": case.checkpoint_fingerprint,
        "loading_verified": case.loading_verified,
        "uses_external_data": any(initializer.external_data for initializer in graph.graph.initializer),
        "export_seconds": elapsed,
        "model_bytes": model_path.stat().st_size,
        "config": case.config,
        "runtime": runtime_metadata(),
    }
    write_json(output_dir / "export_metadata.json", metadata)
    print(f"[OK] {case.name}: {model_path}, nodes={len(graph.graph.node)}, seconds={elapsed:.3f}")
    return metadata


def main() -> None:
    args = parse_args()
    if args.opset < 18:
        raise ValueError("Dynamo ONNX exporter 要求本流程使用 opset >= 18")
    components = THINKING_COMPONENTS if args.component == "all" else (args.component,)
    package_dir = resolve_package_dir(args.package_dir)
    evidence = tuple(package_dir / name for name in (
        "manifest.json", "validation/end_to_end.json", "operators/summary.json", "operators/all_operators.csv",
    ))

    full_model = None
    if args.mode == "real":
        if args.model_path is None:
            raise ValueError("real 模式必须提供 --model-path")
        checkpoint_path = args.model_path.expanduser().resolve()
        check_execution_resources(inspect_checkpoint(checkpoint_path), package_dir, args.minimum_memory_gib,
                                  args.device, "CPUExecutionProvider")
        dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
        full_model = load_real_thinking_model(str(checkpoint_path), dtype=dtype, device=args.device)

    snapshot = source_snapshot()
    profile = f"{args.mode}-fixed-shape"

    def export_one(name: str, staging: Path) -> None:
        case = (
            build_tiny_thinking_component(name, args.seed)
            if args.mode == "tiny"
            else build_real_thinking_component(full_model, name, args.seed)
        )
        export_component(case, staging, args.opset, args.seed, profile, snapshot)

    # 先在暂存目录导出并通过 Checker，全部成功后再整体替换旧产物并失效化旧全局证据；
    # 任何失败都会回滚，旧 ONNX 与旧证据保持原样。
    export_transaction(
        {name: check_component_dir(package_dir, name, args.force) for name in components},
        args.force,
        export_one,
        evidence=evidence,
        lock_root=package_dir,
        authorize=(lambda: authorize_tiny_export(package_dir, args.force and args.allow_real_overwrite))
        if args.mode == "tiny" else None,
    )


if __name__ == "__main__":
    main()
