from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import onnx
import psutil
import torch

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
    return parser.parse_args()


def estimate_weight_bytes(checkpoint_path: Path) -> int:
    """估算官方 checkpoint 的权重总字节数；索引缺失字段时改用磁盘上的分片实际大小。"""
    index_path = checkpoint_path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        declared = int(index.get("metadata", {}).get("total_size", 0))
        if declared:
            return declared
        shard_names = sorted(set(index.get("weight_map", {}).values()))
    else:
        shard_names = ["model.safetensors"]
    total = 0
    for name in shard_names:
        shard = checkpoint_path / name
        if shard.is_file():
            total += shard.stat().st_size
    if not total:
        raise RuntimeError(f"无法估算权重体积：{checkpoint_path}（既无 index 元数据也无可用权重分片）")
    return total


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


def safe_component_dir(package_dir: Path, component: str, force: bool) -> Path:
    package_dir = package_dir.expanduser().resolve()
    workspace = WORKSPACE.resolve()
    try:
        package_dir.relative_to(workspace)
    except ValueError as error:
        raise ValueError(f"产品目录必须位于工作区 {workspace} 内") from error
    if package_dir == workspace:
        raise ValueError("产品目录不能是工作区根目录")
    output_dir = package_dir / "onnx" / component
    resolved_output = output_dir.resolve()
    try:
        resolved_output.relative_to(package_dir)
    except ValueError as error:
        raise ValueError(f"输出目录经符号链接解析后越出产品目录：{output_dir}") from error
    for candidate in (package_dir, package_dir / "onnx", output_dir):
        if candidate.exists() and candidate.is_symlink():
            raise ValueError(f"拒绝符号链接路径：{candidate}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(f"{output_dir} 非空；如需覆盖请添加 --force")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def export_component(
    case: ThinkingComponentCase,
    output_dir: Path,
    opset: int,
    seed: int,
    profile: str,
) -> dict:
    model_path = output_dir / "model.onnx"
    case.model.requires_grad_(False)
    test_vectors = []
    routing_hashes = set()

    for index, vector in enumerate(case.test_vectors):
        with torch.inference_mode():
            reference = normalize_outputs(case.model(*vector))
        input_name = f"inputs_{index}.npz"
        reference_name = f"reference_outputs_{index}.npz"
        np.savez(output_dir / input_name, **tensor_dict(case.input_names, vector))
        np.savez(output_dir / reference_name, **tensor_dict(case.output_names, reference))
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

    metadata = {
        "case": case.name,
        "product_component": case.name,
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

    full_model = None
    if args.mode == "real":
        if args.model_path is None:
            raise ValueError("real 模式必须提供 --model-path")
        memory = psutil.virtual_memory()
        total_memory_gib = memory.total / 1024**3
        available_memory_gib = memory.available / 1024**3
        if total_memory_gib < args.minimum_memory_gib or available_memory_gib < args.minimum_memory_gib * 0.75:
            raise RuntimeError(
                f"real 模式需要至少 {args.minimum_memory_gib:.0f} GiB 总内存且至少 "
                f"{args.minimum_memory_gib * 0.75:.0f} GiB 可用；当前总计 {total_memory_gib:.1f} GiB、"
                f"可用 {available_memory_gib:.1f} GiB"
            )
        checkpoint_path = args.model_path.expanduser().resolve()
        if args.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("请求 CUDA 导出，但 torch.cuda 不可用")
            device = torch.device(args.device)
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"请求的 CUDA 设备不存在：{args.device}（可用显卡数 {torch.cuda.device_count()}）"
                )
            weight_bytes = estimate_weight_bytes(checkpoint_path)
            free_vram, _ = torch.cuda.mem_get_info(device)
            required_vram = int(weight_bytes * 1.2)
            if free_vram < required_vram:
                raise RuntimeError(
                    f"CUDA 可用显存 {free_vram / 1024**3:.1f} GiB，小于估算需求 "
                    f"{required_vram / 1024**3:.1f} GiB"
                )
        dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
        full_model = load_real_thinking_model(str(checkpoint_path), dtype=dtype, device=args.device)

    # 先确认目标目录可写（非空时必须 --force），再失效化全局证据：
    # 顺序反过来会出现“证据已删、模型一个都没换”的损坏中间态。
    package_dir = args.package_dir.expanduser().resolve()
    for component in components:
        safe_component_dir(package_dir, component, args.force)
    invalidate_package_evidence(package_dir)

    for component in components:
        case = (
            build_tiny_thinking_component(component, args.seed)
            if args.mode == "tiny"
            else build_real_thinking_component(full_model, component, args.seed)
        )
        output_dir = safe_component_dir(package_dir, component, args.force)
        export_component(case, output_dir, args.opset, args.seed, f"{args.mode}-fixed-shape")


if __name__ == "__main__":
    main()
