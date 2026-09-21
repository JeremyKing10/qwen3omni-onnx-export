from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import onnx
import torch

from qwen3_omni_onnx_cases import (
    DEFAULT_SEED,
    SUPPORTED_CASES,
    WORKSPACE,
    build_case,
    capture_routing,
    file_sha256,
    normalize_outputs,
    runtime_metadata,
    tensor_dict,
    write_json,
)

ARTIFACTS_ROOT = (WORKSPACE / "artifacts").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出可验证的 Qwen3-Omni ONNX 分级模型")
    parser.add_argument("--case", choices=SUPPORTED_CASES, required=True, help="要导出的模型级别")
    parser.add_argument("--output-dir", type=Path, required=True, help="artifacts 下的导出目录")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset，默认 18")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")
    parser.add_argument("--force", action="store_true", help="覆盖已有导出目录")
    return parser.parse_args()


def prepare_output_dir(path: Path, force: bool) -> Path:
    path = path.expanduser().resolve()
    try:
        relative = path.relative_to(ARTIFACTS_ROOT)
    except ValueError as error:
        raise ValueError(f"输出目录必须位于 {ARTIFACTS_ROOT} 下，实际为 {path}") from error
    if path == ARTIFACTS_ROOT or len(relative.parts) != 1:
        raise ValueError(f"输出目录必须是 {ARTIFACTS_ROOT} 的直接子目录")
    if path.is_symlink():
        raise ValueError(f"拒绝使用符号链接输出目录：{path}")
    if path.exists() and not path.is_dir():
        raise ValueError(f"输出路径不是目录：{path}")
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(f"{path} 非空；如需覆盖请添加 --force")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    args = parse_args()
    if args.opset < 18:
        raise ValueError("本工具使用 dynamo exporter，opset 必须不小于 18")
    output_dir = prepare_output_dir(args.output_dir, args.force)
    model_path = output_dir / "model.onnx"

    case = build_case(args.case, args.seed)
    case.model.requires_grad_(False)
    vector_metadata = []
    routing_hashes = set()

    for index, vector in enumerate(case.test_vectors):
        with torch.inference_mode():
            reference = normalize_outputs(case.model(*vector))
        input_name = f"inputs_{index}.npz"
        reference_name = f"reference_outputs_{index}.npz"
        np.savez(output_dir / input_name, **tensor_dict(case.input_names, vector))
        np.savez(output_dir / reference_name, **tensor_dict(case.output_names, reference))
        routing = capture_routing(case.model, vector)
        if routing:
            routing_hashes.add(routing["sha256"])
        vector_metadata.append(
            {
                "index": index,
                "input_file": input_name,
                "reference_output_file": reference_name,
                "input_sha256": file_sha256(output_dir / input_name),
                "reference_output_sha256": file_sha256(output_dir / reference_name),
                "routing": routing,
            }
        )

    if args.case in {"moe_block", "tiny_thinker"} and len(routing_hashes) < 2:
        raise RuntimeError("两组测试输入没有产生不同的 MoE 路由，拒绝导出以避免假阳性")

    started = time.perf_counter()
    with torch.inference_mode():
        torch.onnx.export(
            case.model,
            case.export_args,
            model_path,
            input_names=list(case.input_names),
            output_names=list(case.output_names),
            opset_version=args.opset,
            dynamo=True,
            external_data=True,
            optimize=True,
            export_params=True,
        )
    elapsed = time.perf_counter() - started

    onnx.checker.check_model(str(model_path), full_check=True)
    graph = onnx.load(str(model_path), load_external_data=False)

    metadata = {
        "case": case.name,
        "description": case.description,
        "model_file": model_path.name,
        "model_sha256": file_sha256(model_path),
        "test_vectors": vector_metadata,
        "input_names": list(case.input_names),
        "output_names": list(case.output_names),
        "opset_requested": args.opset,
        "opset_imports": {item.domain or "ai.onnx": item.version for item in graph.opset_import},
        "seed": args.seed,
        "fixed_shapes": True,
        "attention_implementation": case.config.get("attn_implementation", "eager"),
        "experts_implementation": "batched_mm",
        "uses_external_data": any(initializer.external_data for initializer in graph.graph.initializer),
        "export_seconds": elapsed,
        "model_bytes": model_path.stat().st_size,
        "config": case.config,
        "runtime": runtime_metadata(),
    }
    write_json(output_dir / "export_metadata.json", metadata)

    print(f"[OK] case={case.name}")
    print(f"[OK] ONNX={model_path}")
    print(f"[OK] test_vectors={len(vector_metadata)}")
    print(f"[OK] distinct_routing_patterns={len(routing_hashes)}")
    print(f"[OK] nodes={len(graph.graph.node)}")
    print(f"[OK] export_seconds={elapsed:.3f}")


if __name__ == "__main__":
    main()
