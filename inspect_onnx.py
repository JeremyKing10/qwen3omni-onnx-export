from __future__ import annotations

import argparse
import collections
from pathlib import Path
from typing import Any, Iterable

import onnx

from qwen3_omni_onnx_cases import WORKSPACE, file_sha256, write_json
from onnx_artifact_utils import artifact_identity, external_data_files, iter_messages, safe_path

STANDARD_DOMAINS = {"", "ai.onnx", "ai.onnx.ml"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 Qwen3-Omni ONNX 接口、算子、domain 和 external data")
    parser.add_argument("--model", type=Path, required=True, help="ONNX 模型路径")
    parser.add_argument("--output", type=Path, help="JSON 报告路径；默认写入模型同目录 operators.json")
    parser.add_argument("--fail-on-custom-domain", action="store_true", help="发现非标准 domain 时返回失败")
    return parser.parse_args()


def shape_of(value_info: onnx.ValueInfoProto) -> list[int | str | None]:
    tensor_type = value_info.type.tensor_type
    shape: list[int | str | None] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(dimension.dim_value)
        elif dimension.HasField("dim_param"):
            shape.append(dimension.dim_param)
        else:
            shape.append(None)
    return shape


def value_info_dict(value_info: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor_type = value_info.type.tensor_type
    return {
        "name": value_info.name,
        "dtype": onnx.TensorProto.DataType.Name(tensor_type.elem_type),
        "shape": shape_of(value_info),
    }


def iter_graph_nodes(graph: onnx.GraphProto) -> Iterable[onnx.NodeProto]:
    for node in graph.node:
        yield node
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from iter_graph_nodes(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    yield from iter_graph_nodes(subgraph)


def safe_external_path(model_dir: Path, location: str) -> Path:
    candidate = Path(location)
    if not location or candidate.is_absolute():
        raise ValueError("external data location 必须是相对路径")
    return safe_path(model_dir, model_dir / candidate, must_exist=True)


def main() -> None:
    args = parse_args()
    model_path = safe_path(WORKSPACE, args.model, must_exist=True)
    output_path = safe_path(WORKSPACE, args.output or model_path.parent / "operators.json")
    if output_path.suffix != ".json" or output_path.name in {"export_metadata.json", "validation.json", "manifest.json"}:
        raise ValueError("算子报告必须使用独立 JSON 文件，不能覆盖模型、元数据或验证报告")

    model = onnx.load(str(model_path), load_external_data=False)
    external_files, external_errors = external_data_files(model, model_path)
    external_paths = {Path(item["resolved_path"]) for item in external_files}
    for tensor in iter_messages(model, onnx.TensorProto):
        for entry in tensor.external_data:
            if entry.key == "location" and entry.value:
                location = Path(entry.value)
                if not location.is_absolute() and ".." not in location.parts:
                    external_paths.add(safe_path(model_path.parent, model_path.parent / location))
    if output_path in external_paths:
        raise ValueError("JSON 报告不能覆盖 ONNX external data 文件，即使该权重文件目前缺失")

    nodes = list(iter_graph_nodes(model.graph))
    function_nodes = [node for function in model.functions for node in iter_messages(function, onnx.NodeProto)]
    all_nodes = [*nodes, *function_nodes]
    operator_counts = collections.Counter(((node.domain or "ai.onnx"), node.op_type) for node in all_nodes)
    domain_counts = collections.Counter((node.domain or "ai.onnx") for node in all_nodes)
    opset_domains = {item.domain or "ai.onnx" for item in model.opset_import}
    custom_domains = sorted(
        domain
        for domain in (set(domain_counts) | opset_domains)
        if domain not in STANDARD_DOMAINS
    )

    initializer_names = {initializer.name for initializer in model.graph.initializer}
    graph_inputs = [item for item in model.graph.input if item.name not in initializer_names]
    external_valid = not external_errors and all(
        item["exists"] and item["all_ranges_valid"] for item in external_files
    )
    identity = None
    if external_valid and (model_path.parent / "export_metadata.json").is_file():
        identity = artifact_identity(model_path, scanned_external=external_files)
    report = {
        "passed": bool(external_valid and (not args.fail_on_custom_domain or not custom_domains)),
        "artifact_identity": identity,
        "model": str(model_path),
        "model_bytes": model_path.stat().st_size,
        "model_sha256": file_sha256(model_path),
        "ir_version": model.ir_version,
        "producer": {"name": model.producer_name, "version": model.producer_version},
        "opset_imports": {item.domain or "ai.onnx": item.version for item in model.opset_import},
        "graph": {
            "name": model.graph.name,
            "node_count": len(nodes),
            "function_node_count": len(function_nodes),
            "initializer_count": len(model.graph.initializer),
            "inputs": [value_info_dict(item) for item in graph_inputs],
            "outputs": [value_info_dict(item) for item in model.graph.output],
        },
        "operator_counts": [
            {"domain": domain, "op_type": op_type, "count": count}
            for (domain, op_type), count in sorted(operator_counts.items())
        ],
        "domain_counts": dict(sorted(domain_counts.items())),
        "custom_domains": custom_domains,
        "external_data": {
            "used": any(t.data_location == onnx.TensorProto.EXTERNAL or t.external_data for t in iter_messages(model, onnx.TensorProto)),
            "files": external_files,
            "errors": external_errors,
            "all_files_present_and_valid": external_valid,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, report)

    print(f"[{'OK' if report['passed'] else 'FAIL'}] model={model_path}")
    print(f"[OK] nodes={report['graph']['node_count']}")
    print(f"[OK] custom_domains={custom_domains or 'none'}")
    for item in report["operator_counts"]:
        print(f"{item['domain']}::{item['op_type']} = {item['count']}")
    print(f"[OK] report={output_path}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
