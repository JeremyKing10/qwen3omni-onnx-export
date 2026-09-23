from __future__ import annotations

import argparse
import csv
import io
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import onnx

from inspect_onnx import STANDARD_DOMAINS, iter_graph_nodes, value_info_dict
from onnx_artifact_utils import file_sha256, iter_messages, safe_path, verify_artifact_identity
from qwen3_omni_onnx_cases import WORKSPACE
from qwen3_omni_thinking_components import THINKING_COMPONENTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 Thinking 四组件 ONNX 算子清单")
    parser.add_argument("--package-dir", type=Path, required=True)
    return parser.parse_args()


def atomic_text(root: Path, path: Path, text: str) -> None:
    path = safe_path(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, safe_path(root, path))
    finally:
        temporary.unlink(missing_ok=True)


def collect_operator_report(package_dir: Path, component: str) -> dict[str, Any]:
    if component not in THINKING_COMPONENTS:
        raise ValueError(f"未知组件：{component}")
    report_path = safe_path(
        package_dir, package_dir / "operators" / f"{component.replace('_encoder', '')}.json", must_exist=True
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("passed") is not True or report.get("custom_domains") != []:
        raise RuntimeError(f"组件算子检查未通过或包含自定义 domain：{component}")
    model_path = safe_path(package_dir, package_dir / "onnx" / component / "model.onnx", must_exist=True)
    identity = verify_artifact_identity(model_path, report.get("artifact_identity"))
    if identity.get("case") != component or report.get("model_sha256") != identity["model_sha256"]:
        raise RuntimeError(f"{component} 算子报告组件身份或模型哈希不一致")
    external = report.get("external_data", {})
    files = external.get("files")
    if not isinstance(files, list) or external.get("errors") != [] or external.get("all_files_present_and_valid") is not True:
        raise RuntimeError(f"{component} 缺少完整 external data 检查")
    recorded_external = [
        {key: entry.get(key) for key in ("location", "bytes", "sha256")}
        for entry in files
    ]
    if sorted(recorded_external, key=lambda item: item["location"]) != identity["external_data"]:
        raise RuntimeError(f"{component} external data 集合或哈希不一致")
    if external.get("used") is not bool(files) or any(
        entry.get("exists") is not True or entry.get("all_ranges_valid") is not True for entry in files
    ):
        raise RuntimeError(f"{component} external data 判据不完整")

    model = onnx.load(str(model_path), load_external_data=False)
    graph_nodes = list(iter_graph_nodes(model.graph))
    function_nodes = [node for function in model.functions for node in iter_messages(function, onnx.NodeProto)]
    counts = Counter(((node.domain or "ai.onnx"), node.op_type) for node in [*graph_nodes, *function_nodes])
    domains = {domain for domain, _ in counts} | {item.domain or "ai.onnx" for item in model.opset_import}
    domains.update(item.domain or "ai.onnx" for function in model.functions for item in function.opset_import)
    if domains - STANDARD_DOMAINS:
        raise RuntimeError(f"{component} 当前图包含自定义 domain")
    graph = report.get("graph", {})
    expected_sizes = {"node_count": len(graph_nodes), "function_node_count": len(function_nodes)}
    if any(type(graph.get(key)) is not int or graph[key] != size for key, size in expected_sizes.items()):
        raise RuntimeError(f"{component} 节点计数与当前 ONNX 不一致")
    initializer_names = {item.name for item in model.graph.initializer}
    if graph.get("inputs") != [value_info_dict(item) for item in model.graph.input if item.name not in initializer_names]:
        raise RuntimeError(f"{component} 算子报告输入签名不一致")
    if graph.get("outputs") != [value_info_dict(item) for item in model.graph.output]:
        raise RuntimeError(f"{component} 算子报告输出签名不一致")
    recorded_counts = report.get("operator_counts")
    if not isinstance(recorded_counts, list):
        raise RuntimeError(f"{component} 缺少算子计数")
    recorded: dict[tuple[str, str], int] = {}
    for item in recorded_counts:
        key = (item["domain"], item["op_type"])
        count = item.get("count")
        if key in recorded or type(count) is not int or count <= 0:
            raise RuntimeError(f"{component} 算子计数无效或重复")
        recorded[key] = count
    if recorded != dict(counts) or sum(recorded.values()) != len(graph_nodes) + len(function_nodes):
        raise RuntimeError(f"{component} 算子数量与完整图节点数量不一致")
    return report


def main() -> None:
    args = parse_args()
    package_dir = safe_path(WORKSPACE, args.package_dir.expanduser().absolute())
    if package_dir == WORKSPACE:
        raise ValueError("产品目录不能是工作区根目录")
    operators_dir = safe_path(package_dir, package_dir / "operators")
    summary_path = safe_path(package_dir, operators_dir / "summary.json")
    csv_path = operators_dir / "all_operators.csv"
    operators_dir.mkdir(parents=True, exist_ok=True)
    atomic_text(package_dir, summary_path, json.dumps({"passed": False, "status": "aggregation-in-progress"}) + "\n")
    safe_path(package_dir, csv_path).unlink(missing_ok=True)
    rows: list[dict[str, str | int]] = []
    totals: Counter[tuple[str, str]] = Counter()
    component_summaries = {}
    identities = {}
    try:
        for component in THINKING_COMPONENTS:
            report = collect_operator_report(package_dir, component)
            graph_count = report["graph"]["node_count"]
            function_count = report["graph"]["function_node_count"]
            identities[component] = report["artifact_identity"]
            component_summaries[component] = {
                "node_count": graph_count + function_count,
                "graph_node_count": graph_count,
                "function_node_count": function_count,
                "custom_domains": [],
                "report": f"operators/{component.replace('_encoder', '')}.json",
                "artifact_identity": report["artifact_identity"],
            }
            for item in report["operator_counts"]:
                key = (item["domain"], item["op_type"])
                totals[key] += item["count"]
                rows.append({"component": component, **item})
        for (domain, op_type), count in sorted(totals.items()):
            rows.append({"component": "ALL", "domain": domain, "op_type": op_type, "count": count})
        content = io.StringIO(newline="")
        writer = csv.DictWriter(content, fieldnames=("component", "domain", "op_type", "count"))
        writer.writeheader()
        writer.writerows(rows)
        summary = {
            "schema_version": 2,
            "passed": True,
            "artifact_identities": identities,
            "components": component_summaries,
            "total_node_count": sum(item["node_count"] for item in component_summaries.values()),
            "unique_operators": [
                {"domain": domain, "op_type": op_type, "count": count}
                for (domain, op_type), count in sorted(totals.items())
            ],
            "csv": str(csv_path.relative_to(package_dir)),
        }
        atomic_text(package_dir, csv_path, content.getvalue())
        summary["csv_sha256"] = file_sha256(safe_path(package_dir, csv_path, must_exist=True))
        atomic_text(package_dir, summary_path, json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    except Exception as error:
        safe_path(package_dir, csv_path).unlink(missing_ok=True)
        atomic_text(package_dir, summary_path, json.dumps({"passed": False, "status": "aggregation-failed", "error": str(error)}, ensure_ascii=False) + "\n")
        raise
    print(f"[OK] operators={csv_path}")
    print(f"[OK] total_nodes={summary['total_node_count']}")
    print(f"[OK] unique_operators={len(totals)}")


if __name__ == "__main__":
    main()
