from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from qwen3_omni_onnx_cases import file_sha256, write_json
from qwen3_omni_thinking_components import THINKING_COMPONENTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 Thinking 四组件 ONNX 算子清单")
    parser.add_argument("--package-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_dir = args.package_dir.expanduser().resolve()
    operators_dir = package_dir / "operators"
    operators_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []
    totals: Counter[tuple[str, str]] = Counter()
    component_summaries = {}

    for component in THINKING_COMPONENTS:
        report_path = operators_dir / f"{component.replace('_encoder', '')}.json"
        if component == "thinker_prefill":
            report_path = operators_dir / "thinker_prefill.json"
        elif component == "thinker_decode":
            report_path = operators_dir / "thinker_decode.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"缺少组件算子报告：{report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not report.get("passed"):
            raise RuntimeError(f"组件算子检查未通过：{component}")
        model_path = package_dir / "onnx" / component / "model.onnx"
        if report.get("model_sha256") != file_sha256(model_path):
            raise RuntimeError(f"{component} 算子报告与当前 ONNX 哈希不一致")
        if report.get("custom_domains"):
            raise RuntimeError(f"{component} 包含自定义 domain：{report['custom_domains']}")
        component_summaries[component] = {
            "node_count": report["graph"]["node_count"],
            "custom_domains": report["custom_domains"],
            "report": str(report_path.relative_to(package_dir)),
        }
        for item in report["operator_counts"]:
            key = (item["domain"], item["op_type"])
            totals[key] += item["count"]
            rows.append(
                {
                    "component": component,
                    "domain": item["domain"],
                    "op_type": item["op_type"],
                    "count": item["count"],
                }
            )

    for (domain, op_type), count in sorted(totals.items()):
        rows.append({"component": "ALL", "domain": domain, "op_type": op_type, "count": count})

    csv_path = operators_dir / "all_operators.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("component", "domain", "op_type", "count"))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "components": component_summaries,
        "total_node_count": sum(item["node_count"] for item in component_summaries.values()),
        "unique_operators": [
            {"domain": domain, "op_type": op_type, "count": count}
            for (domain, op_type), count in sorted(totals.items())
        ],
        "csv": str(csv_path.relative_to(package_dir)),
    }
    write_json(operators_dir / "summary.json", summary)
    print(f"[OK] operators={csv_path}")
    print(f"[OK] total_nodes={summary['total_node_count']}")
    print(f"[OK] unique_operators={len(totals)}")


if __name__ == "__main__":
    main()
