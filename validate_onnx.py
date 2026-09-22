from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

from qwen3_omni_onnx_cases import SUPPORTED_CASES, file_sha256, write_json
from qwen3_omni_thinking_components import THINKING_COMPONENTS

KNOWN_CASES = tuple(SUPPORTED_CASES) + tuple(THINKING_COMPONENTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验 Qwen3-Omni ONNX 的结构、Shape 和多组输入数值")
    parser.add_argument("--model", type=Path, required=True, help="ONNX 模型路径")
    parser.add_argument(
        "--case",
        choices=KNOWN_CASES,
        help="可选；必须与导出元数据一致。早期三级回归用 rmsnorm/moe_block/tiny_thinker，四组件用 vision_encoder/audio_encoder/thinker_prefill/thinker_decode",
    )
    parser.add_argument("--rtol", type=float, default=1e-4, help="相对误差阈值")
    parser.add_argument("--atol", type=float, default=1e-5, help="绝对误差阈值")
    parser.add_argument("--skip-shape-inference", action="store_true", help="跳过 Shape Inference")
    parser.add_argument(
        "--provider",
        action="append",
        dest="providers",
        help="ONNX Runtime provider；可重复指定，默认 CPUExecutionProvider",
    )
    return parser.parse_args()


def validate_tolerance(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} 必须是有限非负数，实际为 {value}")


def finite_and_error(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    shape_match = actual.shape == expected.shape
    dtype_match = actual.dtype == expected.dtype
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    if not shape_match:
        return {
            "shape_match": False,
            "dtype_match": dtype_match,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "actual_dtype": str(actual.dtype),
            "expected_dtype": str(expected.dtype),
            "finite": finite,
            "max_abs_error": None,
            "max_rel_error": None,
        }

    absolute_error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    denominator = np.maximum(np.abs(expected.astype(np.float64)), 1e-12)
    return {
        "shape_match": True,
        "dtype_match": dtype_match,
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
        "finite": finite,
        "max_abs_error": float(absolute_error.max(initial=0.0)),
        "max_rel_error": float((absolute_error / denominator).max(initial=0.0)),
    }


def count_unknown_shapes(model: onnx.ModelProto) -> dict[str, int]:
    unknown_tensors = 0
    unknown_dimensions = 0
    values = [*model.graph.input, *model.graph.output, *model.graph.value_info]
    for value in values:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            unknown_tensors += 1
            continue
        for dimension in tensor_type.shape.dim:
            if not dimension.HasField("dim_value") and not dimension.HasField("dim_param"):
                unknown_dimensions += 1
    return {"unknown_tensors": unknown_tensors, "unknown_dimensions": unknown_dimensions}


def load_metadata(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "export_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少导出元数据：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_vector(
    session: ort.InferenceSession,
    model_dir: Path,
    vector: dict[str, Any],
    expected_inputs: set[str],
    expected_outputs: set[str],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    inputs_path = (model_dir / vector["input_file"]).resolve()
    reference_path = (model_dir / vector["reference_output_file"]).resolve()
    if file_sha256(inputs_path) != vector["input_sha256"]:
        raise RuntimeError(f"输入文件哈希不匹配：{inputs_path}")
    if file_sha256(reference_path) != vector["reference_output_sha256"]:
        raise RuntimeError(f"参考输出哈希不匹配：{reference_path}")

    with np.load(inputs_path) as loaded_inputs:
        available_inputs = {name: loaded_inputs[name] for name in loaded_inputs.files}
    with np.load(reference_path) as loaded_reference:
        references = {name: loaded_reference[name] for name in loaded_reference.files}

    if set(available_inputs) != expected_inputs:
        raise KeyError(f"输入名称不一致：NPZ={sorted(available_inputs)} ONNX={sorted(expected_inputs)}")
    if set(references) != expected_outputs:
        raise KeyError(f"输出名称不一致：参考={sorted(references)} ONNX={sorted(expected_outputs)}")

    output_names = [item.name for item in session.get_outputs()]
    actual_outputs = session.run(output_names, available_inputs)
    comparisons: dict[str, dict[str, Any]] = {}
    vector_passed = True
    for name, actual in zip(output_names, actual_outputs):
        expected = references[name]
        stats = finite_and_error(actual, expected)
        output_close = bool(
            stats["shape_match"]
            and stats["dtype_match"]
            and stats["finite"]
            and np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False)
        )
        stats["allclose"] = output_close
        comparisons[name] = stats
        vector_passed = vector_passed and output_close

    return {
        "index": vector["index"],
        "passed": vector_passed,
        "inputs": str(inputs_path),
        "inputs_sha256": vector["input_sha256"],
        "reference_outputs": str(reference_path),
        "reference_outputs_sha256": vector["reference_output_sha256"],
        "routing": vector.get("routing"),
        "comparisons": comparisons,
    }


def main() -> None:
    args = parse_args()
    validate_tolerance("rtol", args.rtol)
    validate_tolerance("atol", args.atol)

    model_path = args.model.expanduser().resolve()
    model_dir = model_path.parent
    report_path = model_dir / "validation.json"
    report_path.unlink(missing_ok=True)
    started = time.perf_counter()

    try:
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        metadata = load_metadata(model_dir)
        if args.case and metadata["case"] != args.case:
            raise ValueError(f"case 不一致：参数={args.case} 元数据={metadata['case']}")
        model_hash = file_sha256(model_path)
        if model_hash != metadata["model_sha256"]:
            raise RuntimeError("ONNX 文件哈希与导出元数据不一致")

        onnx.checker.check_model(str(model_path), full_check=True)
        shape_inference: dict[str, Any] = {
            "attempted": not args.skip_shape_inference,
            "passed": bool(args.skip_shape_inference),
            "error": None,
            "unknown_tensors": None,
            "unknown_dimensions": None,
        }
        if not args.skip_shape_inference:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".shape_inferred_",
                suffix=".onnx",
                dir=model_dir,
            )
            os.close(descriptor)
            inferred_path = Path(temporary_name)
            inferred_path.unlink()
            try:
                onnx.shape_inference.infer_shapes_path(
                    str(model_path),
                    str(inferred_path),
                    check_type=True,
                    strict_mode=True,
                    data_prop=False,
                )
                onnx.checker.check_model(str(inferred_path), full_check=True)
                inferred = onnx.load(str(inferred_path), load_external_data=False)
                shape_inference.update({"passed": True, **count_unknown_shapes(inferred)})
            finally:
                inferred_path.unlink(missing_ok=True)

        requested_providers = args.providers or ["CPUExecutionProvider"]
        unavailable = sorted(set(requested_providers) - set(ort.get_available_providers()))
        if unavailable:
            raise RuntimeError(
                f"ONNX Runtime provider 不可用：{unavailable}；当前可用：{ort.get_available_providers()}"
            )
        session = ort.InferenceSession(str(model_path), providers=requested_providers)
        session_input_names = {item.name for item in session.get_inputs()}
        session_output_names = {item.name for item in session.get_outputs()}
        if session_input_names != set(metadata["input_names"]):
            raise RuntimeError("ONNX 输入与导出元数据不一致")
        if session_output_names != set(metadata["output_names"]):
            raise RuntimeError("ONNX 输出与导出元数据不一致")

        vector_reports = [
            validate_vector(
                session,
                model_dir,
                vector,
                session_input_names,
                session_output_names,
                args.rtol,
                args.atol,
            )
            for vector in metadata["test_vectors"]
        ]
        routing_hashes = {
            vector["routing"]["sha256"]
            for vector in metadata["test_vectors"]
            if vector.get("routing")
        }
        routing_required = metadata["case"] in {
            "moe_block",
            "tiny_thinker",
            "thinker_prefill",
            "thinker_decode",
        }
        routing_coverage = not routing_required or len(routing_hashes) >= 2
        passed = bool(
            shape_inference["passed"]
            and all(vector["passed"] for vector in vector_reports)
            and routing_coverage
        )
        report = {
            "passed": passed,
            "case": metadata["case"],
            "model": str(model_path),
            "model_sha256": model_hash,
            "checker": "passed",
            "shape_inference": shape_inference,
            "onnxruntime": {
                "version": ort.__version__,
                "providers": session.get_providers(),
                "input_names": sorted(session_input_names),
                "output_names": sorted(session_output_names),
            },
            "tolerances": {"rtol": args.rtol, "atol": args.atol},
            "routing_coverage": {
                "required": routing_required,
                "distinct_patterns": len(routing_hashes),
                "passed": routing_coverage,
            },
            "test_vectors": vector_reports,
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(report_path, report)
    except Exception as error:
        write_json(
            report_path,
            {
                "passed": False,
                "model": str(model_path),
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise

    for vector in report["test_vectors"]:
        for name, stats in vector["comparisons"].items():
            print(
                f"[{('OK' if stats['allclose'] else 'FAIL')}] vector={vector['index']} {name}: "
                f"max_abs={stats['max_abs_error']:.6g}, max_rel={stats['max_rel_error']:.6g}"
            )
    print(f"[OK] distinct_routing_patterns={report['routing_coverage']['distinct_patterns']}")
    print(f"[{'OK' if report['passed'] else 'FAIL'}] report={report_path}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
