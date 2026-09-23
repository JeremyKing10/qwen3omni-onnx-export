from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterator

import ml_dtypes
import numpy as np
import onnx
import onnxruntime as ort
from google.protobuf.message import Message

EVIDENCE_SCHEMA_VERSION = 2
WORKSPACE = Path(__file__).resolve().parent
DEFAULT_RTOL = 1e-4
DEFAULT_ATOL = 1e-5


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def safe_path(root: Path, path: Path, *, must_exist: bool = False) -> Path:
    root = Path(root).expanduser().absolute()
    path = Path(path).expanduser().absolute()
    if ".." in path.parts or ".." in root.parts:
        raise ValueError(f"拒绝包含父目录跳转的路径：{path}")
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"路径越出允许目录 {root}：{path}") from error
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"拒绝符号链接路径：{current}")
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"拒绝符号链接路径：{current}")
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(f"缺少普通文件：{resolved}")
    return resolved


def iter_messages(message: Message, wanted: type) -> Iterator[Any]:
    if isinstance(message, wanted):
        yield message
    for field, value in message.ListFields():
        if field.message_type is None:
            continue
        if field.is_repeated:
            for child in value:
                yield from iter_messages(child, wanted)
        else:
            yield from iter_messages(value, wanted)


def _tensor_nbytes(tensor: onnx.TensorProto) -> int:
    if any(dim < 0 for dim in tensor.dims):
        raise ValueError("initializer 维度不能为负")
    count = math.prod(tensor.dims)
    four_bit = {getattr(onnx.TensorProto, name, -1) for name in ("INT4", "UINT4", "FLOAT4E2M1")}
    if tensor.data_type in four_bit:
        return (count + 1) // 2
    if tensor.data_type == onnx.TensorProto.STRING:
        raise ValueError("不支持 external STRING tensor 的字节范围验证")
    dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type))
    return count * dtype.itemsize


def external_data_files(model: onnx.ModelProto, model_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    model_dir = Path(model_path).absolute().parent
    files: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for tensor in iter_messages(model, onnx.TensorProto):
        if tensor.data_location != onnx.TensorProto.EXTERNAL and not tensor.external_data:
            continue
        entry = None
        recorded = False
        try:
            values = {item.key: item.value for item in tensor.external_data}
            if len(values) != len(tensor.external_data):
                raise ValueError("external data 属性重复")
            location = values.get("location", "")
            relative = Path(location)
            if not location or relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"不安全的 external data location：{location!r}")
            file_path = safe_path(model_dir, model_dir / relative)
            location = file_path.relative_to(model_dir.resolve()).as_posix()
            if location not in files:
                exists = file_path.is_file()
                files[location] = {
                    "location": location,
                    "resolved_path": str(file_path),
                    "exists": exists,
                    "bytes": file_path.stat().st_size if exists else None,
                    "sha256": file_sha256(file_path) if exists else None,
                    "all_ranges_valid": exists,
                    "initializers": [],
                }
            entry = files[location]
            if not entry["exists"]:
                raise FileNotFoundError(f"external data 文件缺失：{location}")
            offset = int(values.get("offset", "0"))
            length = int(values["length"]) if "length" in values else None
            required = _tensor_nbytes(tensor)
            available = entry["bytes"] - offset if length is None else length
            valid = (
                0 <= offset <= entry["bytes"]
                and available >= required
                and offset + available <= entry["bytes"]
            )
            entry["all_ranges_valid"] = entry["all_ranges_valid"] and valid
            entry["initializers"].append({
                "name": tensor.name, "offset": offset, "length": length,
                "required_bytes": required, "range_valid": valid,
            })
            recorded = True
            if not valid:
                errors.append(f"{tensor.name}: external data offset/length 越界或数据不足")
        except (ValueError, TypeError, KeyError, OSError) as error:
            if entry is not None:
                entry["all_ranges_valid"] = False
                if not recorded:
                    entry["initializers"].append({"name": tensor.name, "range_valid": False, "error": str(error)})
            errors.append(f"{tensor.name}: {error}")
    return sorted(files.values(), key=lambda item: item["location"]), errors


def model_identity(
    model_path: Path, *, scanned_external: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    model_path = Path(model_path).absolute()
    safe_path(model_path.parent, model_path, must_exist=True)
    if scanned_external is None:
        graph = onnx.load(str(model_path), load_external_data=False)
        files, errors = external_data_files(graph, model_path)
        if errors:
            raise ValueError("external data 不完整：" + "; ".join(errors))
    else:
        files = scanned_external
        if not all(item["all_ranges_valid"] for item in files):
            raise ValueError("external data 扫描未通过")
    payload = {
        "model_sha256": file_sha256(model_path),
        "external_data": [
            {key: item[key] for key in ("location", "bytes", "sha256")}
            for item in files
        ],
    }
    return {**payload, "digest": canonical_digest(payload)}


def artifact_identity(
    model_path: Path, *, scanned_external: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    model_path = Path(model_path).absolute()
    root = model_path.parent
    metadata_path = safe_path(root, root / "export_metadata.json", must_exist=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("evidence_schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise ValueError("导出元数据不是 evidence schema v2；请重新导出，不能沿用旧验证证据")
    current = model_identity(model_path, scanned_external=scanned_external)
    if metadata.get("model_identity") != current or metadata.get("model_sha256") != current["model_sha256"]:
        raise RuntimeError("ONNX 或 external data 与导出时的模型指纹不一致")
    vectors = metadata.get("test_vectors")
    if not isinstance(vectors, list) or len(vectors) < 2:
        raise ValueError("至少需要两组非空测试向量")
    evidence_vectors = []
    seen = set()
    for vector in vectors:
        index = vector["index"]
        if index in seen:
            raise ValueError("测试向量 index 重复")
        seen.add(index)
        entry = {"index": index}
        for field, digest_field in (("input_file", "input_sha256"), ("reference_output_file", "reference_output_sha256")):
            name = vector[field]
            relative = Path(name)
            if relative.is_absolute() or len(relative.parts) != 1:
                raise ValueError(f"测试向量文件名不安全：{name}")
            path = safe_path(root, root / relative, must_exist=True)
            digest = file_sha256(path)
            if digest != vector[digest_field]:
                raise RuntimeError(f"测试向量内容哈希不一致：{name}")
            entry[field] = name
            entry[digest_field] = digest
        evidence_vectors.append(entry)
    payload = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "case": metadata["case"],
        "model_sha256": current["model_sha256"],
        "external_data": current["external_data"],
        "metadata_sha256": file_sha256(metadata_path),
        "test_vectors": evidence_vectors,
    }
    return {**payload, "digest": canonical_digest(payload)}


def verify_artifact_identity(model_path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    current = artifact_identity(model_path)
    if current != expected:
        raise RuntimeError(f"证据与当前模型/权重/元数据/向量不一致：{model_path}")
    return current


def validate_tolerances(rtol: float, atol: float) -> None:
    for name, value in (("rtol", rtol), ("atol", atol)):
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} 必须为有限非负数，实际为 {value}")


def check_providers(providers: list[str]) -> None:
    if not providers:
        raise ValueError("必须指定至少一个 ONNX Runtime provider")
    unavailable = set(providers) - set(ort.get_available_providers())
    if unavailable:
        raise RuntimeError(f"ORT provider 不可用：{sorted(unavailable)}；可用：{ort.get_available_providers()}")


def require_strict_validation(report: dict[str, Any], identity: dict[str, Any]) -> None:
    if report.get("passed") is not True or report.get("artifact_identity") != identity:
        raise RuntimeError("数值验证报告未通过或身份与当前产物不一致")
    if report.get("case") != identity["case"] or report.get("model_sha256") != identity["model_sha256"]:
        raise RuntimeError("数值验证报告组件身份不一致")
    shape = report.get("shape_inference", {})
    if not (report.get("checker") == "passed" and shape.get("attempted") is True and shape.get("passed") is True
            and shape.get("unknown_tensors") == 0 and shape.get("unknown_dimensions") == 0):
        raise RuntimeError("严格验收必须完成 Checker 和无未知维度的 Shape Inference")
    tolerances = report.get("tolerances", {})
    rtol, atol = tolerances.get("rtol", float("inf")), tolerances.get("atol", float("inf"))
    validate_tolerances(rtol, atol)
    if rtol > DEFAULT_RTOL or atol > DEFAULT_ATOL:
        raise RuntimeError("诊断验证容差宽于严格验收基线，不能作为正式通过证据")
    vectors = report.get("test_vectors", [])
    if len(vectors) != len(identity["test_vectors"]) or len(vectors) < 2:
        raise RuntimeError("数值验证未覆盖全部测试向量")
    for vector, expected in zip(vectors, identity["test_vectors"]):
        if not (vector.get("passed") is True and vector.get("index") == expected["index"]
                and vector.get("inputs_sha256") == expected["input_sha256"]
                and vector.get("reference_outputs_sha256") == expected["reference_output_sha256"]):
            raise RuntimeError("数值报告中的测试向量身份或状态不一致")
        comparisons = vector.get("comparisons", {})
        output_names = report.get("onnxruntime", {}).get("output_names", [])
        if not comparisons or set(comparisons) != set(output_names) or not all(
            item.get("allclose") is True and item.get("shape_match") is True
            and item.get("dtype_match") is True and item.get("finite") is True
            and isinstance(item.get("max_abs_error"), (int, float))
            and math.isfinite(item["max_abs_error"]) and item["max_abs_error"] >= 0
            for item in comparisons.values()
        ):
            raise RuntimeError("数值报告缺少完整、有限且通过的逐输出比较")
    if report.get("routing_coverage", {}).get("passed") is not True:
        raise RuntimeError("MoE 路由覆盖检查未通过")


def save_tensor_archive(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, str]:
    if not arrays:
        raise ValueError("测试向量不能是空字典")
    descriptors: dict[str, str] = {}
    stored = {}
    for name, value in arrays.items():
        value = np.asarray(value)
        descriptors[name] = str(value.dtype)
        if value.dtype == np.dtype(ml_dtypes.bfloat16):
            stored[name] = value.view(np.uint16)
        elif value.dtype.kind in "biuf":
            stored[name] = value
        else:
            raise TypeError(f"不支持序列化的 tensor dtype：{name}={value.dtype}")
    with Path(path).open("wb") as handle:
        np.savez(handle, **stored)
    return descriptors


def load_tensor_archive(path: Path, descriptors: dict[str, str] | None = None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if descriptors is not None and set(descriptors) != set(archive.files):
            raise ValueError("NPZ tensor 名称与 dtype 描述不一致")
        result = {}
        for name in archive.files:
            value = archive[name]
            target = descriptors.get(name) if descriptors is not None else None
            if target == "bfloat16":
                if value.dtype != np.uint16:
                    raise TypeError("BF16 存储必须为 uint16 位模式")
                value = value.view(ml_dtypes.bfloat16)
            elif value.dtype.kind not in "biuf" or (target is not None and str(value.dtype) != target):
                raise TypeError(f"NPZ dtype 不一致或缺少可恢复的类型描述：{name}={value.dtype}")
            result[name] = value
    return result


def run_ort_session(
    session: ort.InferenceSession, feed: dict[str, np.ndarray], output_names: list[str] | None = None
) -> tuple[np.ndarray, ...]:
    inputs = {item.name: item for item in session.get_inputs()}
    if set(feed) != set(inputs):
        raise ValueError("ORT 输入名称与图签名不一致")
    outputs = session.get_outputs()
    names = output_names if output_names is not None else [item.name for item in outputs]
    if len(set(names)) != len(names) or not set(names) <= {item.name for item in outputs}:
        raise ValueError("ORT 输出名称无效")
    contains_bf16 = any(item.type == "tensor(bfloat16)" for item in (*inputs.values(), *outputs))
    if not contains_bf16:
        return tuple(session.run(names, feed))
    keep_alive: list[np.ndarray] = []
    binding = session.io_binding()
    for name, item in inputs.items():
        original = np.asarray(feed[name])
        array = np.ascontiguousarray(original).reshape(original.shape)
        if item.type == "tensor(bfloat16)":
            if array.dtype != np.dtype(ml_dtypes.bfloat16):
                raise TypeError(f"{name} 需要 BF16 数值数组，不可把 float32 或普通 uint16 当作 BF16")
            element_type = onnx.TensorProto.BFLOAT16
        else:
            element_type = onnx.helper.np_dtype_to_tensor_dtype(array.dtype)
        keep_alive.append(array)
        binding.bind_input(name, "cpu", 0, element_type, array.shape, array.ctypes.data)
    for name in names:
        binding.bind_output(name, "cpu")
    session.run_with_iobinding(binding)
    results = []
    for value in binding.get_outputs():
        if value.data_type() == "tensor(bfloat16)":
            shape = tuple(value.shape())
            count = math.prod(shape)
            if count:
                storage = np.ctypeslib.as_array((ctypes.c_uint16 * count).from_address(value.data_ptr())).copy()
            else:
                storage = np.empty(0, dtype=np.uint16)
            results.append(storage.view(ml_dtypes.bfloat16).reshape(shape))
        else:
            results.append(value.numpy())
    return tuple(results)


def source_snapshot() -> dict[str, Any]:
    files = sorted(WORKSPACE.glob("*.py")) + [WORKSPACE / "requirements.txt"]
    versions = {}
    for name in ("torch", "transformers", "onnx", "onnxruntime", "onnxscript", "numpy", "ml_dtypes"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = subprocess.run(["git", "-C", str(WORKSPACE), "rev-parse", "HEAD"], capture_output=True, text=True)
    status = subprocess.run(["git", "-C", str(WORKSPACE), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True)
    return {
        "files": {path.name: file_sha256(path) for path in files if path.is_file()},
        "git_revision": result.stdout.strip() if result.returncode == 0 else None,
        "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "python": platform.python_version(), "platform": platform.platform(), "packages": versions,
        "scope": "source bytes and installed versions observed at export; not a signature or reproducible wheel lock",
    }
