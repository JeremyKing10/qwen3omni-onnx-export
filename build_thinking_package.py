from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from qwen3_omni_onnx_cases import WORKSPACE, file_sha256, write_json
from qwen3_omni_thinking_components import THINKING_COMPONENTS

DEFAULT_REPO = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
DEFAULT_REVISION = "2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b"
DEFAULT_PACKAGE = WORKSPACE / "Qwen3-Omni-30B-A3B-Thinking-ONNX"
PROCESSOR_FILES = (
    "chat_template.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
)
ROOT_CONFIG_FILES = ("config.json", "generation_config.json")
TOOL_FILES = (
    "qwen3_omni_onnx_cases.py",
    "qwen3_omni_thinking_components.py",
    "export_onnx.py",
    "export_thinking_onnx.py",
    "validate_onnx.py",
    "inspect_onnx.py",
    "validate_thinking_pipeline.py",
    "aggregate_operators.py",
    "build_thinking_package.py",
    "run_local_thinking_pipeline.py",
    "run_real_thinking_pipeline.py",
    "portable-export-requirements-lock.txt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整理 Qwen3-Omni Thinking ONNX 产品目录")
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--source-dir", type=Path, help="优先从本地 checkpoint 复制配置和 Processor 文件")
    parser.add_argument("--offline", action="store_true", help="只使用本地文件或缓存，不访问网络")
    return parser.parse_args()


def copy_model_file(
    repo_id: str,
    revision: str,
    filename: str,
    destination: Path,
    local_only: bool,
    source_dir: Path | None,
) -> None:
    local_source = source_dir / filename if source_dir is not None else None
    if local_source is not None and local_source.is_file():
        source = local_source
    else:
        source = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_files_only=local_only,
            )
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def ensure_safe_subpath(root: Path, path: Path, *, must_exist: bool = False) -> Path:
    root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"路径越出产品目录：{path}") from error
    current = root
    for part in resolved.relative_to(root).parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError(f"拒绝符号链接路径：{current}")
    if must_exist and (not resolved.is_file() or resolved.is_symlink()):
        raise FileNotFoundError(f"缺少普通文件：{resolved}")
    return resolved


def safe_metadata_file(component_dir: Path, filename: str) -> Path:
    candidate = Path(filename)
    if candidate.is_absolute() or len(candidate.parts) != 1 or ".." in candidate.parts:
        raise ValueError(f"metadata 中存在不安全文件名：{filename}")
    return ensure_safe_subpath(component_dir, component_dir / candidate, must_exist=True)


def collect_component(package_dir: Path, component: str) -> dict[str, Any]:
    component_dir = ensure_safe_subpath(package_dir, package_dir / "onnx" / component)
    metadata_path = component_dir / "export_metadata.json"
    validation_path = component_dir / "validation.json"
    operator_name = component.replace("_encoder", "") + ".json"
    operator_path = package_dir / "operators" / operator_name
    required = (component_dir / "model.onnx", metadata_path, validation_path, operator_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{component} 缺少文件：{missing}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    operators = json.loads(operator_path.read_text(encoding="utf-8"))
    if not validation.get("passed") or not operators.get("passed"):
        raise RuntimeError(f"{component} 尚未通过验证或算子检查")
    model_path = ensure_safe_subpath(package_dir, component_dir / "model.onnx", must_exist=True)
    current_model_hash = file_sha256(model_path)
    if not (
        metadata.get("model_sha256")
        == validation.get("model_sha256")
        == operators.get("model_sha256")
        == current_model_hash
    ):
        raise RuntimeError(f"{component} 的 ONNX、导出元数据、验证和算子报告哈希不一致")
    if operators.get("custom_domains"):
        raise RuntimeError(f"{component} 包含自定义 ONNX domain：{operators['custom_domains']}")
    for external in operators.get("external_data", {}).get("files", []):
        external_path = ensure_safe_subpath(
            package_dir,
            Path(external["resolved_path"]),
            must_exist=True,
        )
        if external.get("sha256") != file_sha256(external_path):
            raise RuntimeError(f"{component} external data 哈希不一致：{external_path}")

    validation_output = ensure_safe_subpath(package_dir, package_dir / "validation" / f"{component}.json")
    validation_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(validation_path, validation_output)

    test_dir = ensure_safe_subpath(package_dir, package_dir / "test_data" / component)
    test_dir.mkdir(parents=True, exist_ok=True)
    for vector in metadata["test_vectors"]:
        for field, hash_field in (
            ("input_file", "input_sha256"),
            ("reference_output_file", "reference_output_sha256"),
        ):
            source = safe_metadata_file(component_dir, vector[field])
            if file_sha256(source) != vector[hash_field]:
                raise RuntimeError(f"{component} 测试文件哈希不一致：{source}")
            shutil.copy2(source, test_dir / source.name)

    external_files = sorted(path for path in component_dir.glob("model.onnx.data*") if path.is_file())
    return {
        "profile": metadata["profile"],
        "official_weights_included": metadata.get("official_weights_included", False),
        "randomly_initialized": metadata.get("randomly_initialized", True),
        "checkpoint_fingerprint": metadata.get("checkpoint_fingerprint"),
        "source_equivalence": metadata.get("source_equivalence"),
        "description": metadata["description"],
        "model": str((component_dir / "model.onnx").relative_to(package_dir)),
        "model_sha256": current_model_hash,
        "external_data": [
            {
                "path": str(path.relative_to(package_dir)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in external_files
        ],
        "inputs": metadata["input_names"],
        "outputs": metadata["output_names"],
        "interface": metadata["interface"],
        "validation": str(validation_output.relative_to(package_dir)),
        "operators": str(operator_path.relative_to(package_dir)),
        "validated": True,
        "custom_domains": operators["custom_domains"],
    }


def main() -> None:
    args = parse_args()
    package_dir = args.package_dir.expanduser().resolve()
    try:
        package_dir.relative_to(WORKSPACE.resolve())
    except ValueError as error:
        raise ValueError("产品目录必须位于当前工作区内") from error
    for name in ("processor", "onnx", "validation", "operators", "test_data", "tools"):
        directory = package_dir / name
        if directory.exists() and directory.is_symlink():
            raise ValueError(f"拒绝产品子目录符号链接：{directory}")
        ensure_safe_subpath(package_dir, directory)
        directory.mkdir(parents=True, exist_ok=True)

    for filename in TOOL_FILES:
        source = WORKSPACE / filename
        if not source.is_file():
            raise FileNotFoundError(f"缺少打包工具文件：{source}")
        shutil.copy2(source, package_dir / "tools" / filename)

    source_dir = args.source_dir.expanduser().resolve() if args.source_dir else None
    for filename in ROOT_CONFIG_FILES:
        copy_model_file(
            args.repo_id,
            args.revision,
            filename,
            package_dir / filename,
            args.offline,
            source_dir,
        )
    for filename in PROCESSOR_FILES:
        copy_model_file(
            args.repo_id,
            args.revision,
            filename,
            package_dir / "processor" / filename,
            args.offline,
            source_dir,
        )

    components = {component: collect_component(package_dir, component) for component in THINKING_COMPONENTS}
    package_config_path = ensure_safe_subpath(package_dir, package_dir / "config.json", must_exist=True)
    fingerprinted = [
        entry["checkpoint_fingerprint"]
        for entry in components.values()
        if entry.get("checkpoint_fingerprint")
    ]
    if fingerprinted:
        package_config_sha = file_sha256(package_config_path)
        stale_configs = [
            fingerprint.get("config_sha256")
            for fingerprint in fingerprinted
            if fingerprint.get("config_sha256") != package_config_sha
        ]
        if stale_configs:
            raise RuntimeError(
                "产品目录中的 config.json 与导出所用官方 checkpoint 的 config 指纹不一致；"
                "请确认 --source-dir/--repo-id 指向导出时的同一 checkpoint"
            )
    real_weights = all(
        entry["profile"] == "real-fixed-shape"
        and entry["official_weights_included"]
        and not entry["randomly_initialized"]
        and entry["checkpoint_fingerprint"]
        for entry in components.values()
    )
    fingerprints = {
        json.dumps(entry["checkpoint_fingerprint"], sort_keys=True)
        for entry in components.values()
        if entry["checkpoint_fingerprint"]
    }
    shared_checkpoint = real_weights and len(fingerprints) == 1
    source_equivalence = all(
        entry["source_equivalence"] and entry["source_equivalence"].get("checked")
        for entry in components.values()
    )
    end_to_end_path = package_dir / "validation" / "end_to_end.json"
    end_to_end = (
        json.loads(end_to_end_path.read_text(encoding="utf-8"))
        if end_to_end_path.is_file()
        else None
    )
    end_to_end_models_match = bool(
        end_to_end
        and all(
            end_to_end.get("models", {}).get(component, {}).get("sha256") == entry["model_sha256"]
            for component, entry in components.items()
        )
    )
    end_to_end_valid = bool(
        end_to_end
        and end_to_end.get("passed")
        and end_to_end_models_match
        and end_to_end.get("decode_steps", 0) >= 3
    )
    if real_weights:
        status = (
            "official-weight-components-validated"
            if shared_checkpoint and source_equivalence and end_to_end_valid and end_to_end.get("profile") == "real-fixed-shape"
            else "unverified-real-artifacts"
        )
    else:
        status = "tiny-interface-validation-only"
    manifest = {
        "product": "Qwen3-Omni-30B-A3B-Thinking-ONNX",
        "source": {"repo_id": args.repo_id, "revision": args.revision},
        "intended_capabilities": {
            "inputs": ["text", "image", "video", "audio"],
            "outputs": ["text"],
            "audio_output": False,
        },
        "validated_profiles": {
            component: entry["interface"] for component, entry in components.items()
        },
        "status": status,
        "official_weights_included": real_weights,
        "shared_checkpoint_fingerprint": shared_checkpoint,
        "source_equivalence_passed": source_equivalence,
        "end_to_end_validation_passed": end_to_end_valid,
        "components": components,
        "end_to_end_validation": "validation/end_to_end.json" if end_to_end else None,
        "operator_summary": (
            "operators/summary.json"
            if (package_dir / "operators" / "summary.json").is_file()
            else None
        ),
        "host_responsibilities": [
            "tokenization and multimodal file preprocessing",
            "vision grid and audio chunk auxiliary tensor construction",
            "multimodal embedding placement and MRoPE position IDs",
            "autoregressive loop, token selection, and stop conditions",
            "prefill/decode KV-cache handoff",
        ],
    }
    write_json(package_dir / "manifest.json", manifest)
    print(f"[OK] package={package_dir}")
    if real_weights and status != "official-weight-components-validated":
        # 官方权重产品没通过验收判据时必须失败退出，否则上游一条龙脚本会误报成功
        print(
            f"[FAIL] status={status}：官方权重产品未通过验收判据，详见 manifest.json",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"[OK] status={manifest['status']}")


if __name__ == "__main__":
    main()
