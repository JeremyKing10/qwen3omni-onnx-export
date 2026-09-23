from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from aggregate_operators import atomic_text, collect_operator_report
from onnx_artifact_utils import (
    file_sha256,
    require_strict_validation,
    safe_path,
    source_snapshot,
    verify_artifact_identity,
)
from qwen3_omni_onnx_cases import WORKSPACE
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
    "requirements.txt",
    "qwen3_omni_onnx_cases.py",
    "qwen3_omni_thinking_components.py",
    "onnx_artifact_utils.py",
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


def resolve_package_sources(
    repo_id: str = DEFAULT_REPO,
    revision: str = DEFAULT_REVISION,
    source_dir: Path | None = None,
    offline: bool = False,
) -> dict[str, Path]:
    filenames = (*ROOT_CONFIG_FILES, *PROCESSOR_FILES)
    if source_dir is not None:
        source_dir = Path(source_dir).expanduser().absolute()
        return {
            filename: safe_path(source_dir, source_dir / filename, must_exist=True)
            for filename in filenames
        }
    sources: dict[str, Path] = {}
    for filename in filenames:
        # HF snapshots intentionally link immutable cache blobs; copy only the resolved regular blob.
        source = Path(hf_hub_download(
            repo_id=repo_id, filename=filename, revision=revision, local_files_only=offline
        )).resolve(strict=True)
        sources[filename] = safe_path(source.parent, source, must_exist=True)
    return sources


def copy_checked(source: Path, destination: Path, root: Path) -> None:
    source = safe_path(source.parent, source, must_exist=True)
    destination = safe_path(root, destination)
    if destination.exists() and not destination.is_file():
        raise ValueError(f"复制目标不是普通文件：{destination}")
    if source == destination:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, safe_path(root, destination), follow_symlinks=False)


def ensure_safe_subpath(root: Path, path: Path, *, must_exist: bool = False) -> Path:
    return safe_path(root, path, must_exist=must_exist)


def safe_metadata_file(component_dir: Path, filename: str) -> Path:
    candidate = Path(filename)
    if candidate.is_absolute() or len(candidate.parts) != 1 or ".." in candidate.parts:
        raise ValueError(f"metadata 中存在不安全文件名：{filename}")
    return ensure_safe_subpath(component_dir, component_dir / candidate, must_exist=True)


def collect_component(package_dir: Path, component: str) -> dict[str, Any]:
    if component not in THINKING_COMPONENTS:
        raise ValueError(f"未知组件：{component}")
    component_dir = safe_path(package_dir, package_dir / "onnx" / component)
    metadata_path = safe_path(package_dir, component_dir / "export_metadata.json", must_exist=True)
    validation_path = safe_path(package_dir, component_dir / "validation.json", must_exist=True)
    model_path = safe_path(package_dir, component_dir / "model.onnx", must_exist=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if metadata.get("case") != component or metadata.get("product_component") != component or validation.get("case") != component:
        raise RuntimeError(f"{component} 的 case/product_component 不一致")
    profile = metadata.get("profile")
    if profile not in {"tiny-fixed-shape", "real-fixed-shape"}:
        raise RuntimeError(f"{component} profile 无效：{profile!r}")
    if validation.get("profile") != profile or validation.get("product_component") != component:
        raise RuntimeError(f"{component} 验证报告缺少一致的产品身份；请重新导出并验证")
    if metadata.get("model_file") != "model.onnx" or not isinstance(metadata.get("interface"), dict):
        raise RuntimeError(f"{component} 导出接口元数据无效")
    real = profile == "real-fixed-shape"
    if metadata.get("official_weights_included") is not real or metadata.get("randomly_initialized") is not (not real):
        raise RuntimeError(f"{component} 权重声明与 profile 不一致")
    identity = verify_artifact_identity(model_path, validation.get("artifact_identity"))
    require_strict_validation(validation, identity)
    operators = collect_operator_report(package_dir, component)
    if operators["artifact_identity"] != identity:
        raise RuntimeError(f"{component} inspect 与 validation 不属于同一完整制品")
    if metadata.get("model_sha256") != identity["model_sha256"]:
        raise RuntimeError(f"{component} metadata 模型哈希不一致")
    for side, metadata_key in (("inputs", "input_names"), ("outputs", "output_names")):
        names = [item["name"] for item in operators["graph"][side]]
        if not names or metadata.get(metadata_key) != names:
            raise RuntimeError(f"{component} {side} 名称与当前 ONNX 签名不一致")
        if sorted(validation.get("onnxruntime", {}).get(metadata_key, [])) != sorted(names):
            raise RuntimeError(f"{component} {side} 名称与验证报告不一致")
        for vector in validation["test_vectors"]:
            if side == "outputs" and set(vector.get("comparisons", {})) != set(names):
                raise RuntimeError(f"{component} 测试输出覆盖不完整")
    for vector in metadata["test_vectors"]:
        for field in ("input_file", "reference_output_file"):
            safe_metadata_file(component_dir, vector[field])
    snapshot = metadata.get("source_snapshot")
    if not isinstance(snapshot, dict) or not snapshot:
        raise RuntimeError(f"{component} 缺少导出时代码快照；请重新导出")
    return {
        "profile": profile,
        "official_weights_included": real,
        "randomly_initialized": not real,
        "checkpoint_fingerprint": metadata.get("checkpoint_fingerprint"),
        "source_equivalence": metadata.get("source_equivalence"),
        "source_snapshot": snapshot,
        "description": metadata["description"],
        "model": str(model_path.relative_to(package_dir)),
        "model_sha256": identity["model_sha256"],
        "artifact_identity": identity,
        "test_vectors": metadata["test_vectors"],
        "external_data": [
            {"path": str((component_dir / entry["location"]).relative_to(package_dir)), **entry}
            for entry in identity["external_data"]
        ],
        "inputs": metadata["input_names"],
        "outputs": metadata["output_names"],
        "interface": metadata["interface"],
        "validation": f"validation/{component}.json",
        "operators": f"operators/{component.replace('_encoder', '')}.json",
        "validated": True,
        "custom_domains": [],
    }


def finite_nonnegative(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def end_to_end_matches(report: dict[str, Any] | None, components: dict[str, Any]) -> bool:
    if not isinstance(report, dict) or report.get("passed") is not True:
        return False
    if report.get("evidence_schema_version") != 2 or set(components) != set(THINKING_COMPONENTS):
        return False
    if report.get("reference_scope") != "official_top_level_with_raw_synthetic_features" or report.get("reference_positions") != "official_forward_independent_mrope_and_cache":
        return False
    profiles = {entry["profile"] for entry in components.values()}
    if profiles != {report.get("profile")} or type(report.get("decode_steps")) is not int or report["decode_steps"] < 3:
        return False
    if report.get("artifact_identities") != {name: entry["artifact_identity"] for name, entry in components.items()}:
        return False
    for name, entry in components.items():
        if report.get("models", {}).get(name, {}).get("sha256") != entry["model_sha256"]:
            return False
    tolerances = report.get("tolerances", {})
    if any(not finite_nonnegative(tolerances.get(name)) or tolerances[name] > limit
           for name, limit in (("rtol", 1e-4), ("atol", 1e-5))):
        return False
    stages = {"vision": "vision_encoder", "audio": "audio_encoder", "prefill": "thinker_prefill"}
    stages.update({f"decode_step_{index}": "thinker_decode" for index in range(1, report["decode_steps"] + 1)})
    comparisons = report.get("comparisons", {})
    if set(comparisons) != set(stages):
        return False
    for stage, component in stages.items():
        outputs = comparisons[stage]
        if not isinstance(outputs, dict) or set(outputs) != set(components[component]["outputs"]):
            return False
        for stats in outputs.values():
            if not isinstance(stats, dict) or any(stats.get(flag) is not True for flag in ("passed", "shape_match", "dtype_match", "finite")):
                return False
            if not finite_nonnegative(stats.get("max_abs_error")):
                return False
    return True


def cached_package_sources(package_dir: Path, previous: dict[str, Any], repo_id: str, revision: str) -> dict[str, Path] | None:
    if previous.get("source") != {"repo_id": repo_id, "revision": revision}:
        return None
    recorded = previous.get("source_files", {})
    if set(recorded) != set((*ROOT_CONFIG_FILES, *PROCESSOR_FILES)):
        return None
    sources = {}
    for filename in (*ROOT_CONFIG_FILES, *PROCESSOR_FILES):
        relative = filename if filename in ROOT_CONFIG_FILES else f"processor/{filename}"
        entry = recorded[filename]
        if not isinstance(entry, dict) or entry.get("path") != relative:
            return None
        source = safe_path(package_dir, package_dir / relative)
        if not source.is_file() or file_sha256(source) != entry.get("sha256"):
            return None
        sources[filename] = source
    return sources


def snapshot_comparison(exported: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    old_files = exported.get("files", {})
    new_files = current.get("files", {})
    changed = sorted(name for name in set(old_files) | set(new_files) if old_files.get(name) != new_files.get(name))
    critical = {
        "qwen3_omni_onnx_cases.py", "qwen3_omni_thinking_components.py", "export_onnx.py",
        "export_thinking_onnx.py", "onnx_artifact_utils.py", "validate_onnx.py", "inspect_onnx.py",
        "validate_thinking_pipeline.py", "requirements.txt",
    }
    missing = sorted(name for name in critical if not old_files.get(name) or not new_files.get(name))
    changed_critical = sorted(critical.intersection(changed))
    return {
        "all_source_hashes_match": bool(old_files) and not changed,
        "changed_files": changed,
        "missing_evidence_sources": missing,
        "changed_evidence_sources": changed_critical,
        "evidence_sources_match": not missing and not changed_critical,
        "runtime_versions_match": exported.get("packages") == current.get("packages"),
    }


def build_package(args: argparse.Namespace, package_dir: Path, previous: dict[str, Any]) -> dict[str, Any]:
    for name in ("processor", "onnx", "validation", "operators", "test_data", "tools"):
        safe_path(package_dir, package_dir / name)
    components = {component: collect_component(package_dir, component) for component in THINKING_COMPONENTS}
    profiles = {entry["profile"] for entry in components.values()}
    if len(profiles) != 1:
        raise RuntimeError("产品目录混合了不同 profile")
    sources = None if args.source_dir is not None else cached_package_sources(package_dir, previous, args.repo_id, args.revision)
    if sources is None:
        sources = resolve_package_sources(args.repo_id, args.revision, args.source_dir, args.offline)
    package_config_path = sources["config.json"]
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
        isinstance(entry["source_equivalence"], dict)
        and entry["source_equivalence"].get("checked") is True
        and finite_nonnegative(entry["source_equivalence"].get("max_abs_error"))
        and (not component.startswith("thinker_") or entry["source_equivalence"].get("scope") == "official_top_level_with_precomputed_features")
        for component, entry in components.items()
    )
    snapshot = {**source_snapshot(), "scope": "source bytes and installed versions observed at packaging, not at export"}
    snapshot_checks = {
        component: snapshot_comparison(entry["source_snapshot"], snapshot)
        for component, entry in components.items()
    }
    source_evidence_valid = all(check["evidence_sources_match"] for check in snapshot_checks.values())
    end_to_end_path = safe_path(package_dir, package_dir / "validation" / "end_to_end.json")
    end_to_end = json.loads(end_to_end_path.read_text(encoding="utf-8")) if end_to_end_path.is_file() else None
    end_to_end_valid = end_to_end_matches(end_to_end, components)
    failure_path = safe_path(package_dir, package_dir / "validation" / "end_to_end.failure.json")
    if end_to_end_valid and failure_path.is_file():
        failed_attempt = json.loads(failure_path.read_text(encoding="utf-8"))
        if failed_attempt.get("artifact_identities") == end_to_end["artifact_identities"] and failed_attempt.get("profile") == end_to_end["profile"]:
            failed_at, passed_at = failed_attempt.get("attempt_started_ns"), end_to_end.get("attempt_started_ns")
            if type(failed_at) is not int or type(passed_at) is not int or failed_at >= passed_at:
                end_to_end_valid = False
    identities = {name: entry["artifact_identity"] for name, entry in components.items()}
    summary_path = safe_path(package_dir, package_dir / "operators" / "summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
    csv_path = safe_path(package_dir, package_dir / "operators" / "all_operators.csv")
    summary_valid = bool(
        isinstance(summary, dict)
        and summary.get("schema_version") == 2
        and summary.get("passed") is True
        and summary.get("artifact_identities") == identities
        and summary.get("csv") == "operators/all_operators.csv"
        and csv_path.is_file()
        and summary.get("csv_sha256") == file_sha256(csv_path)
    )
    accepted = source_equivalence and source_evidence_valid and end_to_end_valid and summary_valid
    if profiles == {"real-fixed-shape"}:
        accepted = accepted and shared_checkpoint
        status = "official-weight-components-validated" if accepted else "unverified-real-artifacts"
    else:
        status = "tiny-interface-validation-only" if accepted else "unverified-tiny-artifacts"
    copy_plan: list[tuple[Path, Path, str]] = []
    source_files = {}
    for filename, source in sources.items():
        relative = filename if filename in ROOT_CONFIG_FILES else f"processor/{filename}"
        digest = file_sha256(source)
        source_files[filename] = {"path": relative, "sha256": digest}
        copy_plan.append((source, package_dir / relative, digest))
    for filename in TOOL_FILES:
        source = safe_path(WORKSPACE, WORKSPACE / filename, must_exist=True)
        digest = file_sha256(source)
        if filename in snapshot["files"] and digest != snapshot["files"][filename]:
            raise RuntimeError(f"打包代码已在快照后发生变化：{filename}")
        copy_plan.append((source, package_dir / "tools" / filename, digest))
    for component, entry in components.items():
        directory = package_dir / "onnx" / component
        validation = safe_path(package_dir, directory / "validation.json", must_exist=True)
        copy_plan.append((validation, package_dir / entry["validation"], file_sha256(validation)))
        for vector in entry["test_vectors"]:
            for field, hash_field in (("input_file", "input_sha256"), ("reference_output_file", "reference_output_sha256")):
                source = safe_metadata_file(directory, vector[field])
                copy_plan.append((source, package_dir / "test_data" / component / source.name, vector[hash_field]))
    for source, destination, digest in copy_plan:
        safe_path(source.parent, source, must_exist=True)
        safe_path(package_dir, destination)
        if file_sha256(source) != digest:
            raise RuntimeError(f"复制前源文件哈希发生变化：{source}")
    for source, destination, digest in copy_plan:
        copy_checked(source, destination, package_dir)
        if file_sha256(safe_path(package_dir, destination, must_exist=True)) != digest:
            raise RuntimeError(f"复制结果哈希不一致：{destination}")
    for component, entry in components.items():
        verify_artifact_identity(package_dir / entry["model"], entry["artifact_identity"])
    manifest = {
        "schema_version": 2,
        "passed": bool(accepted),
        "product": "Qwen3-Omni-30B-A3B-Thinking-ONNX",
        "source": {"repo_id": args.repo_id, "revision": args.revision},
        "source_files": source_files,
        "artifact_identities": identities,
        "source_snapshot": {name: entry["source_snapshot"] for name, entry in components.items()},
        "snapshot_at_packaging": snapshot,
        "source_snapshot_comparison": snapshot_checks,
        "evidence_sources_match": source_evidence_valid,
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
        "operator_summary": "operators/summary.json" if summary_valid else None,
        "operator_summary_valid": summary_valid,
        "host_responsibilities": [
            "tokenization and multimodal file preprocessing",
            "vision grid and audio chunk auxiliary tensor construction",
            "multimodal embedding placement and MRoPE position IDs",
            "autoregressive loop, token selection, and stop conditions",
            "prefill/decode KV-cache handoff",
        ],
    }
    return manifest


def main() -> None:
    args = parse_args()
    package_dir = safe_path(WORKSPACE, args.package_dir.expanduser().absolute())
    if package_dir == WORKSPACE:
        raise ValueError("产品目录不能是工作区根目录")
    manifest_path = safe_path(package_dir, package_dir / "manifest.json")
    previous = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            previous = loaded if isinstance(loaded, dict) else {}
        except (ValueError, OSError):
            pass
    atomic_text(package_dir, manifest_path, json.dumps({"passed": False, "status": "unverified-packaging-in-progress"}) + "\n")
    try:
        manifest = build_package(args, package_dir, previous)
        atomic_text(package_dir, manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    except Exception as error:
        failure = {"schema_version": 2, "passed": False, "status": "unverified-artifacts", "error": str(error)}
        atomic_text(package_dir, manifest_path, json.dumps(failure, ensure_ascii=False, indent=2) + "\n")
        raise
    if not manifest["passed"]:
        print(f"[FAIL] status={manifest['status']}：产品证据未通过完整验收，详见 manifest.json", file=sys.stderr)
        raise SystemExit(1)
    print(f"[OK] package={package_dir}")
    print(f"[OK] status={manifest['status']}")


if __name__ == "__main__":
    main()
