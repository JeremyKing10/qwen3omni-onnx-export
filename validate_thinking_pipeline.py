from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np
import onnxruntime as ort
import psutil
import torch

from onnx_artifact_utils import (
    artifact_identity,
    check_providers,
    require_strict_validation,
    run_ort_session,
    safe_path,
    validate_tolerances,
    verify_artifact_identity,
)
from qwen3_omni_onnx_cases import DEFAULT_SEED, WORKSPACE, tensor_to_numpy, write_json
from qwen3_omni_thinking_components import (
    THINKING_COMPONENTS,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    build_real_thinking_component,
    build_tiny_thinking_component,
    decode_position_ids,
    flatten_thinker_output,
    inject_multimodal_features,
    load_real_thinking_model,
    make_tiny_thinker_config,
    reference_decode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="端到端验证 Thinking 四组件数据流")
    parser.add_argument("--mode", choices=("tiny", "real"), default="tiny")
    parser.add_argument("--model-path", type=Path, help="real 模式官方 checkpoint 目录")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--minimum-memory-gib", type=float, default=128.0)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "Qwen3-Omni-30B-A3B-Thinking-ONNX",
    )
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--force",
        action="store_true",
        help="允许覆盖模式不同的旧端到端报告（例如用 tiny 覆盖 real 证据）",
    )
    return parser.parse_args()


ORT_DTYPE_MAP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    # bf16 图的输入必须喂 bf16；喂 float32 会被 ORT 直接拒绝（real 模式 --dtype bfloat16 会走这里）
    "tensor(bfloat16)": ml_dtypes.bfloat16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(bool)": np.bool_,
}


def ensure_feed_dtype_supported(session: ort.InferenceSession, provider: str) -> None:
    """BF16 由公共 IOBinding 路径处理；实际 kernel 能力由 ORT session 判断。"""
    check_providers([provider])


def cast_feed(
    session: ort.InferenceSession, feed: dict[str, np.ndarray], provider: str
) -> dict[str, np.ndarray]:
    """Cast each feed array to the dtype declared by the corresponding graph input."""
    ensure_feed_dtype_supported(session, provider)
    if set(feed) != {item.name for item in session.get_inputs()}:
        raise ValueError("ORT 输入名称与图签名不一致")
    casted: dict[str, np.ndarray] = {}
    for item in session.get_inputs():
        value = feed[item.name]
        target = ORT_DTYPE_MAP.get(item.type)
        casted[item.name] = value.astype(target) if target is not None else value
    return casted


def run_ort(
    model_path: Path,
    names: tuple[str, ...],
    tensors: tuple[torch.Tensor, ...],
    provider: str,
) -> tuple[np.ndarray, ...]:
    check_providers([provider])
    session = ort.InferenceSession(str(model_path), providers=[provider])
    if len(names) != len(tensors):
        raise ValueError("ORT 输入名称与张量数量不一致")
    feed = {name: tensor_to_numpy(tensor) for name, tensor in zip(names, tensors)}
    return run_ort_session(session, cast_feed(session, feed, provider))


def compare_outputs(
    actual: tuple[np.ndarray, ...],
    expected: tuple[torch.Tensor, ...],
    names: tuple[str, ...],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    validate_tolerances(rtol, atol)
    if len(actual) != len(expected) or len(actual) != len(names):
        return {"output_count": {"passed": False, "actual": len(actual), "expected": len(expected), "names": len(names)}}
    result: dict[str, Any] = {}
    for name, actual_value, expected_value in zip(names, actual, expected):
        reference = tensor_to_numpy(expected_value)
        shape_match = actual_value.shape == reference.shape
        dtype_match = actual_value.dtype == reference.dtype
        actual_float, expected_float = actual_value.astype(np.float64), reference.astype(np.float64)
        finite = bool(np.isfinite(actual_float).all() and np.isfinite(expected_float).all())
        absolute = np.abs(actual_float - expected_float) if shape_match and finite else None
        passed = bool(
            shape_match and dtype_match and finite
            and np.allclose(actual_float, expected_float, rtol=rtol, atol=atol)
        )
        result[name] = {
            "passed": passed,
            "shape": list(actual_value.shape),
            "expected_shape": list(reference.shape),
            "shape_match": shape_match,
            "dtype": str(actual_value.dtype),
            "expected_dtype": str(reference.dtype),
            "dtype_match": dtype_match,
            "finite": finite,
            "max_abs_error": float(absolute.max(initial=0.0)) if absolute is not None else None,
        }
    return result


def load_component_evidence(package_dir: Path, mode: str, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    identities, metadata = {}, {}
    for name in THINKING_COMPONENTS:
        root = package_dir / "onnx" / name
        model_path = safe_path(WORKSPACE, root / "model.onnx", must_exist=True)
        meta_path = safe_path(WORKSPACE, root / "export_metadata.json", must_exist=True)
        validation_path = safe_path(WORKSPACE, root / "validation.json", must_exist=True)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("profile") != f"{mode}-fixed-shape" or meta.get("case") != name or meta.get("product_component") != name:
            raise ValueError(f"{name} 组件身份/profile 与请求的 {mode} 不一致")
        if meta.get("seed") != seed:
            raise ValueError(f"{name} 导出 seed 与参考 seed 不一致")
        identity = artifact_identity(model_path)
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        require_strict_validation(validation, identity)
        identities[name], metadata[name] = identity, meta
    return identities, metadata


def official_raw_prefill(cases: dict[str, Any], full_model: Any = None) -> tuple[Any, tuple, tuple, tuple]:
    vision_case, audio_case, prefill_case = (cases[name] for name in THINKING_COMPONENTS[:3])
    if full_model is None:
        thinker = Qwen3OmniMoeThinkerForConditionalGeneration(make_tiny_thinker_config()).eval()
        thinker.visual = vision_case.model.vision
        thinker.audio_tower = audio_case.model.audio
        thinker.model = prefill_case.model.text_model
        thinker.lm_head = prefill_case.model.lm_head
    else:
        thinker = full_model.thinker
    template = prefill_case.export_args
    if vision_case.raw_reference_inputs is None or audio_case.raw_reference_inputs is None:
        raise ValueError("官方 raw 参考必须保留原始媒体输入，不能从 wrapper 预处理结果逆推")
    pixel_values, grid = vision_case.raw_reference_inputs
    raw_features, feature_lengths = audio_case.raw_reference_inputs
    if feature_lengths.numel() != 1 or raw_features.shape[1] != int(feature_lengths[0]):
        raise ValueError("固定 E2E profile 只支持一段未填充原始音频")
    feature_length = int(feature_lengths[0])
    captured: dict[str, Any] = {}

    def capture_vision(_module: Any, _inputs: Any, output: Any) -> None:
        captured["vision"] = (output.pooler_output, *output.deepstack_features)

    def capture_audio(_module: Any, _inputs: Any, output: Any) -> None:
        captured["audio"] = (output.last_hidden_state,)

    hooks = [thinker.visual.register_forward_hook(capture_vision), thinker.audio_tower.register_forward_hook(capture_audio)]
    try:
        with torch.inference_mode():
            output = thinker(
                input_ids=template[0],
                attention_mask=template[1],
                pixel_values=pixel_values,
                image_grid_thw=grid.to(template[0].device),
                input_features=raw_features.unsqueeze(0),
                feature_attention_mask=torch.ones((1, feature_length), dtype=torch.int64, device=raw_features.device),
                use_cache=True,
                return_dict=True,
            )
    finally:
        for hook in hooks:
            hook.remove()
    return thinker, captured["vision"], captured["audio"], flatten_thinker_output(output)


def _numpy_tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    if value.dtype == np.dtype(ml_dtypes.bfloat16):
        return torch.from_numpy(value.view(np.uint16).copy()).view(torch.bfloat16).to(device)
    return torch.from_numpy(value.copy()).to(device)


def validate_pipeline(args: argparse.Namespace, package_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    full_model = None
    if args.mode == "real":
        if args.model_path is None:
            raise ValueError("real 模式必须提供 --model-path")
        checkpoint_path = args.model_path.expanduser().resolve()
        memory = psutil.virtual_memory()
        total_memory_gib = memory.total / 1024**3
        available_memory_gib = memory.available / 1024**3
        if total_memory_gib < args.minimum_memory_gib or available_memory_gib < args.minimum_memory_gib * 0.75:
            raise RuntimeError(
                f"real 模式需要至少 {args.minimum_memory_gib:.0f} GiB 总内存且至少 "
                f"{args.minimum_memory_gib * 0.75:.0f} GiB 可用；当前总计 {total_memory_gib:.1f} GiB、"
                f"可用 {available_memory_gib:.1f} GiB"
            )
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("请求 CUDA 端到端验证，但 torch.cuda 不可用")
        dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
        full_model = load_real_thinking_model(str(checkpoint_path), dtype=dtype, device=args.device)
    cases = {
        name: (
            build_tiny_thinking_component(name, args.seed)
            if args.mode == "tiny"
            else build_real_thinking_component(
                full_model, name, args.seed,
                text_sequence_length=metadata["thinker_prefill"]["interface"]["sequence_profile"],
                vision_grid=tuple(metadata["vision_encoder"]["interface"]["grid_profile"]),
            )
        )
        for name in THINKING_COMPONENTS
    }
    for name, case in cases.items():
        if list(case.input_names) != metadata[name]["input_names"] or list(case.output_names) != metadata[name]["output_names"]:
            raise ValueError(f"{name} 导出接口与参考组件不一致")
        if args.mode == "real":
            expected = {key: value for key, value in metadata[name]["checkpoint_fingerprint"].items() if key != "model_path"}
            actual = {key: value for key, value in case.checkpoint_fingerprint.items() if key != "model_path"}
            if expected != actual:
                raise ValueError(f"{name} 参考 checkpoint 与导出 checkpoint 不一致")

    vision_case, audio_case, prefill_case, decode_case = (cases[name] for name in THINKING_COMPONENTS)
    thinker, vision_pt, audio_pt, prefill_pt = official_raw_prefill(cases, full_model)
    vision_ort = run_ort(
        package_dir / "onnx" / "vision_encoder" / "model.onnx",
        vision_case.input_names, vision_case.export_args, args.provider,
    )
    audio_ort = run_ort(
        package_dir / "onnx" / "audio_encoder" / "model.onnx",
        audio_case.input_names, audio_case.export_args, args.provider,
    )
    comparisons = {
        "vision": compare_outputs(vision_ort, vision_pt, vision_case.output_names, args.rtol, args.atol),
        "audio": compare_outputs(audio_ort, audio_pt, audio_case.output_names, args.rtol, args.atol),
    }
    if any(not item.get("shape_match", False) for stage in comparisons.values() for item in stage.values()):
        return {"comparisons": comparisons, "decode_steps": 0}

    template = prefill_case.export_args
    device = template[0].device
    prefill_inputs = inject_multimodal_features(
        template, thinker.config,
        tuple(_numpy_tensor(value, device) for value in vision_ort),
        tuple(_numpy_tensor(value, device) for value in audio_ort),
    )
    prefill_ort = run_ort(
        package_dir / "onnx" / "thinker_prefill" / "model.onnx",
        prefill_case.input_names, prefill_inputs, args.provider,
    )
    comparisons["prefill"] = compare_outputs(prefill_ort, prefill_pt, prefill_case.output_names, args.rtol, args.atol)
    if any(not item.get("shape_match", False) for item in comparisons["prefill"].values()):
        return {"comparisons": comparisons, "decode_steps": 0}

    rope_delta = torch.tensor([[prefill_case.interface["rope_deltas"][0]]], dtype=torch.float32, device=device)
    pt_cache, ort_cache = tuple(prefill_pt[1:]), tuple(prefill_ort[1:])
    decode_session = ort.InferenceSession(
        str(package_dir / "onnx" / "thinker_decode" / "model.onnx"), providers=[args.provider],
    )
    if tuple(item.name for item in decode_session.get_outputs()) != decode_case.output_names:
        raise ValueError("Decode ONNX 输出顺序与组件接口不一致")
    for step in range(3):
        next_token = torch.tensor([[13 + step]], dtype=torch.int64, device=device)
        with torch.inference_mode():
            decode_pt = reference_decode(thinker, next_token, pt_cache)
        past_length = int(ort_cache[0].shape[-2])
        decode_prefix = (
            next_token,
            torch.ones(1, past_length + 1, dtype=torch.int64, device=device),
            decode_position_ids(past_length, rope_delta, device),
            torch.tensor([past_length], dtype=torch.int64, device=device),
        )
        decode_feed = {name: tensor_to_numpy(tensor) for name, tensor in zip(decode_case.input_names[:4], decode_prefix)}
        decode_feed.update(zip(decode_case.input_names[4:], ort_cache))
        decode_ort = run_ort_session(decode_session, cast_feed(decode_session, decode_feed, args.provider))
        stage = f"decode_step_{step + 1}"
        comparisons[stage] = compare_outputs(decode_ort, decode_pt, decode_case.output_names, args.rtol, args.atol)
        if any(not item.get("shape_match", False) for item in comparisons[stage].values()):
            return {"comparisons": comparisons, "decode_steps": step + 1}
        pt_cache, ort_cache = tuple(decode_pt[1:]), tuple(decode_ort[1:])
    return {"comparisons": comparisons, "decode_steps": 3}


def main() -> None:
    args = parse_args()
    package_dir = safe_path(WORKSPACE, args.package_dir)
    if package_dir == WORKSPACE.resolve():
        raise ValueError("产品目录不能是工作区根目录")
    output_path = safe_path(WORKSPACE, package_dir / "validation" / "end_to_end.json")
    failure_path = safe_path(WORKSPACE, package_dir / "validation" / "end_to_end.failure.json")
    attempt_started_ns = time.time_ns()
    identities: dict[str, Any] = {}
    try:
        validate_tolerances(args.rtol, args.atol)
        check_providers([args.provider])
        if output_path.is_file() and not args.force:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if existing.get("profile") != f"{args.mode}-fixed-shape":
                raise ValueError(f"已存在 profile={existing.get('profile')!r} 的端到端报告；不同模式覆盖必须加 --force")
        identities, metadata = load_component_evidence(package_dir, args.mode, args.seed)
        result = validate_pipeline(args, package_dir, metadata)
        comparisons = result["comparisons"]
        for stage in ("vision", "audio", "prefill", "decode_step_1", "decode_step_2", "decode_step_3"):
            comparisons.setdefault(stage, {"not_run": {"passed": False, "reason": "upstream shape mismatch"}})
        for name in THINKING_COMPONENTS:
            model_path = safe_path(WORKSPACE, package_dir / "onnx" / name / "model.onnx", must_exist=True)
            verify_artifact_identity(model_path, identities[name])
        passed = result["decode_steps"] == 3 and all(
            stage and all(item.get("passed") is True for item in stage.values())
            for stage in comparisons.values()
        )
        report = {
            "evidence_schema_version": 2,
            "attempt_started_ns": attempt_started_ns,
            "passed": passed,
            "profile": f"{args.mode}-fixed-shape",
            "flow": list(THINKING_COMPONENTS),
            "artifact_identities": identities,
            "models": {
                name: {"path": f"onnx/{name}/model.onnx", "sha256": identity["model_sha256"], "artifact_identity": identity}
                for name, identity in identities.items()
            },
            "decode_steps": result["decode_steps"],
            "tolerances": {"rtol": args.rtol, "atol": args.atol},
            "comparisons": comparisons,
            "reference_scope": "official_top_level_with_raw_synthetic_features",
            "reference_positions": "official_forward_independent_mrope_and_cache",
            "experts_reference": "official_batched_mm; real eager low-precision equivalence not established",
            "note": "Synthetic raw image/audio tensors with official injection and independent reference caches; not a perceptual media quality benchmark.",
        }
        write_json(safe_path(WORKSPACE, output_path), report)
    except (Exception, KeyboardInterrupt) as error:
        write_json(safe_path(WORKSPACE, failure_path), {
            "evidence_schema_version": 2,
            "attempt_started_ns": attempt_started_ns,
            "passed": False,
            "profile": f"{args.mode}-fixed-shape",
            "artifact_identities": identities,
            "error": f"{type(error).__name__}: {error}",
        })
        raise
    print(f"[{'OK' if passed else 'FAIL'}] end_to_end={output_path}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
