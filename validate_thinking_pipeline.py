from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import psutil
import torch

from qwen3_omni_onnx_cases import DEFAULT_SEED, file_sha256, normalize_outputs, tensor_to_numpy, write_json
from qwen3_omni_thinking_components import (
    build_real_thinking_component,
    build_tiny_thinking_component,
    load_real_thinking_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="端到端验证 Thinking 四组件数据流")
    parser.add_argument("--mode", choices=("tiny", "real"), default="tiny")
    parser.add_argument("--model-path", type=Path, help="real 模式官方 checkpoint 目录")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--minimum-memory-gib", type=float, default=96.0)
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
    "tensor(bfloat16)": np.float32,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(bool)": np.bool_,
}


def cast_feed(session: ort.InferenceSession, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Cast each feed array to the dtype declared by the corresponding graph input."""
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
    session = ort.InferenceSession(str(model_path), providers=[provider])
    feed = {name: tensor_to_numpy(tensor) for name, tensor in zip(names, tensors)}
    output_names = [item.name for item in session.get_outputs()]
    return tuple(session.run(output_names, feed))


def compare_outputs(
    actual: tuple[np.ndarray, ...],
    expected: tuple[torch.Tensor, ...],
    names: tuple[str, ...],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    if len(actual) != len(expected) or len(actual) != len(names):
        raise RuntimeError("端到端输出数量不一致")
    result: dict[str, Any] = {}
    for name, actual_value, expected_value in zip(names, actual, expected):
        reference = tensor_to_numpy(expected_value)
        absolute = np.abs(actual_value.astype(np.float64) - reference.astype(np.float64))
        passed = bool(
            actual_value.shape == reference.shape
            and actual_value.dtype == reference.dtype
            and np.isfinite(actual_value).all()
            and np.allclose(actual_value, reference, rtol=rtol, atol=atol)
        )
        result[name] = {
            "passed": passed,
            "shape": list(actual_value.shape),
            "dtype": str(actual_value.dtype),
            "max_abs_error": float(absolute.max(initial=0.0)),
        }
    return result


def main() -> None:
    args = parse_args()
    package_dir = args.package_dir.expanduser().resolve()
    if args.provider not in ort.get_available_providers():
        raise RuntimeError(
            f"ORT provider {args.provider} 不可用；当前可用：{ort.get_available_providers()}"
        )
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
        dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
        full_model = load_real_thinking_model(str(args.model_path.resolve()), dtype=dtype, device=args.device)
    cases = {
        name: (
            build_tiny_thinking_component(name, args.seed)
            if args.mode == "tiny"
            else build_real_thinking_component(full_model, name, args.seed)
        )
        for name in ("vision_encoder", "audio_encoder", "thinker_prefill", "thinker_decode")
    }

    vision_case = cases["vision_encoder"]
    audio_case = cases["audio_encoder"]
    prefill_case = cases["thinker_prefill"]
    decode_case = cases["thinker_decode"]

    with torch.inference_mode():
        vision_pt = normalize_outputs(vision_case.model(*vision_case.export_args))
        audio_pt = normalize_outputs(audio_case.model(*audio_case.export_args))
    vision_ort = run_ort(
        package_dir / "onnx" / "vision_encoder" / "model.onnx",
        vision_case.input_names,
        vision_case.export_args,
        args.provider,
    )
    audio_ort = run_ort(
        package_dir / "onnx" / "audio_encoder" / "model.onnx",
        audio_case.input_names,
        audio_case.export_args,
        args.provider,
    )

    prefill_template = list(prefill_case.export_args)
    vision_tokens = vision_pt[0].shape[0]
    audio_tokens = audio_pt[0].shape[0]
    multimodal_end = 1 + vision_tokens + audio_tokens
    if multimodal_end >= prefill_template[0].shape[1]:
        raise RuntimeError("Prefill profile 无法容纳 Vision、Audio 和至少一个文本位置")
    multimodal_pt = torch.zeros_like(prefill_template[4])
    multimodal_pt[:, 1 : 1 + vision_tokens, :] = vision_pt[0]
    multimodal_pt[:, 1 + vision_tokens : multimodal_end, :] = audio_pt[0]
    multimodal_mask = torch.zeros_like(prefill_template[5])
    multimodal_mask[:, 1:multimodal_end] = True
    visual_mask = torch.zeros_like(prefill_template[6])
    visual_mask[:, 1 : 1 + vision_tokens] = True
    deepstack_pt = list(vision_pt[1:])
    prefill_pt_args = tuple(
        prefill_template[:4]
        + [multimodal_pt, multimodal_mask, visual_mask, *deepstack_pt]
    )
    with torch.inference_mode():
        prefill_pt = normalize_outputs(prefill_case.model(*prefill_pt_args))

    multimodal_ort = np.zeros(
        tuple(prefill_template[4].shape),
        dtype=vision_ort[0].dtype,
    )
    multimodal_ort[:, 1 : 1 + vision_tokens, :] = vision_ort[0]
    multimodal_ort[:, 1 + vision_tokens : multimodal_end, :] = audio_ort[0]
    prefill_session = ort.InferenceSession(
        str(package_dir / "onnx" / "thinker_prefill" / "model.onnx"),
        providers=[args.provider],
    )
    prefill_feed = {
        prefill_case.input_names[0]: tensor_to_numpy(prefill_template[0]),
        prefill_case.input_names[1]: tensor_to_numpy(prefill_template[1]),
        prefill_case.input_names[2]: tensor_to_numpy(prefill_template[2]),
        prefill_case.input_names[3]: tensor_to_numpy(prefill_template[3]),
        prefill_case.input_names[4]: multimodal_ort,
        prefill_case.input_names[5]: tensor_to_numpy(multimodal_mask),
        prefill_case.input_names[6]: tensor_to_numpy(visual_mask),
    }
    for name, value in zip(prefill_case.input_names[7:], vision_ort[1:]):
        prefill_feed[name] = value
    prefill_output_names = [item.name for item in prefill_session.get_outputs()]
    prefill_ort = tuple(prefill_session.run(prefill_output_names, cast_feed(prefill_session, prefill_feed)))

    del prefill_session
    device = prefill_template[0].device
    rope_delta = int(prefill_case.interface["rope_deltas"][0])
    pt_cache = tuple(prefill_pt[1:])
    ort_cache = tuple(prefill_ort[1:])
    decode_session = ort.InferenceSession(
        str(package_dir / "onnx" / "thinker_decode" / "model.onnx"),
        providers=[args.provider],
    )
    decode_output_names = [item.name for item in decode_session.get_outputs()]
    decode_comparisons: dict[str, Any] = {}
    for step in range(3):
        past_length = int(pt_cache[0].shape[-2])
        next_token = torch.tensor([[13 + step]], dtype=torch.int64, device=device)
        decode_position_value = past_length + rope_delta
        decode_prefix = (
            next_token,
            torch.ones(1, past_length + 1, dtype=torch.int64, device=device),
            torch.full((3, 1, 1), decode_position_value, dtype=torch.int64, device=device),
            torch.tensor([past_length], dtype=torch.int64, device=device),
        )
        with torch.inference_mode():
            decode_pt = normalize_outputs(decode_case.model(*decode_prefix, *pt_cache))
        decode_feed = {
            name: tensor_to_numpy(tensor)
            for name, tensor in zip(decode_case.input_names[:4], decode_prefix)
        }
        for name, value in zip(decode_case.input_names[4:], ort_cache):
            decode_feed[name] = value
        decode_ort = tuple(decode_session.run(decode_output_names, cast_feed(decode_session, decode_feed)))
        decode_comparisons[f"decode_step_{step + 1}"] = compare_outputs(
            decode_ort,
            decode_pt,
            tuple(decode_output_names),
            args.rtol,
            args.atol,
        )
        pt_cache = tuple(decode_pt[1:])
        ort_cache = tuple(decode_ort[1:])

    comparisons = {
        "vision": compare_outputs(vision_ort, vision_pt, vision_case.output_names, args.rtol, args.atol),
        "audio": compare_outputs(audio_ort, audio_pt, audio_case.output_names, args.rtol, args.atol),
        "prefill": compare_outputs(prefill_ort, prefill_pt, tuple(prefill_output_names), args.rtol, args.atol),
        **decode_comparisons,
    }
    passed = all(item["passed"] for stage in comparisons.values() for item in stage.values())
    report = {
        "passed": passed,
        "profile": f"{args.mode}-fixed-shape",
        "flow": ["vision_encoder", "audio_encoder", "thinker_prefill", "thinker_decode"],
        "models": {
            name: {
                "path": f"onnx/{name}/model.onnx",
                "sha256": file_sha256(package_dir / "onnx" / name / "model.onnx"),
            }
            for name in ("vision_encoder", "audio_encoder", "thinker_prefill", "thinker_decode")
        },
        "decode_steps": 3,
        "tolerances": {"rtol": args.rtol, "atol": args.atol},
        "comparisons": comparisons,
        "note": (
            "This validates tensor handoff with synthetic embeddings; "
            "real mode uses official weights but is still not a perceptual media quality benchmark."
        ),
    }
    output_path = package_dir / "validation" / "end_to_end.json"
    if output_path.is_file() and not args.force:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("profile") != report["profile"]:
            raise SystemExit(
                f"已存在 profile={existing.get('profile')!r} 的端到端报告；"
                f"如需用 {report['profile']!r} 覆盖请加 --force"
            )
    write_json(output_path, report)
    print(f"[{'OK' if passed else 'FAIL'}] end_to_end={output_path}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
