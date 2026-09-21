from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np
import torch
from torch import nn
import transformers
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextModel,
    Qwen3OmniMoeThinkerTextRMSNorm,
    Qwen3OmniMoeThinkerTextSparseMoeBlock,
    Qwen3OmniMoeThinkerTextTopKRouter,
)

WORKSPACE = Path(__file__).resolve().parent
TRANSFORMERS_REPO = WORKSPACE / "transformers-v5.2.0"
EXPECTED_TRANSFORMERS_REVISION = "7d9754a05193eb79b1d86aa744b622b8068008cd"
DEFAULT_SEED = 1234
SUPPORTED_CASES = ("rmsnorm", "moe_block", "tiny_thinker")


@dataclass(frozen=True)
class ExportCase:
    name: str
    model: nn.Module
    test_vectors: tuple[tuple[torch.Tensor, ...], ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    description: str
    config: dict[str, Any]

    @property
    def export_args(self) -> tuple[torch.Tensor, ...]:
        return self.test_vectors[0]


class TinyThinkerWithLmHead(nn.Module):
    def __init__(self, config: Qwen3OmniMoeTextConfig) -> None:
        super().__init__()
        self.text_model = Qwen3OmniMoeThinkerTextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=False,
        ).last_hidden_state
        return self.lm_head(hidden_states)


def git_revision(repo: Path) -> str | None:
    if not (repo / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def assert_transformers_provenance() -> dict[str, str]:
    imported_file = Path(transformers.__file__).resolve()
    expected_package = (TRANSFORMERS_REPO / "src" / "transformers").resolve()
    revision = git_revision(TRANSFORMERS_REPO)
    try:
        imported_file.relative_to(expected_package)
    except ValueError as error:
        raise RuntimeError(
            f"当前导入的 Transformers 不来自固定源码目录：{imported_file}；期望位于 {expected_package}"
        ) from error
    if revision != EXPECTED_TRANSFORMERS_REVISION:
        raise RuntimeError(
            f"Transformers revision 不匹配：{revision}；期望 {EXPECTED_TRANSFORMERS_REVISION}"
        )
    return {"imported_file": str(imported_file), "revision": revision}


def make_tiny_text_config() -> Qwen3OmniMoeTextConfig:
    config = Qwen3OmniMoeTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        hidden_act="silu",
        max_position_embeddings=16,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1_000_000.0,
            "mrope_section": [1, 1, 2],
            "interleaved": True,
        },
        attention_bias=False,
        attention_dropout=0.0,
        decoder_sparse_step=1,
        moe_intermediate_size=4,
        num_experts_per_tok=2,
        num_experts=4,
        norm_topk_prob=True,
        output_router_logits=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attn_implementation="eager",
        experts_implementation="batched_mm",
        return_dict=True,
    )
    if config._attn_implementation != "eager" or config._experts_implementation != "batched_mm":
        raise RuntimeError("tiny config 未采用 eager attention + batched_mm experts")
    return config


def _initialize_standalone_module(module: nn.Module, std: float = 0.02) -> None:
    for parameter in module.parameters():
        if parameter.ndim == 1:
            nn.init.ones_(parameter)
        else:
            nn.init.normal_(parameter, mean=0.0, std=std)


def _text_inputs(token_ids: list[int], attention: list[int]) -> tuple[torch.Tensor, ...]:
    input_ids = torch.tensor([token_ids], dtype=torch.int64)
    attention_mask = torch.tensor([attention], dtype=torch.int64)
    cache_position = torch.arange(input_ids.shape[1], dtype=torch.int64)
    position_ids = cache_position.view(1, 1, -1).expand(3, input_ids.shape[0], -1).contiguous()
    return input_ids, attention_mask, position_ids, cache_position


def build_case(case_name: str, seed: int = DEFAULT_SEED) -> ExportCase:
    if case_name not in SUPPORTED_CASES:
        raise ValueError(f"Unsupported case {case_name!r}; choose from {SUPPORTED_CASES}")

    assert_transformers_provenance()
    torch.manual_seed(seed)
    np.random.seed(seed)
    config = make_tiny_text_config()
    config_dict = config.to_dict()

    if case_name == "rmsnorm":
        model = Qwen3OmniMoeThinkerTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        first = torch.randn(1, 4, config.hidden_size, dtype=torch.float32)
        second = torch.randn(1, 4, config.hidden_size, dtype=torch.float32) * 1.7 - 0.3
        return ExportCase(
            name=case_name,
            model=model.eval(),
            test_vectors=((first,), (second,)),
            input_names=("hidden_states",),
            output_names=("normalized_hidden_states",),
            description="Qwen3-Omni Thinker 文本 RMSNorm 基础层",
            config=config_dict,
        )

    if case_name == "moe_block":
        model = Qwen3OmniMoeThinkerTextSparseMoeBlock(config)
        _initialize_standalone_module(model, config.initializer_range)
        first = torch.randn(1, 4, config.hidden_size, dtype=torch.float32)
        second = -first + torch.linspace(-0.2, 0.2, config.hidden_size).view(1, 1, -1)
        return ExportCase(
            name=case_name,
            model=model.eval(),
            test_vectors=((first,), (second,)),
            input_names=("hidden_states",),
            output_names=("moe_hidden_states",),
            description="Qwen3-Omni Thinker 真实 Sparse MoE Block，采用 4 专家、top-2 和 batched_mm experts",
            config=config_dict,
        )

    model = TinyThinkerWithLmHead(config).eval()
    first = _text_inputs([1, 5, 9, 2], [1, 1, 1, 1])
    second = _text_inputs([2, 11, 7, 0], [1, 1, 1, 0])
    return ExportCase(
        name=case_name,
        model=model,
        test_vectors=(first, second),
        input_names=("input_ids", "attention_mask", "position_ids", "cache_position"),
        output_names=("logits",),
        description="Qwen3-Omni tiny Thinker Text Model + LM Head，1 层、4 专家 top-2、无 KV Cache",
        config=config_dict,
    )


def capture_routing(model: nn.Module, args: tuple[torch.Tensor, ...]) -> dict[str, Any] | None:
    captured: list[torch.Tensor] = []
    hooks = []

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        captured.append(output[2].detach().cpu())

    for module in model.modules():
        if isinstance(module, Qwen3OmniMoeThinkerTextTopKRouter):
            hooks.append(module.register_forward_hook(hook))
    if not hooks:
        return None

    try:
        with torch.inference_mode():
            model(*args)
    finally:
        for registered_hook in hooks:
            registered_hook.remove()

    indices = torch.cat([item.reshape(-1) for item in captured]).numpy().astype(np.int64)
    return {
        "sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "indices": indices.tolist(),
        "unique_experts": sorted(set(indices.tolist())),
    }


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu()
    if tensor.dtype == torch.bfloat16:
        return np.asarray(tensor.float().numpy(), dtype=ml_dtypes.bfloat16)
    return tensor.numpy()


def tensor_dict(names: tuple[str, ...], tensors: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
    if len(names) != len(tensors):
        raise ValueError(f"Expected {len(names)} tensors, got {len(tensors)}")
    return {name: tensor_to_numpy(tensor) for name, tensor in zip(names, tensors)}


def normalize_outputs(output: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(output, torch.Tensor):
        return (output,)
    if isinstance(output, (tuple, list)) and all(isinstance(item, torch.Tensor) for item in output):
        return tuple(output)
    raise TypeError(f"Only Tensor or flat Tensor tuple/list outputs are supported, got {type(output)!r}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_metadata() -> dict[str, Any]:
    provenance = assert_transformers_provenance()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "transformers_imported_file": provenance["imported_file"],
        "transformers_revision": provenance["revision"],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
