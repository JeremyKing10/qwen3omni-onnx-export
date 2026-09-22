from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeTextConfig,
    Qwen3OmniMoeThinkerConfig,
    Qwen3OmniMoeVisionEncoderConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerTextModel,
    Qwen3OmniMoeVisionEncoder,
    _get_feat_extract_output_lengths,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)

from qwen3_omni_onnx_cases import (
    DEFAULT_SEED,
    assert_transformers_provenance,
    capture_routing,
    file_sha256,
    make_tiny_text_config,
    normalize_outputs,
)

THINKING_COMPONENTS = ("vision_encoder", "audio_encoder", "thinker_prefill", "thinker_decode")


@dataclass(frozen=True)
class ThinkingComponentCase:
    name: str
    model: nn.Module
    test_vectors: tuple[tuple[torch.Tensor, ...], ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    description: str
    config: dict[str, Any]
    interface: dict[str, Any]
    source_equivalence: dict[str, Any] | None = None
    dynamic_shapes: Any | None = None
    checkpoint_fingerprint: dict[str, Any] | None = None

    @property
    def export_args(self) -> tuple[torch.Tensor, ...]:
        return self.test_vectors[0]


class VisionEncoderExportWrapper(nn.Module):
    """Vision neural network with grid-dependent Python preprocessing moved to tensor inputs."""

    def __init__(self, vision: Qwen3OmniMoeVisionEncoder) -> None:
        super().__init__()
        self.vision = vision

    @staticmethod
    def _attention(
        attention: nn.Module,
        hidden_states: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        sequence_length = hidden_states.shape[0]
        query, key, value = (
            attention.qkv(hidden_states)
            .reshape(sequence_length, 3, attention.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        query, key = apply_rotary_pos_emb_vision(query, key, rotary_cos, rotary_sin)
        query = query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)
        output, _ = eager_attention_forward(
            attention,
            query,
            key,
            value,
            attention_mask=attention_mask,
            scaling=attention.scaling,
            dropout=0.0,
            is_causal=False,
        )
        return attention.proj(output.reshape(sequence_length, -1).contiguous())

    def forward(
        self,
        pixel_values: torch.Tensor,
        position_indices: torch.Tensor,
        position_weights: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        hidden_states = self.vision.patch_embed(pixel_values)
        position_embeddings = (
            self.vision.pos_embed(position_indices) * position_weights[:, :, None]
        ).sum(dim=0)
        hidden_states = hidden_states + position_embeddings.to(hidden_states.dtype)

        deepstack: list[torch.Tensor] = []
        for layer_index, block in enumerate(self.vision.blocks):
            hidden_states = hidden_states + self._attention(
                block.attn,
                block.norm1(hidden_states),
                rotary_cos,
                rotary_sin,
                attention_mask,
            )
            hidden_states = hidden_states + block.mlp(block.norm2(hidden_states))
            if layer_index in self.vision.deepstack_visual_indexes:
                merger_index = self.vision.deepstack_visual_indexes.index(layer_index)
                deepstack.append(self.vision.deepstack_merger_list[merger_index](hidden_states))

        pooled = self.vision.merger(hidden_states)
        return (pooled, *deepstack)


class AudioEncoderExportWrapper(nn.Module):
    """Audio CNN + Transformer; variable-length chunk preparation is a host-side contract."""

    def __init__(self, audio: Qwen3OmniMoeAudioEncoder) -> None:
        super().__init__()
        self.audio = audio

    def forward(
        self,
        padded_features: torch.Tensor,
        valid_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        features = padded_features.unsqueeze(1).to(dtype=self.audio.conv2d1.weight.dtype)
        features = F.gelu(self.audio.conv2d1(features))
        features = F.gelu(self.audio.conv2d2(features))
        features = F.gelu(self.audio.conv2d3(features))
        batch, channels, frequency, time = features.shape
        features = self.audio.conv_out(
            features.permute(0, 3, 1, 2).contiguous().view(batch, time, channels * frequency)
        )
        position = self.audio.positional_embedding.positional_embedding[: features.shape[1], :]
        features = features + position.unsqueeze(0).to(features.dtype)
        hidden_states = torch.index_select(features.reshape(-1, features.shape[-1]), 0, valid_indices)

        for layer in self.audio.layers:
            residual = hidden_states
            normalized = layer.self_attn_layer_norm(hidden_states)
            attended = layer.self_attn(
                hidden_states=normalized,
                cu_seqlens=cu_seqlens,
                attention_mask=None,
            )
            hidden_states = residual + attended
            residual = hidden_states
            hidden_states = layer.final_layer_norm(hidden_states)
            hidden_states = layer.fc2(layer.activation_fn(layer.fc1(hidden_states)))
            hidden_states = residual + hidden_states
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        hidden_states = self.audio.ln_post(hidden_states)
        hidden_states = self.audio.proj2(self.audio.act(self.audio.proj1(hidden_states)))
        return hidden_states


class ThinkerPrefillExportWrapper(nn.Module):
    """Full prompt pass with dense multimodal embedding injection and explicit tensor cache outputs."""

    def __init__(self, text_model: Qwen3OmniMoeThinkerTextModel, lm_head: nn.Linear, deepstack_count: int) -> None:
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head
        self.deepstack_count = deepstack_count

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        multimodal_embeddings: torch.Tensor,
        multimodal_mask: torch.Tensor,
        visual_position_mask: torch.Tensor,
        *deepstack_visual_features: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        token_embeddings = self.text_model.embed_tokens(input_ids)
        inputs_embeds = torch.where(
            multimodal_mask.unsqueeze(-1),
            multimodal_embeddings.to(token_embeddings.dtype),
            token_embeddings,
        )
        cache = DynamicCache(config=self.text_model.config)
        deepstack = list(deepstack_visual_features[: self.deepstack_count]) or None
        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
            visual_pos_masks=visual_position_mask,
            deepstack_visual_embeds=deepstack,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        flat_cache: list[torch.Tensor] = []
        for layer in outputs.past_key_values.layers:
            flat_cache.extend((layer.keys, layer.values))
        return (logits, *flat_cache)


def make_decode_dynamic_shapes(num_layers: int, max_past_length: int) -> tuple[Any, ...]:
    past = torch.export.Dim("past_sequence_length", min=1, max=max_past_length)
    cache_shapes = tuple({2: past} for _ in range(num_layers * 2))
    return ({}, {1: past + 1}, {}, {}, cache_shapes)


class ThinkerDecodeExportWrapper(nn.Module):
    """One-token decode pass with flattened K/V tensors as the public ONNX interface."""

    def __init__(self, text_model: Qwen3OmniMoeThinkerTextModel, lm_head: nn.Linear) -> None:
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head
        self.num_layers = len(text_model.layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        *past_key_values: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        cache_pairs = [
            (past_key_values[2 * index], past_key_values[2 * index + 1])
            for index in range(self.num_layers)
        ]
        cache = DynamicCache(ddp_cache_data=cache_pairs)
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        flat_cache: list[torch.Tensor] = []
        for layer in outputs.past_key_values.layers:
            flat_cache.extend((layer.keys, layer.values))
        return (logits, *flat_cache)


def make_tiny_vision_config() -> Qwen3OmniMoeVisionEncoderConfig:
    return Qwen3OmniMoeVisionEncoderConfig(
        depth=1,
        hidden_size=8,
        hidden_act="gelu",
        intermediate_size=16,
        num_heads=2,
        in_channels=3,
        patch_size=4,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=8,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0],
        initializer_range=0.02,
        attn_implementation="eager",
    )


def make_tiny_thinker_config() -> Qwen3OmniMoeThinkerConfig:
    return Qwen3OmniMoeThinkerConfig(
        audio_config=make_tiny_audio_config().to_dict(),
        vision_config=make_tiny_vision_config().to_dict(),
        text_config=make_tiny_text_config().to_dict(),
        audio_token_id=24,
        image_token_id=25,
        video_token_id=26,
        vision_start_token_id=27,
        vision_end_token_id=28,
        audio_start_token_id=29,
        audio_end_token_id=30,
        position_id_per_seconds=13,
        initializer_range=0.02,
    )


def make_tiny_audio_config() -> Qwen3OmniMoeAudioEncoderConfig:
    return Qwen3OmniMoeAudioEncoderConfig(
        num_mel_bins=16,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        d_model=8,
        dropout=0.0,
        attention_dropout=0.0,
        activation_function="gelu",
        activation_dropout=0.0,
        max_source_positions=64,
        n_window=8,
        output_dim=8,
        n_window_infer=16,
        conv_chunksize=8,
        downsample_hidden_size=4,
        initializer_range=0.02,
        attn_implementation="eager",
    )


def _vision_position_indices_weights(
    config: Qwen3OmniMoeVisionEncoderConfig, grid_thw: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    grid_side = int(config.num_position_embeddings**0.5)
    merge = config.spatial_merge_size
    all_indices = [[] for _ in range(4)]
    all_weights = [[] for _ in range(4)]
    for temporal, height, width in grid_thw.tolist():
        h_coords = torch.linspace(0, grid_side - 1, height)
        w_coords = torch.linspace(0, grid_side - 1, width)
        h_floor, w_floor = h_coords.int(), w_coords.int()
        h_ceil = (h_floor + 1).clip(max=grid_side - 1)
        w_ceil = (w_floor + 1).clip(max=grid_side - 1)
        dh, dw = h_coords - h_floor, w_coords - w_floor
        base_h, base_h_ceil = h_floor * grid_side, h_ceil * grid_side
        indices = [
            (base_h[:, None] + w_floor[None, :]).flatten(),
            (base_h[:, None] + w_ceil[None, :]).flatten(),
            (base_h_ceil[:, None] + w_floor[None, :]).flatten(),
            (base_h_ceil[:, None] + w_ceil[None, :]).flatten(),
        ]
        weights = [
            ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten(),
            ((1 - dh)[:, None] * dw[None, :]).flatten(),
            (dh[:, None] * (1 - dw)[None, :]).flatten(),
            (dh[:, None] * dw[None, :]).flatten(),
        ]
        for corner in range(4):
            index = indices[corner].repeat(temporal)
            weight = weights[corner].repeat(temporal)
            index = index.view(temporal, height // merge, merge, width // merge, merge).permute(0, 1, 3, 2, 4).flatten()
            weight = weight.view(temporal, height // merge, merge, width // merge, merge).permute(0, 1, 3, 2, 4).flatten()
            all_indices[corner].append(index)
            all_weights[corner].append(weight)
    return (
        torch.stack([torch.cat(values) for values in all_indices]),
        torch.stack([torch.cat(values) for values in all_weights]),
    )


def prepare_vision_inputs(
    vision: Qwen3OmniMoeVisionEncoder, pixel_values: torch.Tensor, grid_thw: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    indices, weights = _vision_position_indices_weights(vision.config, grid_thw)
    rotary = vision.rot_pos_emb(grid_thw).reshape(pixel_values.shape[0], -1)
    embedding = torch.cat((rotary, rotary), dim=-1)
    cos, sin = embedding.cos(), embedding.sin()
    segment_lengths = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    cumulative = F.pad(segment_lengths.cumsum(0), (1, 0), value=0)
    minimum = torch.finfo(pixel_values.dtype).min
    mask = torch.full((1, 1, pixel_values.shape[0], pixel_values.shape[0]), minimum, dtype=pixel_values.dtype)
    for index in range(1, len(cumulative)):
        start, end = int(cumulative[index - 1]), int(cumulative[index])
        mask[..., start:end, start:end] = 0
    return (
        pixel_values,
        indices.to(pixel_values.device),
        weights.to(device=pixel_values.device, dtype=vision.pos_embed.weight.dtype),
        cos.to(pixel_values.device),
        sin.to(pixel_values.device),
        mask.to(pixel_values.device),
    )


def prepare_audio_inputs(
    audio: Qwen3OmniMoeAudioEncoder, input_features: torch.Tensor, feature_lens: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    chunk_num = torch.ceil(feature_lens / (audio.n_window * 2)).long()
    chunk_lengths = torch.full(
        (int(chunk_num.sum()),),
        audio.n_window * 2,
        dtype=torch.long,
        device=feature_lens.device,
    )
    tail_indices = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_indices] = feature_lens % (audio.n_window * 2)
    chunk_lengths[chunk_lengths == 0] = audio.n_window * 2
    chunks = input_features.T.split(chunk_lengths.tolist(), dim=0)
    padded = nn.utils.rnn.pad_sequence(chunks, batch_first=True).transpose(1, 2)
    after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
    max_after_cnn = int(after_cnn.max())
    valid_mask = torch.arange(max_after_cnn, device=feature_lens.device)[None, :] < after_cnn[:, None]
    valid_indices = valid_mask.flatten().nonzero().squeeze(-1)

    total_after_cnn = _get_feat_extract_output_lengths(feature_lens)
    window_after_cnn = max_after_cnn * (audio.n_window_infer // (audio.n_window * 2))
    boundaries = [0]
    for length in total_after_cnn.tolist():
        full_windows, remainder = divmod(length, window_after_cnn)
        boundaries.extend([window_after_cnn] * full_windows)
        if remainder:
            boundaries.append(remainder)
    # dtype 必须显式传给 cumsum，否则 PyTorch 会把 int32 提升成 int64（与 HF 参考实现的 int32 契约不符）
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device=feature_lens.device).cumsum(
        0, dtype=torch.int32
    )
    return padded, valid_indices, cu_seqlens


def _move_tensors(values: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(value.to(device) for value in values)


def _make_tiny_text_modules(seed: int) -> tuple[Qwen3OmniMoeTextConfig, Qwen3OmniMoeThinkerTextModel, nn.Linear]:
    torch.manual_seed(seed)
    config = make_tiny_text_config()
    model = Qwen3OmniMoeThinkerTextModel(config).eval()
    lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).eval()
    nn.init.normal_(lm_head.weight, mean=0.0, std=config.initializer_range)
    return config, model, lm_head


def build_multimodal_prompt(
    thinker: Qwen3OmniMoeThinkerForConditionalGeneration,
    total_length: int,
    vision_grid: tuple[int, int, int],
    audio_feature_length: int,
    variant: int,
) -> tuple[list[int], list[int], list[int], torch.Tensor, torch.Tensor]:
    vision_tokens = math.prod(vision_grid) // (thinker.spatial_merge_size**2)
    audio_tokens = int(
        _get_feat_extract_output_lengths(torch.tensor([audio_feature_length], dtype=torch.int64))[0]
    )
    config = thinker.config
    prefix = [1 + variant]
    vision_positions = list(range(len(prefix) + 1, len(prefix) + 1 + vision_tokens))
    tokens = prefix + [config.vision_start_token_id] + [config.image_token_id] * vision_tokens
    tokens += [config.vision_end_token_id, config.audio_start_token_id]
    audio_positions = list(range(len(tokens), len(tokens) + audio_tokens))
    tokens += [config.audio_token_id] * audio_tokens + [config.audio_end_token_id]
    filler = [3 + ((index + variant * 3) % 16) for index in range(total_length - len(tokens))]
    tokens += filler
    if len(tokens) != total_length:
        raise ValueError(f"total_length={total_length} 不足以容纳多模态 placeholder")
    input_ids = torch.tensor([tokens], dtype=torch.int64)
    attention_mask = torch.ones_like(input_ids)
    position_ids, rope_delta = thinker.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=torch.tensor([vision_grid], dtype=torch.int64),
        attention_mask=attention_mask,
        audio_seqlens=torch.tensor([audio_feature_length], dtype=torch.int64),
    )
    return tokens, vision_positions, audio_positions, position_ids, rope_delta


def _prefill_inputs(
    config: Qwen3OmniMoeTextConfig,
    token_ids: list[int],
    scale: float,
    deepstack_count: int = 1,
    vision_positions: list[int] | None = None,
    audio_positions: list[int] | None = None,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    input_ids = torch.tensor([token_ids], dtype=torch.int64)
    sequence = input_ids.shape[1]
    vision_positions = vision_positions or [1, 2, 3, 4]
    audio_positions = audio_positions or [5, 6, 7]
    all_multimodal_positions = [*vision_positions, *audio_positions]
    if not all_multimodal_positions or max(all_multimodal_positions) >= sequence:
        raise ValueError("多模态 placeholder 位置超出 Prefill 序列")
    attention_mask = torch.ones_like(input_ids)
    cache_position = torch.arange(sequence, dtype=torch.int64)
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, 1, -1).contiguous()
    multimodal_embeddings = torch.zeros(1, sequence, config.hidden_size)
    values = torch.linspace(
        -scale,
        scale,
        len(all_multimodal_positions) * config.hidden_size,
    ).view(len(all_multimodal_positions), config.hidden_size)
    multimodal_embeddings[:, all_multimodal_positions, :] = values
    multimodal_mask = torch.zeros(1, sequence, dtype=torch.bool)
    multimodal_mask[:, all_multimodal_positions] = True
    visual_position_mask = torch.zeros(1, sequence, dtype=torch.bool)
    visual_position_mask[:, vision_positions] = True
    deepstack = tuple(
        torch.linspace(
            scale * (index + 1),
            -scale * (index + 1),
            len(vision_positions) * config.hidden_size,
        ).view(len(vision_positions), -1)
        for index in range(deepstack_count)
    )
    return (
        input_ids,
        attention_mask,
        position_ids,
        cache_position,
        multimodal_embeddings,
        multimodal_mask,
        visual_position_mask,
        *deepstack,
    )


def reference_prefill(
    text_model: Qwen3OmniMoeThinkerTextModel,
    lm_head: nn.Linear,
    args: tuple[torch.Tensor, ...],
    deepstack_count: int,
) -> tuple[torch.Tensor, ...]:
    (
        input_ids,
        attention_mask,
        position_ids,
        cache_position,
        multimodal_embeddings,
        multimodal_mask,
        visual_position_mask,
        *deepstack,
    ) = args
    token_embeddings = text_model.embed_tokens(input_ids)
    inputs_embeds = torch.where(
        multimodal_mask.unsqueeze(-1),
        multimodal_embeddings.to(token_embeddings.dtype),
        token_embeddings,
    )
    cache = DynamicCache(config=text_model.config)
    outputs = text_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
        cache_position=cache_position,
        visual_pos_masks=visual_position_mask,
        deepstack_visual_embeds=list(deepstack[:deepstack_count]),
    )
    flattened: list[torch.Tensor] = [lm_head(outputs.last_hidden_state)]
    for layer in outputs.past_key_values.layers:
        flattened.extend((layer.keys, layer.values))
    return tuple(flattened)


def reference_decode(
    text_model: Qwen3OmniMoeThinkerTextModel,
    lm_head: nn.Linear,
    args: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    input_ids, attention_mask, position_ids, cache_position, *flat_cache = args
    cache = DynamicCache(
        ddp_cache_data=[
            (flat_cache[2 * index], flat_cache[2 * index + 1])
            for index in range(len(text_model.layers))
        ]
    )
    outputs = text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
        cache_position=cache_position,
    )
    flattened: list[torch.Tensor] = [lm_head(outputs.last_hidden_state)]
    for layer in outputs.past_key_values.layers:
        flattened.extend((layer.keys, layer.values))
    return tuple(flattened)


def _assert_close_tuple(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...], label: str) -> float:
    if len(actual) != len(expected):
        raise RuntimeError(f"{label} 输出数量不一致：{len(actual)} != {len(expected)}")
    max_error = 0.0
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-4, atol=1e-5)
        difference = (actual_tensor - expected_tensor).abs()
        max_error = max(max_error, float(difference.max()) if difference.numel() else 0.0)
    return max_error


def build_tiny_thinking_component(component: str, seed: int = DEFAULT_SEED) -> ThinkingComponentCase:
    if component not in THINKING_COMPONENTS:
        raise ValueError(f"未知 Thinking 组件 {component!r}")
    assert_transformers_provenance()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if component == "vision_encoder":
        config = make_tiny_vision_config()
        vision = Qwen3OmniMoeVisionEncoder(config).eval()
        wrapper = VisionEncoderExportWrapper(vision).eval()
        grids = (
            torch.tensor([[1, 4, 4]], dtype=torch.int64),
            torch.tensor([[1, 2, 4], [1, 2, 4]], dtype=torch.int64),
            torch.tensor([[2, 2, 4]], dtype=torch.int64),
        )
        first_pixels = torch.randn(16, 3 * 2 * 4 * 4)
        pixels = (first_pixels, first_pixels * 0.7 + 0.1, first_pixels * -0.3 + 0.2)
        vectors = tuple(
            prepare_vision_inputs(vision, pixel_values, grid)
            for pixel_values, grid in zip(pixels, grids)
        )
        output_names = ("vision_embeddings", "deepstack_visual_0")
        errors = []
        with torch.inference_mode():
            for pixel_values, grid, vector in zip(pixels, grids, vectors):
                original = vision(pixel_values, grid)
                wrapped = normalize_outputs(wrapper(*vector))
                expected = (original.pooler_output, *original.deepstack_features)
                errors.append(_assert_close_tuple(wrapped, expected, "Vision wrapper"))
        error = max(errors)
        return ThinkingComponentCase(
            name=component,
            model=wrapper,
            test_vectors=vectors,
            input_names=(
                "pixel_values",
                "position_indices",
                "position_weights",
                "rotary_cos",
                "rotary_sin",
                "attention_mask",
            ),
            output_names=output_names,
            description="Thinking Vision Encoder；grid 相关 Python 预处理已转换为显式张量接口",
            config=config.to_dict(),
            interface={
                "grid_profiles": [[1, 4, 4], [[1, 2, 4], [1, 2, 4]], [2, 2, 4]],
                "host_preprocessing": True,
            },
            source_equivalence={"checked": True, "max_abs_error": error},
        )

    if component == "audio_encoder":
        config = make_tiny_audio_config()
        audio = Qwen3OmniMoeAudioEncoder(config).eval()
        wrapper = AudioEncoderExportWrapper(audio).eval()
        feature_lens = torch.tensor([20], dtype=torch.int64)
        first_features = torch.randn(config.num_mel_bins, 20)
        second_features = first_features * -0.4 + 0.2
        vectors = (
            prepare_audio_inputs(audio, first_features, feature_lens),
            prepare_audio_inputs(audio, second_features, feature_lens),
        )
        errors = []
        with torch.inference_mode():
            for features, vector in zip((first_features, second_features), vectors):
                original = normalize_outputs(audio(features, feature_lens=feature_lens).last_hidden_state)
                wrapped = normalize_outputs(wrapper(*vector))
                errors.append(_assert_close_tuple(wrapped, original, "Audio wrapper"))
        error = max(errors)
        return ThinkingComponentCase(
            name=component,
            model=wrapper,
            test_vectors=vectors,
            input_names=("padded_features", "valid_indices", "cu_seqlens"),
            output_names=("audio_embeddings",),
            description="Thinking Audio Encoder；变长分块与 mask 构造由宿主预处理",
            config=config.to_dict(),
            interface={
                "feature_length_profile": 20,
                "host_preprocessing": True,
                "chunk_count": int(vectors[0][0].shape[0]),
            },
            source_equivalence={"checked": True, "max_abs_error": error},
        )

    config, text_model, lm_head = _make_tiny_text_modules(seed)
    position_helper = Qwen3OmniMoeThinkerForConditionalGeneration(make_tiny_thinker_config()).eval()
    vision_grid = (1, 4, 4)
    audio_feature_length = 20
    first_prompt = build_multimodal_prompt(position_helper, 12, vision_grid, audio_feature_length, 0)
    second_prompt = build_multimodal_prompt(position_helper, 12, vision_grid, audio_feature_length, 1)
    if component == "thinker_prefill":
        wrapper = ThinkerPrefillExportWrapper(text_model, lm_head, deepstack_count=1).eval()
        vectors = (
            _prefill_inputs(
                config,
                first_prompt[0],
                0.15,
                1,
                first_prompt[1],
                first_prompt[2],
                first_prompt[3],
            ),
            _prefill_inputs(
                config,
                second_prompt[0],
                0.35,
                1,
                second_prompt[1],
                second_prompt[2],
                second_prompt[3],
            ),
        )
        output_names = ["logits"]
        for layer_index in range(config.num_hidden_layers):
            output_names.extend((f"present_key_{layer_index}", f"present_value_{layer_index}"))
        equivalence_errors = []
        with torch.inference_mode():
            for vector in vectors:
                wrapped = normalize_outputs(wrapper(*vector))
                reference = reference_prefill(text_model, lm_head, vector, deepstack_count=1)
                equivalence_errors.append(_assert_close_tuple(wrapped, reference, "Thinker Prefill wrapper"))
        return ThinkingComponentCase(
            name=component,
            model=wrapper,
            test_vectors=vectors,
            input_names=(
                "input_ids",
                "attention_mask",
                "position_ids",
                "cache_position",
                "multimodal_embeddings",
                "multimodal_mask",
                "visual_position_mask",
                "deepstack_visual_0",
            ),
            output_names=tuple(output_names),
            description="Thinking Thinker Prefill；支持 dense 多模态 embedding、DeepStack 与显式 K/V 输出",
            config=config.to_dict(),
            interface={
                "sequence_profile": 12,
                "cache_outputs": config.num_hidden_layers * 2,
                "mrope_source": "Qwen3OmniMoeThinkerForConditionalGeneration.get_rope_index",
                "rope_deltas": [float(first_prompt[4].item()), float(second_prompt[4].item())],
            },
            source_equivalence={"checked": True, "max_abs_error": max(equivalence_errors)},
        )

    prefill = ThinkerPrefillExportWrapper(text_model, lm_head, deepstack_count=1).eval()
    decode = ThinkerDecodeExportWrapper(text_model, lm_head).eval()
    second_decode_prompt = build_multimodal_prompt(position_helper, 14, vision_grid, audio_feature_length, 1)
    prefill_vectors = (
        _prefill_inputs(
            config,
            first_prompt[0],
            0.15,
            1,
            first_prompt[1],
            first_prompt[2],
            first_prompt[3],
        ),
        _prefill_inputs(
            config,
            second_decode_prompt[0],
            0.35,
            1,
            second_decode_prompt[1],
            second_decode_prompt[2],
            second_decode_prompt[3],
        ),
    )
    rope_deltas = (first_prompt[4], second_decode_prompt[4])
    decode_vectors = []
    for index, (prefill_input, rope_delta) in enumerate(zip(prefill_vectors, rope_deltas)):
        with torch.inference_mode():
            prefill_output = normalize_outputs(prefill(*prefill_input))
        next_token = torch.tensor([[6 + index]], dtype=torch.int64)
        past_length = prefill_input[0].shape[1]
        decode_mask = torch.ones(1, past_length + 1, dtype=torch.int64)
        decode_position_value = past_length + int(rope_delta.item())
        # 与 Prefill 保持一致：Prefill 的 position_ids 来自官方 get_rope_index，是 float32
        decode_position = torch.full((3, 1, 1), decode_position_value, dtype=torch.float32)
        decode_cache_position = torch.tensor([past_length], dtype=torch.int64)
        decode_vectors.append(
            (next_token, decode_mask, decode_position, decode_cache_position, *prefill_output[1:])
        )
    input_names = ["input_ids", "attention_mask", "position_ids", "cache_position"]
    output_names = ["logits"]
    for layer_index in range(config.num_hidden_layers):
        input_names.extend((f"past_key_{layer_index}", f"past_value_{layer_index}"))
        output_names.extend((f"present_key_{layer_index}", f"present_value_{layer_index}"))
    equivalence_errors = []
    with torch.inference_mode():
        for vector in decode_vectors:
            wrapped = normalize_outputs(decode(*vector))
            reference = reference_decode(text_model, lm_head, vector)
            equivalence_errors.append(_assert_close_tuple(wrapped, reference, "Thinker Decode wrapper"))
    return ThinkingComponentCase(
        name=component,
        model=decode,
        test_vectors=tuple(decode_vectors),
        input_names=tuple(input_names),
        output_names=tuple(output_names),
        description="Thinking Thinker Decode；单 token 输入与显式 flattened K/V 输入输出",
        config=config.to_dict(),
        interface={
            "past_sequence_profiles_tested": [12, 14],
            "past_sequence_dynamic_max": 64,
            "decode_sequence": 1,
            "cache_tensors": config.num_hidden_layers * 2,
        },
        source_equivalence={"checked": True, "max_abs_error": max(equivalence_errors)},
        dynamic_shapes=make_decode_dynamic_shapes(config.num_hidden_layers, max_past_length=64),
    )


def component_routing(model: nn.Module, args: tuple[torch.Tensor, ...]) -> dict[str, Any] | None:
    return capture_routing(model, args)


def fingerprint_checkpoint(model_path: Path) -> dict[str, Any]:
    model_path = model_path.expanduser().resolve()
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("官方 checkpoint 必须包含 config.json 和 model.safetensors.index.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_omni_moe" or config.get("enable_audio_output") is not False:
        raise ValueError("checkpoint 不是 enable_audio_output=false 的 Qwen3-Omni Thinking 模型")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    if not shard_names:
        raise ValueError("权重索引中没有 safetensors shard")
    shards = []
    for name in shard_names:
        path = (model_path / name).resolve()
        path.relative_to(model_path)
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"权重分片缺失或为空：{path}")
        shards.append({"name": name, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    return {
        "model_path": str(model_path),
        "config_sha256": file_sha256(config_path),
        "index_sha256": file_sha256(index_path),
        "weight_file_count": len(shards),
        "weight_bytes": sum(item["bytes"] for item in shards),
        "shards": shards,
    }


def load_real_thinking_model(
    model_path: str,
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
) -> Qwen3OmniMoeForConditionalGeneration:
    """Load and fingerprint a local Thinking checkpoint on a large-memory host."""
    assert_transformers_provenance()
    checkpoint_path = Path(model_path).expanduser().resolve()
    checkpoint_fingerprint = fingerprint_checkpoint(checkpoint_path)
    load_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
        "experts_implementation": "batched_mm",
    }
    if device != "cpu":
        load_kwargs["device_map"] = device
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        str(checkpoint_path),
        **load_kwargs,
    ).eval()
    if model.config.enable_audio_output or model.has_talker:
        raise ValueError("该导出器只接受 enable_audio_output=false 的 Thinking checkpoint")
    model.requires_grad_(False)
    model._onnx_checkpoint_provenance = checkpoint_fingerprint
    return model


def build_real_thinking_component(
    full_model: Qwen3OmniMoeForConditionalGeneration,
    component: str,
    seed: int = DEFAULT_SEED,
    text_sequence_length: int = 32,
    vision_grid: tuple[int, int, int] = (1, 4, 4),
) -> ThinkingComponentCase:
    """Build an official-weight component case. Intended for the large-memory Linux stage."""
    if component not in THINKING_COMPONENTS:
        raise ValueError(f"未知 Thinking 组件 {component!r}")
    assert_transformers_provenance()
    torch.manual_seed(seed)
    np.random.seed(seed)
    thinker = full_model.thinker

    if component == "vision_encoder":
        vision = thinker.visual.eval()
        wrapper = VisionEncoderExportWrapper(vision).eval()
        temporal, height, width = vision_grid
        if height % vision.spatial_merge_size or width % vision.spatial_merge_size:
            raise ValueError("vision grid 的 height/width 必须能被 spatial_merge_size 整除")
        rows = temporal * height * width
        columns = (
            vision.config.in_channels
            * vision.config.temporal_patch_size
            * vision.config.patch_size
            * vision.config.patch_size
        )
        grid = torch.tensor([vision_grid], dtype=torch.int64)
        first = torch.randn(rows, columns, dtype=vision.dtype, device=vision.device)
        second = first * 0.7 + 0.1
        vectors = (
            prepare_vision_inputs(vision, first, grid.to(vision.device)),
            prepare_vision_inputs(vision, second, grid.to(vision.device)),
        )
        equivalence_errors = []
        with torch.inference_mode():
            for pixels, vector in zip((first, second), vectors):
                original = vision(pixels, grid.to(vision.device))
                expected = (original.pooler_output, *original.deepstack_features)
                wrapped = normalize_outputs(wrapper(*vector))
                equivalence_errors.append(_assert_close_tuple(wrapped, expected, "Real Vision wrapper"))
        output_names = ("vision_embeddings",) + tuple(
            f"deepstack_visual_{index}" for index in range(len(vision.deepstack_visual_indexes))
        )
        return ThinkingComponentCase(
            name=component,
            model=wrapper,
            test_vectors=vectors,
            input_names=(
                "pixel_values",
                "position_indices",
                "position_weights",
                "rotary_cos",
                "rotary_sin",
                "attention_mask",
            ),
            output_names=output_names,
            description="官方权重 Thinking Vision Encoder",
            config=vision.config.to_dict(),
            interface={"grid_profile": list(vision_grid), "host_preprocessing": True},
            source_equivalence={"checked": True, "max_abs_error": max(equivalence_errors)},
            checkpoint_fingerprint=getattr(full_model, "_onnx_checkpoint_provenance", None),
        )

    if component == "audio_encoder":
        audio = thinker.audio_tower.eval()
        wrapper = AudioEncoderExportWrapper(audio).eval()
        feature_length = audio.n_window * 2 + 1
        feature_lens = torch.tensor([feature_length], dtype=torch.int64, device=audio.device)
        first = torch.randn(audio.num_mel_bins, feature_length, dtype=audio.dtype, device=audio.device)
        second = first * -0.4 + 0.2
        vectors = (
            prepare_audio_inputs(audio, first, feature_lens),
            prepare_audio_inputs(audio, second, feature_lens),
        )
        equivalence_errors = []
        with torch.inference_mode():
            for features, vector in zip((first, second), vectors):
                original = normalize_outputs(audio(features, feature_lens=feature_lens).last_hidden_state)
                wrapped = normalize_outputs(wrapper(*vector))
                equivalence_errors.append(_assert_close_tuple(wrapped, original, "Real Audio wrapper"))
        return ThinkingComponentCase(
            name=component,
            model=wrapper,
            test_vectors=vectors,
            input_names=("padded_features", "valid_indices", "cu_seqlens"),
            output_names=("audio_embeddings",),
            description="官方权重 Thinking Audio Encoder",
            config=audio.config.to_dict(),
            interface={
                "feature_length_profile": feature_length,
                "host_preprocessing": True,
                "chunk_count": int(vectors[0][0].shape[0]),
            },
            source_equivalence={"checked": True, "max_abs_error": max(equivalence_errors)},
            checkpoint_fingerprint=getattr(full_model, "_onnx_checkpoint_provenance", None),
        )

    text_model = thinker.model.eval()
    lm_head = thinker.lm_head.eval()
    text_config = text_model.config
    deepstack_count = len(thinker.visual.deepstack_visual_indexes)
    vision_tokens = (
        vision_grid[0]
        * vision_grid[1]
        * vision_grid[2]
        // (thinker.visual.spatial_merge_size**2)
    )
    # 守卫必须用与实际建 prompt 完全相同的 audio_feature_length，否则会放行随后必然失败的序列长度
    audio_feature_length = thinker.audio_tower.n_window * 2 + 1
    audio_tokens = int(
        _get_feat_extract_output_lengths(
            torch.tensor([audio_feature_length], dtype=torch.int64)
        )[0]
    )
    # build_multimodal_prompt 实际需要：prefix(1) + vision_start(1) + V + vision_end(1)
    #                                + audio_start(1) + A + audio_end(1) = V + A + 5
    minimum_length = vision_tokens + audio_tokens + 5
    if text_sequence_length < minimum_length:
        raise ValueError(
            f"text_sequence_length={text_sequence_length} 无法容纳 {vision_tokens} 个视觉和 "
            f"{audio_tokens} 个音频 token（至少需要 {minimum_length}）"
        )
    first_prompt = build_multimodal_prompt(
        thinker,
        text_sequence_length,
        vision_grid,
        audio_feature_length,
        0,
    )
    second_prompt = build_multimodal_prompt(
        thinker,
        text_sequence_length,
        vision_grid,
        audio_feature_length,
        1,
    )
    second_decode_prompt = build_multimodal_prompt(
        thinker,
        text_sequence_length + 2,
        vision_grid,
        audio_feature_length,
        1,
    )
    if component == "thinker_prefill":
        wrapper = ThinkerPrefillExportWrapper(text_model, lm_head, deepstack_count).eval()
        vectors = (
            _move_tensors(
                _prefill_inputs(
                    text_config,
                    first_prompt[0],
                    0.15,
                    deepstack_count,
                    first_prompt[1],
                    first_prompt[2],
                    first_prompt[3],
                ),
                text_model.device,
            ),
            _move_tensors(
                _prefill_inputs(
                    text_config,
                    second_prompt[0],
                    0.35,
                    deepstack_count,
                    second_prompt[1],
                    second_prompt[2],
                    second_prompt[3],
                ),
                text_model.device,
            ),
        )
        input_names = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "cache_position",
            "multimodal_embeddings",
            "multimodal_mask",
            "visual_position_mask",
        ] + [f"deepstack_visual_{index}" for index in range(deepstack_count)]
        output_names = ["logits"]
        for layer_index in range(text_config.num_hidden_layers):
            output_names.extend((f"present_key_{layer_index}", f"present_value_{layer_index}"))
        equivalence_errors = []
        with torch.inference_mode():
            for vector in vectors:
                wrapped = normalize_outputs(wrapper(*vector))
                reference = reference_prefill(text_model, lm_head, vector, deepstack_count)
                equivalence_errors.append(_assert_close_tuple(wrapped, reference, "Real Thinker Prefill wrapper"))
        return ThinkingComponentCase(
            name=component,
            model=wrapper,
            test_vectors=vectors,
            input_names=tuple(input_names),
            output_names=tuple(output_names),
            description="官方权重 Thinking Thinker Prefill",
            config=text_config.to_dict(),
            interface={
                "sequence_profile": text_sequence_length,
                "cache_outputs": text_config.num_hidden_layers * 2,
                "mrope_source": "Qwen3OmniMoeThinkerForConditionalGeneration.get_rope_index",
                "rope_deltas": [float(first_prompt[4].item()), float(second_prompt[4].item())],
            },
            source_equivalence={
                "checked": True,
                "scope": "official_mrope_and_text_backbone",
                "max_abs_error": max(equivalence_errors),
            },
            checkpoint_fingerprint=getattr(full_model, "_onnx_checkpoint_provenance", None),
        )

    prefill = ThinkerPrefillExportWrapper(text_model, lm_head, deepstack_count).eval()
    decode = ThinkerDecodeExportWrapper(text_model, lm_head).eval()
    prefill_vectors = (
        _move_tensors(
            _prefill_inputs(
                text_config,
                first_prompt[0],
                0.15,
                deepstack_count,
                first_prompt[1],
                first_prompt[2],
                first_prompt[3],
            ),
            text_model.device,
        ),
        _move_tensors(
            _prefill_inputs(
                text_config,
                second_decode_prompt[0],
                0.35,
                deepstack_count,
                second_decode_prompt[1],
                second_decode_prompt[2],
                second_decode_prompt[3],
            ),
            text_model.device,
        ),
    )
    decode_vectors = []
    rope_deltas = (first_prompt[4], second_decode_prompt[4])
    for index, (prefill_input, rope_delta) in enumerate(zip(prefill_vectors, rope_deltas)):
        with torch.inference_mode():
            prefill_output = normalize_outputs(prefill(*prefill_input))
        next_token = torch.tensor([[6 + index]], dtype=torch.int64, device=text_model.device)
        past_length = prefill_input[0].shape[1]
        decode_mask = torch.ones(1, past_length + 1, dtype=torch.int64, device=text_model.device)
        decode_position_value = past_length + int(rope_delta.item())
        decode_position = torch.full((3, 1, 1), decode_position_value, dtype=torch.int64, device=text_model.device)
        decode_cache_position = torch.tensor([past_length], dtype=torch.int64, device=text_model.device)
        decode_vectors.append(
            (next_token, decode_mask, decode_position, decode_cache_position, *prefill_output[1:])
        )
    input_names = ["input_ids", "attention_mask", "position_ids", "cache_position"]
    output_names = ["logits"]
    for layer_index in range(text_config.num_hidden_layers):
        input_names.extend((f"past_key_{layer_index}", f"past_value_{layer_index}"))
        output_names.extend((f"present_key_{layer_index}", f"present_value_{layer_index}"))
    equivalence_errors = []
    with torch.inference_mode():
        for vector in decode_vectors:
            wrapped = normalize_outputs(decode(*vector))
            reference = reference_decode(text_model, lm_head, vector)
            equivalence_errors.append(_assert_close_tuple(wrapped, reference, "Real Thinker Decode wrapper"))
    return ThinkingComponentCase(
        name=component,
        model=decode,
        test_vectors=tuple(decode_vectors),
        input_names=tuple(input_names),
        output_names=tuple(output_names),
        description="官方权重 Thinking Thinker Decode",
        config=text_config.to_dict(),
        interface={
            "past_sequence_profiles_tested": [text_sequence_length, text_sequence_length + 2],
            "past_sequence_dynamic_max": text_config.max_position_embeddings - 1,
            "decode_sequence": 1,
            "cache_tensors": text_config.num_hidden_layers * 2,
        },
        source_equivalence={"checked": True, "max_abs_error": max(equivalence_errors)},
        dynamic_shapes=make_decode_dynamic_shapes(
            text_config.num_hidden_layers,
            max_past_length=text_config.max_position_embeddings - 1,
        ),
        checkpoint_fingerprint=getattr(full_model, "_onnx_checkpoint_provenance", None),
    )
