from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig

import qwen3_omni_thinking_components as components
import export_thinking_onnx as exporter
import run_local_thinking_pipeline as local
import run_real_thinking_pipeline as real
import onnx_artifact_utils as utils

GIB = 1024**3


class RunnerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix=".test-runners-", dir=utils.WORKSPACE)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        self.config = {"model_type": "qwen3_omni_moe", "enable_audio_output": False,
                       "thinker_config": components.make_tiny_thinker_config().to_dict()}
        (self.checkpoint / "config.json").write_text(json.dumps(self.config))
        save_file({"sample": torch.zeros(2)}, str(self.checkpoint / "part.safetensors"))
        self.index = {"metadata": {"total_size": 8}, "weight_map": {"sample": "part.safetensors"}}
        self.index_path = self.checkpoint / "model.safetensors.index.json"
        self.index_path.write_text(json.dumps(self.index))
        for filename in ("generation_config.json", "chat_template.json", "merges.txt", "preprocessor_config.json", "tokenizer_config.json", "vocab.json"):
            (self.checkpoint / filename).write_text("{}")
        self.args = argparse.Namespace(model_path=self.checkpoint, package_dir=self.root / "product", dtype="float16",
                                       device="cpu", provider="CPUExecutionProvider", minimum_memory_gib=192.,
                                       run_end_to_end=True, source_dir=None, offline=True, preflight_only=True)

    def resources(self, **overrides):
        info = {"config": self.config, "weight_bytes": 60 * GIB}
        args = dict(checkpoint=info, package_dir=self.args.package_dir, minimum_memory_gib=192.,
                    device="cpu", provider="CPUExecutionProvider", end_to_end=True)
        args.update(overrides)
        return utils.check_execution_resources(**args)

    def test_preflight_checks_actual_headers_and_resources_without_model_loading(self):
        with patch("psutil.virtual_memory", return_value=SimpleNamespace(total=256*GIB, available=200*GIB)) as ram, patch(
            "shutil.disk_usage", return_value=SimpleNamespace(free=400*GIB)
        ) as disk, patch.object(components, "load_real_thinking_model", side_effect=AssertionError("must not load")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(real.preflight(self.args), self.checkpoint)
        ram.assert_called_once(); disk.assert_called_once()
        self.assertEqual(disk.call_args.args[0], self.root)
        self.assertFalse(self.args.package_dir.exists())

    def test_missing_or_corrupt_shard_rejected(self):
        for data in (b"", b"invalid header"):
            (self.checkpoint / "part.safetensors").write_bytes(data)
            with self.assertRaises(Exception):
                components.inspect_checkpoint(self.checkpoint)
        (self.checkpoint / "part.safetensors").unlink()
        with self.assertRaises(FileNotFoundError):
            components.inspect_checkpoint(self.checkpoint)

    def test_index_header_mismatch_and_traversal_rejected(self):
        for mapping in ({"not-present": "part.safetensors"}, {"sample": "../part.safetensors"}):
            self.index["weight_map"] = mapping
            self.index_path.write_text(json.dumps(self.index))
            with self.assertRaises(ValueError):
                components.inspect_checkpoint(self.checkpoint)

    def test_invalid_size_and_model_config_rejected(self):
        for value in (None, True, -1, 0, "8", 999999):
            self.index["metadata"]["total_size"] = value
            self.index_path.write_text(json.dumps(self.index))
            with self.subTest(value=value), self.assertRaises(ValueError):
                components.inspect_checkpoint(self.checkpoint)
        self.index["metadata"]["total_size"] = 8
        self.index_path.write_text(json.dumps(self.index))
        self.config["enable_audio_output"] = True
        (self.checkpoint / "config.json").write_text(json.dumps(self.config))
        with self.assertRaises(ValueError):
            components.inspect_checkpoint(self.checkpoint)

    def test_e2e_does_not_reserve_export_space_twice(self):
        with patch("psutil.virtual_memory", return_value=SimpleNamespace(total=256*GIB, available=200*GIB)), patch(
            "shutil.disk_usage", return_value=SimpleNamespace(free=40*GIB)
        ):
            self.resources(export_stage=False)
            with self.assertRaisesRegex(RuntimeError, "输出磁盘"):
                self.resources(export_stage=True)

    def test_ram_total_and_available_are_gates(self):
        for total, available in ((48, 40), (256, 10)):
            with patch("psutil.virtual_memory", return_value=SimpleNamespace(total=total*GIB, available=available*GIB)):
                with self.assertRaisesRegex(RuntimeError, "内存不足"):
                    self.resources()

    def test_output_disk_not_checkpoint_disk_is_checked(self):
        with patch("psutil.virtual_memory", return_value=SimpleNamespace(total=256*GIB, available=200*GIB)), patch(
            "shutil.disk_usage", return_value=SimpleNamespace(free=1)
        ) as disk:
            with self.assertRaisesRegex(RuntimeError, "输出磁盘"):
                self.resources()
            self.assertEqual(disk.call_args.args[0], self.root)

    def test_cuda_unavailable_index_and_double_residency_rejected(self):
        with patch("psutil.virtual_memory", return_value=SimpleNamespace(total=256*GIB, available=200*GIB)):
            with patch("torch.cuda.is_available", return_value=False), self.assertRaisesRegex(RuntimeError, "CUDA"):
                self.resources(device="cuda:99")
            with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=1):
                with self.assertRaisesRegex(RuntimeError, "不存在"):
                    self.resources(device="cuda:99")
                with patch("torch.cuda.mem_get_info", return_value=(80*GIB,80*GIB)), patch.object(utils, "check_providers"):
                    with self.assertRaisesRegex(RuntimeError, "显存"):
                        self.resources(device="cuda:0", provider="CUDAExecutionProvider")

    def test_invalid_memory_thresholds_rejected(self):
        for threshold in (float("nan"), float("inf"), -1, 96, True, 128):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                self.resources(minimum_memory_gib=threshold)

    def test_explicit_source_directory_reaches_build(self):
        source = self.root / "processor"
        shutil.copytree(self.checkpoint, source)
        self.args.source_dir = source; self.args.preflight_only = False
        with patch.object(real, "parse_args", return_value=self.args), patch.object(real, "preflight", return_value=source), patch.object(
            real, "run"
        ), patch.object(real.subprocess, "run", return_value=SimpleNamespace(returncode=1)) as process, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                real.main()
        command = process.call_args.args[0]
        self.assertEqual(command[command.index("--source-dir")+1], str(source))

    def test_preflight_only_does_not_run_subprocess(self):
        with patch.object(real, "parse_args", return_value=self.args), patch.object(real, "preflight"), patch.object(real, "run") as run, patch.object(
            real.subprocess, "run", side_effect=AssertionError("must not execute")
        ), contextlib.redirect_stdout(io.StringIO()):
            real.main()
        run.assert_not_called()

    def test_local_guard_sees_real_e2e_and_unknown_model(self):
        root = self.args.package_dir
        (root / "validation").mkdir(parents=True)
        (root / "validation/end_to_end.json").write_text(json.dumps({"profile": "real-fixed-shape"}))
        self.assertTrue(local.existing_real_evidence(root))
        (root / "validation/end_to_end.json").unlink()
        (root / "onnx/vision_encoder").mkdir(parents=True)
        (root / "onnx/vision_encoder/model.onnx").write_bytes(b"unknown")
        self.assertTrue(local.existing_real_evidence(root))

    def test_local_explicit_force_only_grants_cross_mode_permission(self):
        for force in (False, True):
            args = argparse.Namespace(package_dir=self.args.package_dir, force=force, source_dir=None, offline=True)
            with patch.object(local, "parse_args", return_value=args), patch.object(local, "existing_real_evidence", return_value=[]), patch.object(
                local, "resolve_package_sources", return_value={}
            ), patch.object(local, "run", side_effect=RuntimeError("stop")) as run:
                with self.assertRaises(RuntimeError):
                    local.main()
                self.assertEqual("--allow-real-overwrite" in run.call_args.args, force)

    def test_local_cold_cache_stops_before_export(self):
        args = argparse.Namespace(package_dir=self.args.package_dir, force=False, source_dir=None, offline=True)
        with patch.object(local, "parse_args", return_value=args), patch.object(local, "resolve_package_sources", side_effect=FileNotFoundError("cold cache")), patch.object(local, "run") as run:
            with self.assertRaises(FileNotFoundError):
                local.main()
            run.assert_not_called()

    def test_loading_info_all_error_categories_rejected(self):
        for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "conversion_errors", "error_msgs"):
            info = {"missing_keys": [], field: ["broken"]}
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                components.check_loading_info(info)
        components.check_loading_info({"missing_keys": [], "unexpected_keys": [], "error_msgs": []})

    def test_missing_weight_cannot_be_silently_randomized(self):
        config = Qwen3OmniMoeConfig(thinker_config=components.make_tiny_thinker_config().to_dict(), enable_audio_output=False)
        config._attn_implementation = "eager"; config._experts_implementation = "batched_mm"
        model = components.Qwen3OmniMoeForConditionalGeneration(config).eval()
        weights = {key: value.detach().contiguous() for key, value in model.state_dict().items()}
        del weights["thinker.lm_head.weight"]
        folder = self.root / "incomplete"
        folder.mkdir()
        config.to_json_file(folder / "config.json")
        save_file(weights, str(folder / "part.safetensors"))
        index = {"metadata": {"total_size": sum(v.numel()*v.element_size() for v in weights.values())},
                 "weight_map": {key: "part.safetensors" for key in weights}}
        (folder / "model.safetensors.index.json").write_text(json.dumps(index))
        with self.assertRaisesRegex(RuntimeError, "未完整加载"):
            components.load_real_thinking_model(str(folder), dtype=torch.float16)
        weights["thinker.lm_head.weight"] = model.lm_head.weight if hasattr(model, "lm_head") else model.thinker.lm_head.weight.detach().contiguous()
        save_file(weights, str(folder / "part.safetensors"))
        index["weight_map"]["thinker.lm_head.weight"] = "part.safetensors"
        index["metadata"]["total_size"] = sum(v.numel()*v.element_size() for v in weights.values())
        (folder / "model.safetensors.index.json").write_text(json.dumps(index))
        loaded = components.load_real_thinking_model(str(folder), dtype=torch.float16)
        self.assertTrue(loaded._onnx_loading_verified)
        torch.testing.assert_close(loaded.thinker.lm_head.weight, weights["thinker.lm_head.weight"].half(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
