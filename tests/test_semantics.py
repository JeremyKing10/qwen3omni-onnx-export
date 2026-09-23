"""Semantic regressions; small-config-real-code-path is not an official 30B validation."""
from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import ml_dtypes
import numpy as np
import onnxruntime as ort
import torch
from onnx import TensorProto, helper

import qwen3_omni_thinking_components as components
import validate_thinking_pipeline as pipeline
from onnx_artifact_utils import run_ort_session
from qwen3_omni_onnx_cases import WORKSPACE, tensor_to_numpy


class SemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.cases = {name: components.build_tiny_thinking_component(name) for name in components.THINKING_COMPONENTS}

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_outputs_close(self, actual, expected, *, exact=False):
        self.assertEqual(len(actual), len(expected))
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=0 if exact else 1e-4, atol=0 if exact else 1e-5)

    def reference_thinker(self):
        thinker = components.Qwen3OmniMoeThinkerForConditionalGeneration(components.make_tiny_thinker_config()).eval()
        thinker.model = self.cases["thinker_prefill"].model.text_model
        thinker.lm_head = self.cases["thinker_prefill"].model.lm_head
        return thinker

    def test_placeholder_injection_and_deepstack_order(self):
        thinker, vision, audio, reference = pipeline.official_raw_prefill(self.cases)
        template = self.cases["thinker_prefill"].export_args
        injected = components.inject_multimodal_features(template, thinker.config, vision, audio)
        image_mask = injected[0] == thinker.config.image_token_id
        audio_mask = injected[0] == thinker.config.audio_token_id
        self.assertEqual(image_mask[0].nonzero().flatten().tolist(), [2, 3, 4, 5])
        self.assertEqual(audio_mask[0].nonzero().flatten().tolist(), [8, 9, 10])
        self.assertTrue(torch.equal(injected[5], image_mask | audio_mask))
        self.assertTrue(torch.equal(injected[6], image_mask))
        torch.testing.assert_close(injected[4][image_mask], vision[0], rtol=0, atol=0)
        torch.testing.assert_close(injected[4][audio_mask], audio[0], rtol=0, atol=0)
        self.assert_outputs_close(injected[7:], vision[1:], exact=True)
        with torch.inference_mode():
            actual = self.cases["thinker_prefill"].model(*injected)
            self.assert_outputs_close(actual, reference)
            corrupted = list(injected)
            corrupted[7] = corrupted[7].flip(0)
            reversed_output = self.cases["thinker_prefill"].model(*corrupted)
        self.assertFalse(torch.allclose(reversed_output[0], reference[0], rtol=1e-4, atol=1e-5))

    def test_old_shifted_masks_and_wrong_feature_counts_are_rejected(self):
        template = self.cases["thinker_prefill"].export_args
        config = components.make_tiny_thinker_config()
        for index in (5, 6):
            bad = list(template)
            bad[index] = bad[index].roll(-1, dims=1)
            with self.subTest(index=index), self.assertRaises(ValueError):
                components.assert_prefill_contract(tuple(bad), config, 1)
        with self.assertRaises(ValueError):
            components.inject_multimodal_features(template, config, (torch.zeros(3, 8), torch.zeros(3, 8)), (torch.zeros(3, 8),))
        bad = (*template[:7], torch.zeros(3, 8))
        with self.assertRaises(ValueError):
            components.assert_prefill_contract(bad, config, 1)

    def test_official_reference_calls_masks_and_mrope_ignoring_host_positions(self):
        thinker = self.reference_thinker()
        template = self.cases["thinker_prefill"].export_args
        corrupted = list(template)
        corrupted[2] = torch.full_like(template[2], 1000)
        corrupted[3] = template[3] + 1000
        with torch.inference_mode(), patch.object(thinker, "get_rope_index", wraps=thinker.get_rope_index) as rope, patch.object(
            thinker, "get_placeholder_mask", wraps=thinker.get_placeholder_mask
        ) as masks:
            expected = components.reference_prefill(thinker, template, 1, (1, 4, 4), 20)
            actual = components.reference_prefill(thinker, tuple(corrupted), 1, (1, 4, 4), 20)
        self.assertEqual(rope.call_count, 2)
        self.assertEqual(masks.call_count, 4)
        self.assert_outputs_close(actual, expected, exact=True)
        self.assertNotIn("get_image_features", thinker.__dict__)
        self.assertNotIn("get_audio_features", thinker.__dict__)

    def test_raw_reference_does_not_reconstruct_corrupted_wrapper_features(self):
        _, vision, audio, expected = pipeline.official_raw_prefill(self.cases)
        changed = dict(self.cases)
        for name in ("vision_encoder", "audio_encoder"):
            case = self.cases[name]
            args = (torch.zeros_like(case.export_args[0]), *case.export_args[1:])
            changed[name] = dataclasses.replace(case, test_vectors=(args,))
        _, vision_again, audio_again, actual = pipeline.official_raw_prefill(changed)
        self.assert_outputs_close((*vision_again, *audio_again, *actual), (*vision, *audio, *expected), exact=True)

    def test_reference_and_wrapper_caches_advance_independently_three_steps(self):
        thinker, vision, audio, reference = pipeline.official_raw_prefill(self.cases)
        prefill = self.cases["thinker_prefill"]
        args = components.inject_multimodal_features(prefill.export_args, thinker.config, vision, audio)
        with torch.inference_mode():
            actual = prefill.model(*args)
            ref_cache, tested_cache = reference[1:], actual[1:]
            delta = torch.tensor([[prefill.interface["rope_deltas"][0]]], dtype=torch.float32)
            for step in range(3):
                self.assertNotEqual(ref_cache[0].data_ptr(), tested_cache[0].data_ptr())
                token = torch.tensor([[13 + step]])
                reference = components.reference_decode(thinker, token, ref_cache)
                length = tested_cache[0].shape[-2]
                actual = self.cases["thinker_decode"].model(
                    token, torch.ones(1, length + 1, dtype=torch.int64),
                    components.decode_position_ids(length, delta, token.device), torch.tensor([length]), *tested_cache,
                )
                self.assert_outputs_close(actual, reference)
                ref_cache, tested_cache = reference[1:], actual[1:]

    def test_fractional_video_delta_is_preserved_in_decode(self):
        config = components.make_tiny_thinker_config()
        config._attn_implementation = "eager"
        config._experts_implementation = "batched_mm"
        thinker = components.Qwen3OmniMoeThinkerForConditionalGeneration(config).eval()
        ids = torch.tensor([[1, 27, 26, 26, 28, 3]])
        with torch.inference_mode():
            output = thinker(
                input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values_videos=torch.randn(8, 96),
                video_grid_thw=torch.tensor([[2, 2, 2]]), video_second_per_grid=torch.tensor([0.1]), use_cache=True,
            )
            self.assertAlmostEqual(float(output.rope_deltas.item()), 0.30000019, places=6)
            position = components.decode_position_ids(6, output.rope_deltas, ids.device)
            self.assertEqual(position.dtype, torch.float32)
            torch.testing.assert_close(position, torch.full((3, 1, 1), 6.3), rtol=0, atol=1e-6)
            cache = components.flatten_thinker_output(output)[1:]
            token = torch.tensor([[13]])
            expected = components.reference_decode(thinker, token, cache)
            actual = components.ThinkerDecodeExportWrapper(thinker.model, thinker.lm_head)(
                token, torch.ones(1, 7, dtype=torch.int64), position, torch.tensor([6]), *cache,
            )
            self.assert_outputs_close(actual, expected)

    def test_low_precision_four_corner_interpolation_matches_official_order(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), torch.inference_mode():
                torch.manual_seed(1234)
                vision = components.Qwen3OmniMoeVisionEncoder(components.make_tiny_vision_config()).eval().to(dtype)
                grid = torch.tensor([[2, 6, 6]])
                pixels = torch.randn(72, 96, dtype=dtype)
                args = components.prepare_vision_inputs(vision, pixels, grid)
                corners = vision.pos_embed(args[1]) * args[2][:, :, None]
                position = vision.fast_pos_embed_interpolate(grid)
                torch.testing.assert_close(corners[0] + corners[1] + corners[2] + corners[3], position, rtol=0, atol=0)
                self.assertFalse(torch.equal(corners.sum(0), position))
                original = vision(pixels, grid)
                actual = components.VisionEncoderExportWrapper(vision)(*args)
                self.assert_outputs_close(actual, (original.pooler_output, *original.deepstack_features))

    def test_small_config_eager_vs_batched_mm_experts_low_precision(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), torch.inference_mode():
                config, model, _ = components._make_tiny_text_modules(1234)
                experts = model.layers[0].mlp.experts.to(dtype)
                states = torch.randn(24, 8, dtype=dtype)
                indices = torch.tensor([[3, 1], [1, 0], [2, 3]] * 8)
                weights = torch.tensor([[0.35, 0.65], [0.55, 0.45], [0.72, 0.28]] * 8)
                config._experts_implementation = "eager"
                reference = experts(states, indices, weights)
                config._experts_implementation = "batched_mm"
                actual = experts(states, indices, weights)
                torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)

    def test_small_config_real_code_path_low_precision_multideepstack(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(scope="small-config-real-code-path", dtype=dtype):
                config = components.make_tiny_thinker_config()
                config.vision_config.depth = 2
                config.vision_config.deepstack_visual_indexes = [0, 1]
                config.text_config.num_hidden_layers = 2
                config.text_config.max_position_embeddings = 64
                config._attn_implementation = "eager"
                config._experts_implementation = "batched_mm"
                torch.manual_seed(1234)
                thinker = components.Qwen3OmniMoeThinkerForConditionalGeneration(config).eval().to(dtype)
                full = SimpleNamespace(thinker=thinker)
                cases = {name: components.build_real_thinking_component(full, name, text_sequence_length=12)
                         for name in components.THINKING_COMPONENTS}
                self.assertEqual(len(cases["thinker_prefill"].export_args[7:]), 2)
                self.assertEqual(len(cases["vision_encoder"].output_names), 3)
                self.assertEqual(cases["thinker_decode"].export_args[2].dtype, torch.float32)
                for case in cases.values():
                    self.assertIsNone(case.checkpoint_fingerprint)
                    self.assertTrue(case.source_equivalence["checked"])
                    self.assertEqual(case.source_equivalence["max_abs_error"], 0)
                _, vision, audio, reference = pipeline.official_raw_prefill(cases, full)
                args = components.inject_multimodal_features(cases["thinker_prefill"].export_args, config, vision, audio)
                with torch.inference_mode():
                    self.assert_outputs_close(cases["thinker_prefill"].model(*args), reference)

    def test_compare_rejects_bad_tolerances_and_shape_without_broadcast(self):
        for value in (float("nan"), float("inf"), -1.0):
            for rtol, atol in ((value, 1e-5), (1e-4, value)):
                with self.subTest(rtol=rtol, atol=atol), self.assertRaises(ValueError):
                    pipeline.compare_outputs((np.zeros(2, dtype=np.float32),), (torch.zeros(2),), ("x",), rtol, atol)
        result = pipeline.compare_outputs((np.zeros((2, 3), dtype=np.float32),), (torch.zeros(4, 5),), ("x",), 1e-4, 1e-5)
        self.assertFalse(result["x"]["passed"])
        self.assertFalse(result["x"]["shape_match"])
        self.assertIsNone(result["x"]["max_abs_error"])
        self.assertFalse(pipeline.compare_outputs((), (torch.zeros(2),), ("x",), 1e-4, 1e-5)["output_count"]["passed"])
        for value in (float("nan"), float("inf")):
            result = pipeline.compare_outputs((np.array([value], dtype=np.float32),), (torch.tensor([value]),), ("x",), 1e-4, 1e-5)
            self.assertFalse(result["x"]["passed"])
            self.assertFalse(result["x"]["finite"])

    def test_bfloat16_cpu_binding_and_compare(self):
        graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "bf16-identity",
                                  [helper.make_tensor_value_info("x", TensorProto.BFLOAT16, [3])],
                                  [helper.make_tensor_value_info("y", TensorProto.BFLOAT16, [3])])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = 10
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = torch.tensor([0.25, -1.5, 3.125], dtype=torch.bfloat16)
        feed = pipeline.cast_feed(session, {"x": tensor_to_numpy(expected)}, "CPUExecutionProvider")
        self.assertEqual(feed["x"].dtype, np.dtype(ml_dtypes.bfloat16))
        actual = run_ort_session(session, feed)
        self.assertTrue(pipeline.compare_outputs(actual, (expected,), ("y",), 0, 0)["y"]["passed"])
        torch.testing.assert_close(pipeline._numpy_tensor(actual[0], torch.device("cpu")), expected, rtol=0, atol=0)


class E2EEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix=".test-semantics-", dir=WORKSPACE)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "validation" / "end_to_end.json"
        self.output.parent.mkdir()
        self.output.write_text(json.dumps({"passed": True, "profile": "tiny-fixed-shape", "attempt_started_ns": 1}))
        self.old_bytes = self.output.read_bytes()
        self.identities = {name: {"model_sha256": name} for name in components.THINKING_COMPONENTS}
        for name in components.THINKING_COMPONENTS:
            root = self.root / "onnx" / name
            root.mkdir(parents=True)
            (root / "model.onnx").write_bytes(b"test-only; execution is mocked")
            (root / "export_metadata.json").write_text(json.dumps({
                "case": name, "product_component": name, "profile": "tiny-fixed-shape", "seed": 1234,
            }))
            (root / "validation.json").write_text("{}")

    def invoke(self, *arguments):
        with patch("sys.argv", ["validate_thinking_pipeline.py", "--package-dir", str(self.root), *arguments]), contextlib.redirect_stdout(io.StringIO()):
            pipeline.main()

    def successful_result(self):
        stages = ("vision", "audio", "prefill", "decode_step_1", "decode_step_2", "decode_step_3")
        return {"decode_steps": 3, "comparisons": {stage: {"x": {"passed": True}} for stage in stages}}

    def test_bad_arguments_preserve_report_and_stop_before_loading(self):
        for arguments in (("--atol", "inf"), ("--rtol", "nan"), ("--atol=-1",), ("--provider", "UnavailableProvider")):
            with self.subTest(arguments=arguments), patch.object(pipeline, "load_component_evidence") as load:
                with self.assertRaises((ValueError, RuntimeError)):
                    self.invoke(*arguments)
                load.assert_not_called()
                self.assertEqual(self.output.read_bytes(), self.old_bytes)
                failure = json.loads((self.output.parent / "end_to_end.failure.json").read_text())
                self.assertIs(failure["passed"], False)

    def test_profile_and_seed_mismatch_rejected_before_identity_loading(self):
        path = self.root / "onnx" / "vision_encoder" / "export_metadata.json"
        for field, value in (("profile", "real-fixed-shape"), ("seed", 999)):
            meta = {"case": "vision_encoder", "product_component": "vision_encoder", "profile": "tiny-fixed-shape", "seed": 1234}
            meta[field] = value
            path.write_text(json.dumps(meta))
            with self.subTest(field=field), patch.object(pipeline, "artifact_identity") as identity:
                with self.assertRaises(ValueError):
                    pipeline.load_component_evidence(self.root, "tiny", 1234)
                identity.assert_not_called()

    def test_component_requires_strict_validation(self):
        with patch.object(pipeline, "artifact_identity", return_value={"case": "vision_encoder"}):
            with self.assertRaisesRegex(RuntimeError, "验证报告"):
                pipeline.load_component_evidence(self.root, "tiny", 1234)

    def test_execution_failure_preserves_old_report_and_records_identity(self):
        with patch.object(pipeline, "load_component_evidence", return_value=(self.identities, {})), patch.object(
            pipeline, "validate_pipeline", side_effect=RuntimeError("injected execution failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                self.invoke()
        self.assertEqual(self.output.read_bytes(), self.old_bytes)
        failure = json.loads((self.output.parent / "end_to_end.failure.json").read_text())
        self.assertEqual(failure["artifact_identities"], self.identities)
        self.assertGreater(failure["attempt_started_ns"], 1)

    def test_final_identity_change_preserves_old_report(self):
        with patch.object(pipeline, "load_component_evidence", return_value=(self.identities, {})), patch.object(
            pipeline, "validate_pipeline", return_value=self.successful_result()
        ), patch.object(pipeline, "verify_artifact_identity", side_effect=RuntimeError("changed during execution")) as verify:
            with self.assertRaisesRegex(RuntimeError, "changed"):
                self.invoke()
            verify.assert_called_once()
        self.assertEqual(self.output.read_bytes(), self.old_bytes)

    def test_success_checks_all_identities_and_writes_bound_report(self):
        with patch.object(pipeline, "load_component_evidence", return_value=(self.identities, {})), patch.object(
            pipeline, "validate_pipeline", return_value=self.successful_result()
        ), patch.object(pipeline, "verify_artifact_identity") as verify:
            self.invoke()
        self.assertEqual(verify.call_count, 4)
        report = json.loads(self.output.read_text())
        self.assertIs(report["passed"], True)
        self.assertEqual(report["artifact_identities"], self.identities)
        self.assertEqual(report["reference_scope"], "official_top_level_with_raw_synthetic_features")
        for name, identity in self.identities.items():
            self.assertEqual(report["models"][name]["artifact_identity"], identity)

    def test_empty_comparisons_cannot_pass(self):
        result = self.successful_result()
        result["comparisons"]["prefill"] = {}
        with patch.object(pipeline, "load_component_evidence", return_value=(self.identities, {})), patch.object(
            pipeline, "validate_pipeline", return_value=result
        ), patch.object(pipeline, "verify_artifact_identity"):
            with self.assertRaises(SystemExit):
                self.invoke()
        self.assertIs(json.loads(self.output.read_text())["passed"], False)


if __name__ == "__main__":
    unittest.main()
