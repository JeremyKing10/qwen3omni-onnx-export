"""Package identity, relocation and strict acceptance regression tests."""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import aggregate_operators as aggregate
import build_thinking_package as package
import inspect_onnx
import onnx_artifact_utils as utils
import validate_onnx


class PackageEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix=".test-package-evidence-", dir=utils.WORKSPACE)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.product = self.root / "product"
        self.network = patch.object(package, "hf_hub_download", side_effect=AssertionError("network disabled"))
        self.download = self.network.start()
        self.addCleanup(self.network.stop)

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def invoke(self, module, *args):
        with patch("sys.argv", [module.__name__ + ".py", *map(str, args)]), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            module.main()

    def create_component(self, component="vision_encoder", with_function=False):
        root = self.product / "onnx" / component
        root.mkdir(parents=True)
        model_path = root / "model.onnx"
        weights = np.array([1.0, 2.0], dtype=np.float32)
        graph = helper.make_graph(
            [helper.make_node("Add", ["x", "weights"], ["y"])], "package-evidence-fixture",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
            [numpy_helper.from_array(weights, "weights")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = 10
        if with_function:
            model.functions.append(helper.make_function(
                "ai.onnx", "UnusedFixtureIdentity", ["A"], ["B"],
                [helper.make_node("Identity", ["A"], ["B"])], [helper.make_opsetid("", 18)],
            ))
        onnx.save_model(model, str(model_path), save_as_external_data=True,
                        all_tensors_to_one_file=True, location="weights.bin", size_threshold=0)
        vectors = []
        for index in range(2):
            x = np.array([index, index + 4], dtype=np.float32)
            inputs, references = root / f"inputs_{index}.npz", root / f"references_{index}.npz"
            input_dtypes = utils.save_tensor_archive(inputs, {"x": x})
            output_dtypes = utils.save_tensor_archive(references, {"y": x + weights})
            vectors.append({
                "index": index, "input_file": inputs.name, "reference_output_file": references.name,
                "input_sha256": utils.file_sha256(inputs), "reference_output_sha256": utils.file_sha256(references),
                "input_dtypes": input_dtypes, "reference_output_dtypes": output_dtypes,
                "routing": {"sha256": f"fixture-pattern-{index}"},
            })
        metadata = {
            "evidence_schema_version": 2, "case": component, "product_component": component,
            "profile": "tiny-fixed-shape", "official_weights_included": False, "randomly_initialized": True,
            "description": "Independent NumPy Add reference; evidence packaging fixture, not Qwen semantics",
            "model_file": "model.onnx", "model_sha256": utils.file_sha256(model_path),
            "model_identity": utils.model_identity(model_path), "test_vectors": vectors,
            "input_names": ["x"], "output_names": ["y"], "interface": {"fixture": True},
            "source_snapshot": utils.source_snapshot(),
            "source_equivalence": {"checked": True, "max_abs_error": 0.0, "scope": "official_top_level_with_precomputed_features"},
        }
        self.write_json(root / "export_metadata.json", metadata)
        self.invoke(validate_onnx, "--model", model_path)
        self.inspect(component)
        return root

    def inspect(self, component="vision_encoder"):
        self.invoke(inspect_onnx, "--model", self.product / "onnx" / component / "model.onnx",
                    "--output", self.product / "operators" / f"{component.replace('_encoder', '')}.json",
                    "--fail-on-custom-domain")

    def create_sources(self):
        root = self.root / "source"
        root.mkdir()
        for filename in (*package.ROOT_CONFIG_FILES, *package.PROCESSOR_FILES):
            (root / filename).write_text("{}", encoding="utf-8")
        return root

    def args(self, source=None):
        return argparse.Namespace(package_dir=self.product, repo_id=package.DEFAULT_REPO,
                                  revision=package.DEFAULT_REVISION, source_dir=source, offline=True)

    def create_complete_product(self, with_function=False):
        for name in package.THINKING_COMPONENTS:
            self.create_component(name, with_function)
        self.invoke(aggregate, "--package-dir", self.product)
        components = {name: package.collect_component(self.product, name) for name in package.THINKING_COMPONENTS}
        stages = {"vision": "vision_encoder", "audio": "audio_encoder", "prefill": "thinker_prefill",
                  **{f"decode_step_{index}": "thinker_decode" for index in range(1, 4)}}
        comparisons = {}
        for stage, name in stages.items():
            report = json.loads((self.product / "onnx" / name / "validation.json").read_text())
            stats = report["test_vectors"][0]["comparisons"]["y"]
            comparisons[stage] = {"y": {
                "passed": stats["allclose"], "shape_match": stats["shape_match"], "dtype_match": stats["dtype_match"],
                "finite": stats["finite"], "max_abs_error": stats["max_abs_error"],
            }}
        end_to_end = {
            "evidence_schema_version": 2, "attempt_started_ns": 100, "passed": True,
            "profile": "tiny-fixed-shape", "decode_steps": 3,
            "artifact_identities": {name: entry["artifact_identity"] for name, entry in components.items()},
            "models": {name: {"sha256": entry["model_sha256"]} for name, entry in components.items()},
            "tolerances": {"rtol": 1e-4, "atol": 1e-5}, "comparisons": comparisons,
            "reference_scope": "official_top_level_with_raw_synthetic_features",
            "reference_positions": "official_forward_independent_mrope_and_cache",
        }
        self.write_json(self.product / "validation" / "end_to_end.json", end_to_end)
        return self.create_sources(), components, end_to_end

    def test_collect_uses_real_graph_identity_and_does_not_copy(self):
        root = self.create_component()
        (root / "model.onnx.data.stale").write_bytes(b"unused")
        entry = package.collect_component(self.product, "vision_encoder")
        self.assertEqual([item["location"] for item in entry["external_data"]], ["weights.bin"])
        self.assertFalse((self.product / "test_data").exists())
        self.assertEqual(entry["artifact_identity"], utils.artifact_identity(root / "model.onnx"))

    def test_changed_weight_and_refreshed_inspection_cannot_reuse_old_validation(self):
        root = self.create_component()
        (root / "weights.bin").write_bytes(np.array([99.0, 2.0], dtype=np.float32).tobytes())
        metadata_path = root / "export_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["model_identity"] = utils.model_identity(root / "model.onnx")
        self.write_json(metadata_path, metadata)
        self.inspect()
        with self.assertRaises(RuntimeError):
            package.collect_component(self.product, "vision_encoder")

    def test_empty_external_report_cannot_hide_current_weights(self):
        self.create_component()
        path = self.product / "operators" / "vision.json"
        report = json.loads(path.read_text())
        report["external_data"]["files"] = []
        self.write_json(path, report)
        with self.assertRaises(RuntimeError):
            package.collect_component(self.product, "vision_encoder")

    def test_changed_npz_rejected_before_copy(self):
        root = self.create_component()
        with (root / "inputs_0.npz").open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaises(RuntimeError):
            package.collect_component(self.product, "vision_encoder")
        self.assertFalse((self.product / "test_data").exists())

    def test_component_identity_and_interface_rejected(self):
        root = self.create_component()
        metadata_path = root / "export_metadata.json"
        original = json.loads(metadata_path.read_text())
        for key, value in (("product_component", "audio_encoder"), ("input_names", ["wrong"]), ("profile", "legacy")):
            with self.subTest(field=key):
                changed = copy.deepcopy(original)
                changed[key] = value
                self.write_json(metadata_path, changed)
                with self.assertRaises(RuntimeError):
                    package.collect_component(self.product, "vision_encoder")
        self.write_json(metadata_path, original)

    def test_reports_remain_valid_after_package_move(self):
        root = self.create_component()
        expected = utils.artifact_identity(root / "model.onnx")
        moved = self.root / "moved"
        shutil.move(str(self.product), moved)
        entry = package.collect_component(moved, "vision_encoder")
        self.assertEqual(entry["artifact_identity"], expected)

    def test_copy_rejects_live_and_broken_symlink_leaves(self):
        source, target = self.root / "source.txt", self.root / "target.txt"
        source.write_text("source")
        target.write_text("keep")
        for name, referent in (("live", target), ("broken", self.root / "absent")):
            destination = self.root / name
            destination.symlink_to(referent)
            with self.subTest(name=name), self.assertRaises(ValueError):
                package.copy_checked(source, destination, self.root)
        linked_source = self.root / "linked-source"
        linked_source.symlink_to(source)
        with self.assertRaises(ValueError):
            package.copy_checked(linked_source, self.root / "result.txt", self.root)
        self.assertEqual(target.read_text(), "keep")

    def test_local_sources_are_complete_and_never_mixed_with_network(self):
        root = self.create_sources()
        self.assertEqual(len(package.resolve_package_sources(source_dir=root, offline=True)), 7)
        (root / "vocab.json").unlink()
        with self.assertRaises(FileNotFoundError):
            package.resolve_package_sources(source_dir=root, offline=True)
        self.download.assert_not_called()

    def test_offline_hub_resolution_requests_only_seven_nonweight_files(self):
        root = self.create_sources()
        self.download.side_effect = lambda **kwargs: str(root / kwargs["filename"])
        resolved = package.resolve_package_sources(offline=True)
        self.assertEqual(set(resolved), set((*package.ROOT_CONFIG_FILES, *package.PROCESSOR_FILES)))
        self.assertEqual(self.download.call_count, 7)
        self.assertTrue(all(call.kwargs["local_files_only"] for call in self.download.call_args_list))
        self.assertFalse(self.product.exists())

    def test_aggregate_includes_function_nodes_and_binds_csv(self):
        self.create_complete_product(with_function=True)
        summary = json.loads((self.product / "operators" / "summary.json").read_text())
        self.assertEqual(summary["total_node_count"], 8)
        self.assertEqual(sum(item["count"] for item in summary["unique_operators"]), 8)
        self.assertEqual(summary["csv_sha256"], utils.file_sha256(self.product / summary["csv"]))

    def test_aggregate_failure_invalidates_previous_summary(self):
        self.create_complete_product()
        path = self.product / "operators" / "audio.json"
        report = json.loads(path.read_text())
        report["operator_counts"][0]["count"] += 1
        self.write_json(path, report)
        with self.assertRaises(RuntimeError):
            self.invoke(aggregate, "--package-dir", self.product)
        summary = json.loads((self.product / "operators" / "summary.json").read_text())
        self.assertFalse(summary["passed"])
        self.assertFalse((self.product / "operators" / "all_operators.csv").exists())

    def test_complete_package_relocates_without_old_source_or_network(self):
        source, _, _ = self.create_complete_product()
        manifest = package.build_package(self.args(source), self.product, {})
        self.assertTrue(manifest["passed"])
        self.assertIn("onnx_artifact_utils.py", package.TOOL_FILES)
        self.assertTrue((self.product / "tools" / "onnx_artifact_utils.py").is_file())
        moved = self.root / "moved"
        shutil.move(str(self.product), moved)
        self.product = moved
        shutil.rmtree(source)
        rebuilt = package.build_package(self.args(), moved, manifest)
        self.assertTrue(rebuilt["passed"])
        self.assertEqual(manifest["artifact_identities"], rebuilt["artifact_identities"])
        self.download.assert_not_called()
        with self.assertRaises(FileNotFoundError):
            package.build_package(self.args(source), moved, manifest)

    def test_missing_summary_or_changed_csv_never_passes(self):
        source, _, _ = self.create_complete_product()
        summary = self.product / "operators" / "summary.json"
        saved = summary.read_bytes()
        summary.unlink()
        result = package.build_package(self.args(source), self.product, {})
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "unverified-tiny-artifacts")
        summary.write_bytes(saved)
        (self.product / "operators" / "all_operators.csv").write_text("changed")
        self.assertFalse(package.build_package(self.args(source), self.product, {})["passed"])

    def test_e2e_rejects_incomplete_nonfinite_and_old_scope_evidence(self):
        _, components, report = self.create_complete_product()
        self.assertTrue(package.end_to_end_matches(report, components))
        mutations = (
            lambda value: value["artifact_identities"].pop("audio_encoder"),
            lambda value: value["tolerances"].update(rtol=float("nan")),
            lambda value: value["tolerances"].update(atol=1.0),
            lambda value: value.update(decode_steps=2),
            lambda value: value.update(reference_scope="wrapper_self_comparison"),
            lambda value: value["comparisons"].update(decode_step_3={}),
            lambda value: value["comparisons"]["prefill"]["y"].update(max_abs_error=float("inf")),
        )
        for index, mutate in enumerate(mutations):
            bad = copy.deepcopy(report)
            mutate(bad)
            with self.subTest(index=index):
                self.assertFalse(package.end_to_end_matches(bad, components))

    def test_newer_e2e_failure_blocks_old_success_only_for_same_identity(self):
        source, _, report = self.create_complete_product()
        path = self.product / "validation" / "end_to_end.failure.json"
        failure = {"passed": False, "attempt_started_ns": 101, "profile": report["profile"],
                   "artifact_identities": report["artifact_identities"]}
        self.write_json(path, failure)
        self.assertFalse(package.build_package(self.args(source), self.product, {})["passed"])
        failure["artifact_identities"] = {}
        self.write_json(path, failure)
        self.assertTrue(package.build_package(self.args(source), self.product, {})["passed"])

    def test_main_failure_replaces_stale_success_manifest(self):
        self.product.mkdir()
        path = self.product / "manifest.json"
        self.write_json(path, {"passed": True, "status": "official-weight-components-validated"})
        with self.assertRaises(FileNotFoundError):
            self.invoke(package, "--package-dir", self.product, "--offline")
        failure = json.loads(path.read_text())
        self.assertFalse(failure["passed"])
        self.assertEqual(failure["status"], "unverified-artifacts")

    def test_snapshot_difference_does_not_mislabel_packaging_as_export(self):
        original = utils.source_snapshot()
        current = copy.deepcopy(original)
        current["files"]["build_thinking_package.py"] = "different-packaging"
        check = package.snapshot_comparison(original, current)
        self.assertTrue(check["evidence_sources_match"])
        self.assertFalse(check["all_source_hashes_match"])
        current["files"]["qwen3_omni_thinking_components.py"] = "different-model"
        self.assertFalse(package.snapshot_comparison(original, current)["evidence_sources_match"])


if __name__ == "__main__":
    unittest.main()
