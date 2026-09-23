from __future__ import annotations

import contextlib
import copy
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ml_dtypes
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

import onnx_artifact_utils as utils
import validate_onnx


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix=".test-evidence-", dir=utils.WORKSPACE)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def create_model(self):
        path = self.root / "model.onnx"
        graph = helper.make_graph(
            [helper.make_node("Add", ["x", "w"], ["y"])], "independent-numpy-reference",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
            [numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32), "w")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = 10
        onnx.save_model(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
                        location="model.onnx.data", size_threshold=0)
        vectors = []
        for index in range(2):
            x = np.array([index, index + 3], dtype=np.float32)
            ip, rp = self.root / f"inputs_{index}.npz", self.root / f"reference_{index}.npz"
            ins = utils.save_tensor_archive(ip, {"x": x})
            refs = utils.save_tensor_archive(rp, {"y": x + np.array([1.0, 2.0], dtype=np.float32)})
            vectors.append({"index": index, "input_file": ip.name, "reference_output_file": rp.name,
                            "input_sha256": utils.file_sha256(ip), "reference_output_sha256": utils.file_sha256(rp),
                            "input_dtypes": ins, "reference_output_dtypes": refs})
        metadata = {"evidence_schema_version": 2, "case": "rmsnorm", "model_sha256": utils.file_sha256(path),
                    "model_identity": utils.model_identity(path), "test_vectors": vectors,
                    "input_names": ["x"], "output_names": ["y"]}
        (self.root / "export_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return path

    def run_validate(self, path, *arguments):
        with patch("sys.argv", ["validate_onnx.py", "--model", str(path), *arguments]), contextlib.redirect_stdout(io.StringIO()):
            validate_onnx.main()

    def test_independent_numpy_reference_validates(self):
        path = self.create_model()
        self.run_validate(path)
        report = json.loads((self.root / "validation.json").read_text())
        utils.require_strict_validation(report, utils.artifact_identity(path))

    def test_bad_provider_and_case_preserve_report(self):
        path = self.create_model()
        self.run_validate(path)
        before = (self.root / "validation.json").read_bytes()
        for args in (("--provider", "DefinitelyUnavailableProvider"), ("--case", "moe_block"), ("--atol", "inf")):
            with self.subTest(args=args), self.assertRaises((RuntimeError, ValueError)):
                self.run_validate(path, *args)
            self.assertEqual(before, (self.root / "validation.json").read_bytes())

    def test_diagnostic_does_not_replace_strict_report(self):
        path = self.create_model()
        self.run_validate(path)
        before = (self.root / "validation.json").read_bytes()
        self.run_validate(path, "--skip-shape-inference")
        self.assertEqual(before, (self.root / "validation.json").read_bytes())
        report = json.loads((self.root / "validation.diagnostic.json").read_text())
        with self.assertRaises(RuntimeError):
            utils.require_strict_validation(report, utils.artifact_identity(path))

    def test_changed_weights_rejected_before_execution(self):
        path = self.create_model()
        identity = utils.artifact_identity(path)
        data = self.root / "model.onnx.data"
        data.write_bytes(np.array([99.0, 2.0], dtype=np.float32).tobytes())
        with self.assertRaises(RuntimeError):
            utils.verify_artifact_identity(path, identity)

    def test_refreshed_export_metadata_does_not_validate_old_report(self):
        path = self.create_model()
        self.run_validate(path)
        old_report = json.loads((self.root / "validation.json").read_text())
        (self.root / "model.onnx.data").write_bytes(np.array([99.0, 2.0], dtype=np.float32).tobytes())
        mp = self.root / "export_metadata.json"
        meta = json.loads(mp.read_text())
        meta["model_identity"] = utils.model_identity(path)
        mp.write_text(json.dumps(meta))
        with self.assertRaises(RuntimeError):
            utils.require_strict_validation(old_report, utils.artifact_identity(path))

    def test_metadata_change_invalidates_identity(self):
        path = self.create_model()
        old = utils.artifact_identity(path)
        mp = self.root / "export_metadata.json"
        meta = json.loads(mp.read_text())
        meta["case"] = "audio_encoder"
        mp.write_text(json.dumps(meta))
        with self.assertRaises(RuntimeError):
            utils.verify_artifact_identity(path, old)

    def test_empty_vectors_rejected(self):
        path = self.create_model()
        mp = self.root / "export_metadata.json"
        meta = json.loads(mp.read_text()); meta["test_vectors"] = []
        mp.write_text(json.dumps(meta))
        with self.assertRaises(ValueError):
            utils.artifact_identity(path)

    def test_identity_survives_package_move(self):
        path = self.create_model()
        old = utils.artifact_identity(path)
        moved = self.root / "moved"
        moved.mkdir()
        for source in list(self.root.iterdir()):
            if source.is_file():
                shutil.copy2(source, moved / source.name)
        self.assertEqual(old, utils.artifact_identity(moved / path.name))

    def test_symlink_and_parent_traversal_rejected(self):
        target = self.root / "target"; target.mkdir()
        link = self.root / "link"; link.symlink_to(target, target_is_directory=True)
        broken = self.root / "broken"; broken.symlink_to(self.root / "missing")
        for path in (link / "x", broken, self.root / ".." / self.root.name / "x"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                utils.safe_path(self.root, path)

    def test_invalid_tolerances(self):
        for value in (float("inf"), float("nan"), -1.0, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                utils.validate_tolerances(value, 1e-5)

    def test_bfloat16_archive_roundtrip(self):
        expected = np.array([1.0, -2.5, 0.0], dtype=ml_dtypes.bfloat16)
        path = self.root / "bf16.npz"
        desc = utils.save_tensor_archive(path, {"x": expected})
        result = utils.load_tensor_archive(path, desc)["x"]
        self.assertEqual(result.dtype, expected.dtype)
        np.testing.assert_array_equal(result.view(np.uint16), expected.view(np.uint16))

    def test_bfloat16_scalar_preserves_rank(self):
        expected = np.array(1.5, dtype=ml_dtypes.bfloat16)
        path = self.root / "scalar.npz"
        desc = utils.save_tensor_archive(path, {"x": expected})
        result = utils.load_tensor_archive(path, desc)["x"]
        self.assertEqual(result.shape, ())
        graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "scalar",
                                  [helper.make_tensor_value_info("x", TensorProto.BFLOAT16, [])],
                                  [helper.make_tensor_value_info("y", TensorProto.BFLOAT16, [])])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]); model.ir_version = 10
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        actual, = utils.run_ort_session(session, {"x": expected})
        self.assertEqual(actual.shape, ())
        self.assertEqual(float(actual), 1.5)

    def test_nonfinite_numeric_comparison_is_serializable_failure(self):
        stats = validate_onnx.finite_and_error(np.array([np.nan]), np.array([1.0]))
        self.assertFalse(stats["finite"])
        json.dumps(stats, allow_nan=False)

    def test_legacy_void_archive_rejected(self):
        path = self.root / "void.npz"
        np.savez(path, x=np.array([b"xx"], dtype="V2"))
        with self.assertRaises(TypeError):
            utils.load_tensor_archive(path)

    def test_bfloat16_cpu_io_binding_input_and_output(self):
        graph = helper.make_graph([
            helper.make_node("Cast", ["x"], ["y"], to=TensorProto.FLOAT),
            helper.make_node("Identity", ["x"], ["z"]),
        ], "bf16-data-path", [helper.make_tensor_value_info("x", TensorProto.BFLOAT16, [2])], [
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("z", TensorProto.BFLOAT16, [2]),
        ])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]); model.ir_version = 10
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = np.array([1.0, 2.0], dtype=ml_dtypes.bfloat16)
        y, z = utils.run_ort_session(session, {"x": expected})
        np.testing.assert_array_equal(y, [1.0, 2.0])
        self.assertEqual(z.dtype, expected.dtype)
        np.testing.assert_array_equal(z, expected)
        with self.assertRaises(TypeError):
            utils.run_ort_session(session, {"x": expected.astype(np.float32)})

    def test_bfloat16_empty_io_and_archive(self):
        expected = np.empty((0, 2), dtype=ml_dtypes.bfloat16)
        path = self.root / "empty.npz"
        desc = utils.save_tensor_archive(path, {"x": expected})
        self.assertEqual(utils.load_tensor_archive(path, desc)["x"].shape, (0, 2))
        graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "empty",
                                  [helper.make_tensor_value_info("x", TensorProto.BFLOAT16, [0, 2])],
                                  [helper.make_tensor_value_info("y", TensorProto.BFLOAT16, [0, 2])])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]); model.ir_version = 10
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        actual, = utils.run_ort_session(session, {"x": expected})
        self.assertEqual(actual.shape, (0, 2))
        self.assertEqual(actual.dtype, expected.dtype)

    def test_ancestor_symlink_rejected_by_identity(self):
        path = self.create_model()
        target = self.root / "real"; target.mkdir()
        sub = target / "nested"; sub.mkdir()
        link = self.root / "alias"; link.symlink_to(target, target_is_directory=True)
        with self.assertRaises(ValueError):
            utils.model_identity(link / "nested" / path.name)

    def test_report_filename_must_not_overwrite_weights(self):
        path = self.create_model()
        for filename, args in (("validation.diagnostic.json", ("--atol", "1")), ("validation.failure.json", ())):
            with self.subTest(filename=filename):
                model = onnx.load(str(path), load_external_data=False)
                for item in model.graph.initializer[0].external_data:
                    if item.key == "location":
                        item.value = filename
                weights = self.root / filename
                weights.write_bytes(np.array([1.0, 2.0], dtype=np.float32).tobytes())
                path.write_bytes(model.SerializeToString())
                mp = self.root / "export_metadata.json"; meta = json.loads(mp.read_text())
                meta["model_sha256"] = utils.file_sha256(path); meta["model_identity"] = utils.model_identity(path)
                mp.write_text(json.dumps(meta))
                before = weights.read_bytes()
                with self.assertRaises(ValueError):
                    self.run_validate(path, *args)
                self.assertEqual(before, weights.read_bytes())

    def test_wide_tolerance_stays_diagnostic(self):
        path = self.create_model(); self.run_validate(path)
        before = (self.root / "validation.json").read_bytes()
        self.run_validate(path, "--atol", "1")
        report = json.loads((self.root / "validation.diagnostic.json").read_text())
        self.assertEqual(before, (self.root / "validation.json").read_bytes())
        with self.assertRaises(RuntimeError):
            utils.require_strict_validation(report, utils.artifact_identity(path))

    def test_input_and_reference_tampering_rejected(self):
        path = self.create_model()
        for filename in ("inputs_0.npz", "reference_0.npz"):
            with self.subTest(filename=filename):
                vector = self.root / filename; before = vector.read_bytes()
                vector.write_bytes(before + b"mutation")
                with self.assertRaises(RuntimeError):
                    utils.artifact_identity(path)
                vector.write_bytes(before)

    def test_missing_and_malformed_external_report_is_honest(self):
        path = self.create_model(); model = onnx.load(str(path), load_external_data=False)
        for item in model.graph.initializer[0].external_data:
            if item.key == "offset":
                item.value = "not-an-int"
        files, errors = utils.external_data_files(model, path)
        self.assertTrue(errors); self.assertFalse(files[0]["all_ranges_valid"])
        self.assertFalse(files[0]["initializers"][0]["range_valid"])
        (self.root / "model.onnx.data").unlink()
        files, errors = utils.external_data_files(model, path)
        self.assertTrue(errors); self.assertEqual(len(files), 1)
        self.assertFalse(files[0]["exists"])

    def test_shared_external_hashed_once(self):
        path = self.create_model()
        model = onnx.load(str(path), load_external_data=False)
        for name in ("w2", "w3"):
            tensor = copy.deepcopy(model.graph.initializer[0]); tensor.name = name
            model.graph.initializer.append(tensor)
        with patch.object(utils, "file_sha256", wraps=utils.file_sha256) as digest:
            files, errors = utils.external_data_files(model, path)
        self.assertEqual(errors, [])
        self.assertEqual(len(files), 1)
        self.assertEqual(digest.call_count, 1)

    def test_offset_without_length_and_short_segment_rejected(self):
        path = self.create_model()
        for offset, length in ((999, None), (0, 1), (-1, None)):
            model = onnx.load(str(path), load_external_data=False)
            tensor = model.graph.initializer[0]; tensor.ClearField("external_data")
            values = {"location": "model.onnx.data", "offset": str(offset)}
            if length is not None:
                values["length"] = str(length)
            for key, value in values.items():
                entry = tensor.external_data.add(); entry.key = key; entry.value = value
            _, errors = utils.external_data_files(model, path)
            self.assertTrue(errors)

    def test_subgraph_external_is_scanned(self):
        path = self.create_model()
        model = onnx.load(str(path), load_external_data=False)
        tensor = copy.deepcopy(model.graph.initializer[0])
        sub = helper.make_graph([], "sub", [], [], [tensor])
        model.graph.ClearField("initializer")
        model.graph.node.append(helper.make_node("If", ["cond"], ["out"], then_branch=sub, else_branch=sub))
        files, errors = utils.external_data_files(model, path)
        self.assertEqual(errors, [])
        self.assertEqual(len(files), 1)
        self.assertEqual(len(files[0]["initializers"]), 2)


if __name__ == "__main__":
    unittest.main()
