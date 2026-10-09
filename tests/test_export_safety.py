from __future__ import annotations

import argparse
import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import export_onnx as basic
import export_thinking_onnx as thinking
from onnx_artifact_utils import WORKSPACE


class ExportSafetyTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix=".test-transactions-", dir=WORKSPACE)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.product = self.root / "product"
        self.targets = {name: self.product / "onnx" / name for name in ("a", "b")}
        for path in self.targets.values():
            path.mkdir(parents=True); (path / "model.onnx").write_text("OLD")
        self.manifest = self.product / "manifest.json"
        self.manifest.write_text("OLD REPORT")

    def generate(self, name, staging):
        (staging / "model.onnx").write_text("NEW " + name)

    def run_export(self, export=None, **kwargs):
        basic.export_transaction(self.targets, True, export or self.generate, evidence=(self.manifest,),
                                 lock_root=self.product, **kwargs)

    def assert_old(self):
        for path in self.targets.values():
            self.assertEqual((path / "model.onnx").read_text(), "OLD")
        self.assertEqual(self.manifest.read_text(), "OLD REPORT")

    def test_stage_failure_preserves_everything(self):
        def fail(name, path):
            self.generate(name, path)
            if name == "b":
                raise RuntimeError("stage failure")
        with self.assertRaisesRegex(RuntimeError, "stage failure"):
            self.run_export(fail)
        self.assert_old()
        self.assertFalse(list(self.root.glob(".product.transaction-*")))

    def test_commit_rename_failure_rolls_back(self):
        rename = Path.rename
        def fail(path, target):
            if path.name == "b" and path.parent.name == "staged":
                raise OSError("commit failure")
            return rename(path, target)
        with patch.object(Path, "rename", fail), self.assertRaisesRegex(OSError, "commit failure"):
            self.run_export()
        self.assert_old()

    def test_success_invalidates_global_evidence(self):
        self.run_export()
        self.assertFalse(self.manifest.exists())
        self.assertEqual((self.targets["a"] / "model.onnx").read_text(), "NEW a")
        self.assertFalse(list(self.root.glob(".product.transaction-*")))
        self.assertTrue((self.root / ".product.export.lock").exists())

    def test_cleanup_failure_is_not_reported_as_uncommitted_export(self):
        with patch.object(basic.shutil, "rmtree", side_effect=PermissionError("cleanup")), self.assertWarnsRegex(RuntimeWarning, "清理失败"):
            self.run_export()
        self.assertEqual((self.targets["a"] / "model.onnx").read_text(), "NEW a")
        remnants = list(self.root.glob(".product.transaction-*"))
        self.assertEqual(len(remnants), 1)
        with self.assertRaisesRegex(RuntimeError, "未完成事务"):
            self.run_export()

    def test_concurrent_lock_rejected(self):
        with basic.output_lock(self.product):
            with self.assertRaisesRegex(RuntimeError, "另一导出进程"):
                self.run_export()
        self.assert_old()

    def test_preflight_failure_does_not_delete_first_component(self):
        target = self.targets["b"]
        target.rename(self.root / "b-backup")
        target.symlink_to(self.root / "b-backup", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "符号链接"):
            self.run_export()
        self.assertEqual((self.targets["a"] / "model.onnx").read_text(), "OLD")

    def test_authorization_checked_again_inside_lock(self):
        calls = []
        def guard():
            calls.append("check")
            if len(calls) == 2:
                raise RuntimeError("new real product detected under lock")
        with self.assertRaisesRegex(RuntimeError, "under lock"):
            self.run_export(authorize=guard)
        self.assertEqual(len(calls), 2)
        self.assert_old()

    def test_authorization_checked_before_commit(self):
        def guard():
            if self.manifest.read_text() == "REAL":
                raise RuntimeError("real appeared")
        def mutate(name, path):
            self.generate(name, path)
            if name == "b":
                self.manifest.write_text("REAL")
        with self.assertRaisesRegex(RuntimeError, "real appeared"):
            self.run_export(mutate, authorize=guard)
        self.assertEqual((self.targets["a"] / "model.onnx").read_text(), "OLD")
        self.assertEqual(self.manifest.read_text(), "REAL")

    def test_tiny_guard_runs_under_real_export_transaction(self):
        self.manifest.write_text(json.dumps({"official_weights_included": True}))
        with self.assertRaisesRegex(RuntimeError, "拒绝覆盖"):
            self.run_export(authorize=lambda: thinking.authorize_tiny_export(self.product, False))
        self.assertEqual((self.targets["a"] / "model.onnx").read_text(), "OLD")

    def test_valid_json_unknown_identity_cannot_be_overwritten(self):
        self.manifest.unlink()
        folder = self.product / "onnx/vision_encoder"; folder.mkdir()
        (folder / "model.onnx").write_text("KEEP")
        for payload in ({}, {"profile": "other"}, {"profile": "tiny-fixed-shape", "official_weights_included": True}):
            (folder / "export_metadata.json").write_text(json.dumps(payload))
            with self.subTest(payload=payload), self.assertRaisesRegex(RuntimeError, "拒绝覆盖"):
                thinking.authorize_tiny_export(self.product, False)
            self.assertEqual((folder / "model.onnx").read_text(), "KEEP")

    def test_basic_cli_dispatches_transaction_and_snapshot_metadata(self):
        args = argparse.Namespace(case="rmsnorm", output_dir=self.root / "out", force=True, opset=18, seed=1234)
        with patch.object(basic, "parse_args", return_value=args), patch.object(basic, "prepare_output_dir", return_value=args.output_dir), patch.object(
            basic, "export_transaction"
        ) as tx:
            basic.main()
        self.assertEqual(tx.call_args.args[0], {"rmsnorm": args.output_dir})

    def test_fake_real_case_cannot_be_exported(self):
        from types import SimpleNamespace
        with self.assertRaisesRegex(RuntimeError, "完整 checkpoint"):
            thinking.export_component(SimpleNamespace(loading_verified=False, checkpoint_fingerprint=None), self.root, 18, 1234, "real-fixed-shape", {})


if __name__ == "__main__":
    unittest.main()
