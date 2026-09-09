import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from gleep_repro.artifacts import _safe_path, audit_artifacts


class ArtifactTests(unittest.TestCase):
    def test_safe_path_rejects_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                _safe_path(Path(temporary), "../outside.npy")


    def test_artifact_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "artifacts" / "cache" / "tiny.bin"
            payload.parent.mkdir(parents=True)
            payload.write_bytes(b"gleep")
            index = {
                "schema_version": 1,
                "entries": [{
                    "path": "artifacts/cache/tiny.bin",
                    "profiles": ["exp2"],
                    "bytes": 5,
                    "sha256": hashlib.sha256(b"gleep").hexdigest(),
                }],
            }
            (root / "artifacts" / "index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )
            report = audit_artifacts(root)
            self.assertTrue(report["complete"])
            self.assertEqual(report["verified"], 1)
            self.assertEqual(report["index"], "artifacts/index.json")


if __name__ == "__main__":
    unittest.main()
