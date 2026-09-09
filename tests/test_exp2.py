import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gleep_repro.exp2 import (
    _cached_task_arrays,
    parse_tasks,
    score_exp2_cached_task,
    selected_classes,
)


class Exp2TaskTests(unittest.TestCase):
    def test_default_task_count_is_99(self):
        tasks = parse_tasks(None)
        self.assertEqual((tasks[0], tasks[-1], len(tasks)), (2, 100, 99))

    def test_task_specification(self):
        self.assertEqual(parse_tasks("2,10,98:100"), [2, 10, 98, 99, 100])

    def test_published_class_selection_reproduces_discarded_first_draw(self):
        self.assertEqual(selected_classes(2, seed=42, semantics="published"), [53, 64])
        self.assertEqual(selected_classes(2, seed=42, semantics="corrected"), [42, 96])

    def _write_test_cache(self, temporary: str, labels, clustering, prediction) -> Path:
        root = Path(temporary) / "published" / "ResNet18" / "CIFAR10"
        root.mkdir(parents=True)
        arrays = {}
        for name, value in {
            "labels": labels,
            "clustering_logits": clustering,
            "prediction_logits": prediction,
        }.items():
            path = root / f"{name}.npy"
            np.save(path, value, allow_pickle=False)
            arrays[name] = {"path": path.name}
        (root / "manifest.json").write_text(
            json.dumps({
                "model": "ResNet18",
                "source": "CIFAR10",
                "semantics": "published",
                "sample_count": len(labels),
                "source_checkpoint": "test.pth",
                "arrays": arrays,
            }),
            encoding="utf-8",
        )
        return Path(temporary)

    def test_cached_task_reconstructs_filtered_rows_and_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            labels = np.asarray([53, 1, 64, 53, 99, 64], dtype=np.int64)
            clustering = np.arange(18, dtype=np.float32).reshape(6, 3)
            prediction = clustering + 100
            cache_dir = self._write_test_cache(temporary, labels, clustering, prediction)
            actual_clustering, actual_prediction, actual_labels, classes, _ = (
                _cached_task_arrays(
                    cache_dir,
                    model_name="ResNet18",
                    source="CIFAR10",
                    num_classes=2,
                    seed=42,
                    semantics="published",
                )
            )
            self.assertEqual(classes, [53, 64])
            np.testing.assert_array_equal(actual_labels, [0, 1, 0, 1])
            np.testing.assert_array_equal(actual_clustering, clustering[[0, 2, 3, 5]])
            np.testing.assert_array_equal(actual_prediction, prediction[[0, 2, 3, 5]])

    def test_cached_leep_score_uses_reconstructed_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            labels = np.asarray([53, 64, 53, 64], dtype=np.int64)
            logits = np.asarray([[3, 0], [0, 3], [2, 0], [0, 2]], dtype=np.float32)
            cache_dir = self._write_test_cache(temporary, labels, logits, logits)
            result = score_exp2_cached_task(
                cache_dir=cache_dir,
                model_name="ResNet18",
                source="CIFAR10",
                num_classes=2,
                metrics=["LEEP"],
                seed=42,
                random_state=0,
                covariance_type="full",
                semantics="published",
                formula="legacy",
            )
            self.assertEqual(result["sample_count"], 4)
            self.assertTrue(0.5 < result["scores"]["LEEP"] < 1.0)


if __name__ == "__main__":
    unittest.main()
