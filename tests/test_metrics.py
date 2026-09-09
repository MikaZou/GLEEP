import unittest


class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy  # noqa: F401
        except Exception as exc:
            raise unittest.SkipTest(f"NumPy unavailable: {exc}")

    def test_legacy_and_canonical_leep_are_distinct(self):
        import numpy as np

        from gleep_repro.metrics import leep_from_probabilities

        probabilities = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        labels = np.asarray([0, 0, 1, 1])
        legacy = leep_from_probabilities(probabilities, labels, formula="legacy")
        canonical = leep_from_probabilities(probabilities, labels, formula="canonical")
        self.assertGreater(legacy, 0.0)
        self.assertLess(canonical, 0.0)

    def test_source_class_dimension_is_dynamic(self):
        import numpy as np

        from gleep_repro.metrics import leep_from_probabilities

        probabilities = np.full((6, 7), 1.0 / 7.0)
        labels = np.asarray([0, 0, 1, 1, 2, 2])
        self.assertAlmostEqual(leep_from_probabilities(probabilities, labels), 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()

