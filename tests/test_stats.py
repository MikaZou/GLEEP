import math
import unittest

from gleep_repro.stats import kendall_tau_b, pearsonr


class CorrelationTests(unittest.TestCase):
    def test_perfect_correlations(self):
        self.assertAlmostEqual(pearsonr([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(kendall_tau_b([1, 2, 3], [2, 4, 6]), 1.0)

    def test_reverse_correlations(self):
        self.assertAlmostEqual(pearsonr([1, 2, 3], [6, 4, 2]), -1.0)
        self.assertAlmostEqual(kendall_tau_b([1, 2, 3], [6, 4, 2]), -1.0)

    def test_tau_b_ties(self):
        value = kendall_tau_b([1, 1, 2], [1, 2, 3])
        self.assertAlmostEqual(value, 2.0 / math.sqrt(6.0))


if __name__ == "__main__":
    unittest.main()

