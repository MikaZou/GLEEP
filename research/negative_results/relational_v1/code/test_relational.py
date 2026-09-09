import unittest
import numpy as np
from gleep_repro.relational import relational_score, exact_cosine_relational_score


class RelationalTests(unittest.TestCase):
    def test_exact_cosine_matches_dense_matrix(self):
        rng = np.random.default_rng(9)
        p = rng.dirichlet(np.ones(5), size=30)
        r = rng.normal(size=(30,7)); r /= np.linalg.norm(r,axis=1,keepdims=True)
        q = (p/p.sum(0))@p.T
        a = (1+r@r.T)/2
        raw = (q*a).sum()/30
        np.fill_diagonal(a,0); np.fill_diagonal(q,0)
        exact = (q*(a-a.sum(1)[:,None]/29)).sum()/30
        result = exact_cosine_relational_score(p,r)
        self.assertAlmostEqual(result['rpt_cosine_exact'],exact,places=12)
        self.assertAlmostEqual(result['relation_cosine_raw'],raw,places=12)

    def test_sampling_matches_dense_oracle_with_zero_source_column(self):
        rng = np.random.default_rng(12)
        p = np.column_stack([rng.dirichlet(np.ones(3), size=25), np.zeros(25)])
        r = rng.normal(size=(25, 6)); r /= np.linalg.norm(r, axis=1, keepdims=True)
        q = (p[:, :3] / p[:, :3].sum(0)) @ p[:, :3].T
        a = np.exp(-np.maximum(0, 2-2*r@r.T)/2)
        np.fill_diagonal(a, 0)
        np.fill_diagonal(q, 0)
        exact = (q * (a-a.sum(1)[:, None]/24)).sum()/25
        result = relational_score(p, r, draws=150000, bandwidth2=1)
        self.assertLess(abs(result['rpt']-exact), 5*result['rpt_se'])

    def test_identity_transport_is_exactly_zero(self):
        result = relational_score(np.eye(8), np.eye(8), draws=2000, bandwidth2=1)
        self.assertEqual(result['rpt'], 0)
        self.assertEqual(result['self_mass'], 1)

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):
            relational_score(np.ones((3,2)), np.ones((3,4)))
        with self.assertRaises(ValueError):
            relational_score(np.ones((3,2))/2, np.zeros((3,4)))


if __name__ == '__main__':
    unittest.main()
