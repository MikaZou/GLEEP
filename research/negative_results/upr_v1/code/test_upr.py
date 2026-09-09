import unittest
import numpy as np

from gleep_repro.upr import RidgeLOO, project_features, score_upr, structural_diagnostics


class UPRTests(unittest.TestCase):
    def test_press_matches_actual_leave_one_out_refits(self):
        rng=np.random.default_rng(3)
        x=rng.normal(size=(18,5)); targets=rng.normal(size=(18,3))
        r=RidgeLOO(x)
        brute=[]
        for i in range(len(x)):
            keep=np.arange(len(x))!=i
            f=r.f[keep]
            beta=np.linalg.solve(f.T@f+r.alpha*np.eye(f.shape[1]),f.T@targets[keep])
            brute.append(targets[i]-r.f[i]@beta)
        np.testing.assert_allclose(r.residual(targets),brute,rtol=1e-9,atol=1e-10)
        h=r.f@np.linalg.solve(r.f.T@r.f+r.alpha*np.eye(r.f.shape[1]),r.f.T)
        rr=(np.eye(len(x))-h)/r.denominator[:,None]
        self.assertAlmostEqual(r.r_frobenius_squared,float(np.sum(rr**2)),places=8)

    def test_low_rank_risk_and_epsilon_match_dense_definition(self):
        rng=np.random.default_rng(0)
        x=rng.normal(size=(22,4)); v=rng.normal(size=(22,7)); labels=np.arange(22)%3
        ridge=RidgeLOO(x)
        result=structural_diagnostics(ridge,v,labels)
        y=np.eye(3)[labels]
        epsilon=np.linalg.norm(y@y.T-v@v.T,2)
        self.assertAlmostEqual(result['epsilon_oracle'],epsilon,places=8)
        self.assertTrue(result['risk_bound_holds'])
        self.assertTrue(result['classification_bound_holds'])

    def test_label_free_interface_and_determinism(self):
        rng=np.random.default_rng(4)
        pool={key:project_features(rng.normal(size=(30,9))) for key in ['a','b','c']}
        a,_,v,_=score_upr(pool,'a'); b,_,_,_=score_upr(pool,'a')
        self.assertEqual(a,b)
        self.assertNotIn('a',a['references'])
        np.testing.assert_allclose(np.sum(v*v,axis=1),1,atol=1e-12)

    def test_constant_features_finite_not_crash(self):
        pool={key:project_features(np.ones((20,7))) for key in ['a','b']}
        a,_,_,_=score_upr(pool,'a')
        self.assertTrue(np.isfinite(a['score']))

    def test_oracle_relation_yields_valid_bound(self):
        labels=np.arange(30)%2
        y=np.eye(2)[labels]
        ridge=RidgeLOO(y)
        result=structural_diagnostics(ridge,y,labels)
        self.assertLess(result['epsilon_oracle'],1e-10)
        self.assertGreater(result['accuracy_lower_bound'],.9)


if __name__=='__main__':
    unittest.main()
