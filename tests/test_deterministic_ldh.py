import unittest
import numpy as np
from saddle.algorithms.ldh import LDH
from saddle.benchmark import RotatedSaddle

class TestDeterministicLDH(unittest.TestCase):
    def test_unequal_uphill_probes_move_to_best_without_random_kick(self):
        p=RotatedSaddle.plain(.1)
        a=LDH(p,np.array([0.,np.sqrt(.1)]),R=3,penalty=False)
        a._directions=lambda:np.eye(2)
        candidates=[(p.peek_f(a.z+sign*a.delta*v),a.z+sign*a.delta*v)
                    for v in np.eye(2) for sign in (1,-1)]
        expected=min(candidates,key=lambda item:item[0])
        a._search()
        np.testing.assert_array_equal(a.z,expected[1])
        self.assertEqual(a.probe_log[-1][1],'walked')
        self.assertEqual(a.kicks,[])

    def test_optimizer_seed_has_no_effect(self):
        paths=[]
        for seed in (0,123):
            p=RotatedSaddle.random(10,.1,seed=0)
            a=LDH(p,p.start('on'),seed=seed,max_iters=30,R=20)
            a.run();paths.append(np.asarray(a.path))
            self.assertEqual(a.kicks,[])
        np.testing.assert_array_equal(*paths)

if __name__=='__main__':unittest.main()
