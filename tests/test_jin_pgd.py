"""
Checks for Jin et al.'s perturbed GD only.

    python -m unittest tests.test_jin_pgd
"""
import unittest
import numpy as np

from saddle.algorithms.jin_pgd import JinPGD
from saddle.benchmark import Benchmark
from saddle.cases import CASES


def run(case, seed=0, iters=3000, **kw):
    gamma, z0 = CASES[case]
    a = JinPGD(Benchmark(gamma), z0, max_iters=iters, seed=seed, **kw)
    a.run()
    return a


class TestJinPGD(unittest.TestCase):
    def test_escapes_the_exact_saddle_S4(self):
        for seed in range(10):
            a = run("S4", seed)
            self.assertAlmostEqual(a.problem.peek_f(a.z), a.problem.fstar, places=8)

    def test_stops_at_a_minimum_and_stays(self):
        # near a minimum the kick cannot lower f, so the stop rule fires and f stays put
        a = run("S5")
        self.assertTrue(a.stopped)
        self.assertEqual(a.probe_log[-1][1], "gave up")
        nf = a.counts[0]
        a.max_iters += 10; a.done = False
        for _ in range(10):
            a.step()
        self.assertEqual(a.counts[0], nf)          # no more work after stopping

    def test_kicks_are_more_than_t_thres_apart(self):
        a = run("S3", 0)
        self.assertTrue(np.all(np.diff(a.kicks) > 20))


if __name__ == "__main__":
    unittest.main()
