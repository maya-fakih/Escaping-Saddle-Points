"""
Checks for P-GD only.

    python -m unittest tests.test_pgd
"""
import unittest
import numpy as np

from saddle.algorithms.pgd import PGD
from saddle.benchmark import Benchmark
from saddle.cases import CASES


def run(case, seed=0, iters=3000, **kw):
    gamma, z0 = CASES[case]
    a = PGD(Benchmark(gamma), z0, max_iters=iters, seed=seed, **kw)
    a.run()
    return a


class TestPGD(unittest.TestCase):
    def test_escapes_the_exact_saddle_S4(self):
        # GD is stuck here forever; the kick must get P-GD out, for every seed
        for seed in range(10):
            a = run("S4", seed)
            self.assertAlmostEqual(a.problem.peek_f(a.z), a.problem.fstar, places=8)
            self.assertGreater(len(a.kicks), 0)

    def test_same_seed_same_path(self):
        a, b = run("S2", 3, 300), run("S2", 3, 300)
        np.testing.assert_array_equal(a.path, b.path)

    def test_kicks_are_at_least_T_apart(self):
        a = run("S3", 0, 3000, T=20)
        self.assertTrue(np.all(np.diff(a.kicks) >= 20))

    def test_kicks_right_away_on_the_saddle(self):
        P = Benchmark(1.0)
        a = PGD(P, (0.0, 0.0), max_iters=1, seed=0, r=1e-2)
        before = P.nf
        a.step()
        self.assertEqual(a.kicks, [0])
        self.assertGreater(P.nf, before)   # f was evaluated at the kicked point


if __name__ == "__main__":
    unittest.main()
