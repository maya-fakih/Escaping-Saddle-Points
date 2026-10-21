"""
Checks for GD only.

    python -m unittest tests.test_gd
"""
import unittest
import numpy as np

from saddle.algorithms.gd import GD
from saddle.benchmark import Benchmark
from saddle.cases import CASES


def run(case, iters=3000):
    gamma, z0 = CASES[case]
    a = GD(Benchmark(gamma), z0, max_iters=iters)
    a.run()
    return a


class TestGD(unittest.TestCase):
    def test_f_never_goes_up(self):
        a = run("S2", 500)
        f = [a.problem.peek_f(z) for z in a.path]
        self.assertTrue(np.all(np.diff(f) <= 0))

    def test_escapes_S1_and_reaches_a_minimum(self):
        a = run("S1")
        self.assertIsNone(a.failed_at)
        self.assertAlmostEqual(a.problem.peek_f(a.z), a.problem.fstar, places=10)

    def test_stuck_exactly_on_the_saddle_S4(self):
        # gradient is exactly zero at (0, 0), so GD has nothing to follow
        a = run("S4", 50)
        np.testing.assert_array_equal(a.z, [0.0, 0.0])

    def test_one_gradient_per_update(self):
        a = run("S1", 20)
        self.assertEqual(a.counts[1], 20)
        self.assertEqual(a.counts[2], 0)    # never uses the Hessian


if __name__ == "__main__":
    unittest.main()
