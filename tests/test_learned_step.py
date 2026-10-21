"""
Checks for the learned step sizes only.

    python -m unittest tests.test_learned_step
"""
import unittest
import numpy as np

from saddle.algorithms.learned_step import LearnedStep
from saddle.benchmark import Benchmark
from saddle.cases import CASES


def run(case, iters=3000):
    gamma, z0 = CASES[case]
    a = LearnedStep(Benchmark(gamma), z0, max_iters=iters)
    a.run()
    return a


class TestLearnedStep(unittest.TestCase):
    def test_escapes_when_started_near_the_saddle(self):
        for case in ("S1", "S2", "S3"):
            a = run(case)
            self.assertAlmostEqual(a.problem.peek_f(a.z), a.problem.fstar, places=8, msg=case)

    def test_stuck_exactly_on_the_saddle_S4(self):
        # gradient is 0: nothing to learn from, and z - D*0 = z
        a = run("S4", 100)
        np.testing.assert_array_equal(a.z, [0.0, 0.0])

    def test_stays_at_the_minimum_S5(self):
        a = run("S5")
        self.assertAlmostEqual(a.problem.peek_f(a.z), a.problem.fstar, places=8)

    def test_learns_one_over_curvature_along_x(self):
        a = run("S2", 200)                     # curvature along x is 1
        self.assertAlmostEqual(a.d[0], 1.0, places=2)

    def test_no_hessian(self):
        self.assertEqual(run("S1", 50).counts[2], 0)


if __name__ == "__main__":
    unittest.main()
