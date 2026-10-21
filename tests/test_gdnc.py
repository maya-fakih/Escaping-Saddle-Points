"""
Checks for GD-NC only.

    python -m unittest tests.test_gdnc
"""
import unittest

from saddle.algorithms.gdnc import GDNC
from saddle.benchmark import Benchmark
from saddle.cases import CASES


def run(case, iters=3000, **kw):
    gamma, z0 = CASES[case]
    a = GDNC(Benchmark(gamma), z0, max_iters=iters, **kw)
    a.run()
    return a


class TestGDNC(unittest.TestCase):
    def test_escapes_the_exact_saddle_S4(self):
        a = run("S4")
        self.assertAlmostEqual(a.problem.peek_f(a.z), a.problem.fstar, places=8)
        self.assertEqual(a.nc_steps[0], 0)       # first move is the curvature step

    def test_escapes_weak_curvature_S3(self):
        a = run("S3")
        self.assertAlmostEqual(a.problem.peek_f(a.z), a.problem.fstar, places=8)

    def test_no_curvature_step_near_a_minimum_S5(self):
        # at a minimum the Hessian has no negative eigenvalue, so no NC step
        a = run("S5")
        self.assertEqual(a.nc_steps, [])

    def test_hessian_shown_in_simulator(self):
        self.assertTrue(GDNC.uses_hessian)


if __name__ == "__main__":
    unittest.main()
