"""
Checks for the proposed probe method only.

    python -m unittest tests.test_probe
"""
import unittest
import numpy as np

from saddle.algorithms.probe import Probe
from saddle.benchmark import Benchmark
from saddle.cases import CASES


def run(case, iters=3000, **kw):
    gamma, z0 = CASES[case]
    a = Probe(Benchmark(gamma), z0, max_iters=iters, **kw)
    a.run()
    return a


class TestProbe(unittest.TestCase):
    def test_reaches_a_minimum_on_all_cases(self):
        for case in CASES:
            a = run(case)
            self.assertIsNone(a.failed_at, case)
            self.assertAlmostEqual(a.problem.peek_f(a.z), a.problem.fstar, places=8, msg=case)

    def test_escapes_exact_saddle_by_probing_S4(self):
        a = run("S4", 50)
        self.assertEqual(a.jumps[0], 0)

    def test_never_jumps_near_a_minimum_S5(self):
        a = run("S5")
        self.assertEqual(a.jumps, [])

    def test_learned_step_matches_curvature(self):
        # along x the curvature is 1, so the learned d_x should end near 1/1 = 1
        a = run("S2", 200)
        self.assertAlmostEqual(a.d[0], 1.0, places=2)

    def test_both_parts_off_is_plain_gd(self):
        # no learning and no probes: d stays (1, 1), so every step is z - t*grad
        a = run("S2", 200, learn_d=False, probes=False)
        np.testing.assert_array_equal(a.d, [1.0, 1.0])
        self.assertEqual(a.jumps, [])
        self.assertEqual(a.counts[1], 201)    # one gradient at the start + one per update

    def test_no_hessian_unless_variant_a(self):
        self.assertEqual(run("S1", 50).counts[2], 0)
        self.assertEqual(run("S1", 50, hessian_start=True).counts[2], 1)


if __name__ == "__main__":
    unittest.main()
