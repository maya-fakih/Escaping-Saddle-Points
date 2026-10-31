"""
Checks for the proposed method only (each part, then all together).

    python -m unittest tests.test_proposed
"""
import unittest
import numpy as np

from saddle.algorithms.proposed import Proposed
from saddle.benchmark import RotatedSaddle


def run(problem, start, iters=3000, **parts):
    a = Proposed(problem, problem.start(start), max_iters=iters, **parts)
    a.run()
    return a


def at_min(a):
    return abs(a.problem.peek_f(a.z) - a.problem.fstar) < 1e-10


class TestBandedCurvature(unittest.TestCase):
    def test_band_one_is_exact_when_only_neighbours_couple(self):
        P = RotatedSaddle.coupled(10, 0.1)
        a = Proposed(P, P.start("close"), band=1)
        np.testing.assert_allclose(a.measure_curvature(), P.peek_hess(a.z), atol=1e-4)

    def test_full_band_is_exact_for_any_rotation(self):
        P = RotatedSaddle.random(10, 0.1)
        a = Proposed(P, P.start("close"), band="full")
        np.testing.assert_allclose(a.measure_curvature(), P.peek_hess(a.z), atol=1e-4)

    def test_cost_is_2b_plus_1_gradients(self):
        P = RotatedSaddle.coupled(100, 0.1)
        a = Proposed(P, P.start("close"), band=1)
        before = a.counts[1]
        a.measure_curvature()
        self.assertEqual(a.counts[1] - before, 3)


class TestParts(unittest.TestCase):
    def test_axis_probes_fail_on_the_turned_saddle(self):
        # the downhill direction is between the axes: every axis probe goes up
        a = run(RotatedSaddle.rotated_2d(0.01, 45), "on", escape="axes", kick=False,
                penalty=False)
        self.assertFalse(at_min(a))
        self.assertEqual(a.probe_log[-1][1], "gave up")

    def test_bag_from_the_band_escapes_the_turned_saddle(self):
        a = run(RotatedSaddle.rotated_2d(0.01, 45), "on", kick=False, penalty=False)
        self.assertTrue(at_min(a))

    def test_full_method_everywhere_in_2d(self):
        for P in (RotatedSaddle.plain(0.01), RotatedSaddle.rotated_2d(0.01, 45)):
            for start in ("far", "close", "on", "near_min"):
                self.assertTrue(at_min(run(P, start)), (P.label, start))

    def test_kicks_at_a_minimum_are_undone(self):
        # at a minimum nothing drops: it may kick R-1 times, then gives up once,
        # goes back to the best point (the minimum) and never searches again
        a = run(RotatedSaddle.plain(1.0), "near_min")
        outcomes = [o for _, o in a.probe_log]
        self.assertNotIn("escaped", outcomes)
        self.assertEqual(outcomes.count("gave up"), 1)
        self.assertTrue(at_min(a))

    def test_penalty_fades(self):
        a = Proposed(RotatedSaddle.plain(1.0), (0.0, 0.0), beta=0.5)
        a.p[:] = 1.0
        a.step()
        self.assertTrue(np.all(a.p <= 0.5 + 1e-12))

    def test_no_hessian_ever(self):
        self.assertEqual(run(RotatedSaddle.coupled(20, 0.1), "on", iters=200).counts[2], 0)


if __name__ == "__main__":
    unittest.main()
