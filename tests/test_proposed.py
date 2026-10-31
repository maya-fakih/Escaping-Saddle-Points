"""
Checks for the proposed method only (each part, then all together).

    python -m unittest tests.test_proposed
"""
import unittest
import numpy as np

from saddle.algorithms.proposed import Proposed, smallest_directions
from saddle.algorithms.ldh import LDH
from saddle.algorithms.ldh_guessed import LDHGuessed
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

    def test_curvature_steps_follow_the_turned_valley(self):
        # learn_step="curvature": the step directions are B's eigenvectors, which on
        # the 45 deg function are the turned axes (1, 1)/sqrt 2 and (1, -1)/sqrt 2
        a = Proposed(RotatedSaddle.rotated_2d(0.1, 45), (0.05, 1e-8), learn_step="curvature")
        a.step()
        cos = np.abs(a.V.T @ (np.array([[1, 1], [1, -1]]) / np.sqrt(2)))
        self.assertTrue(np.allclose(np.sort(cos.max(axis=1)), 1.0, atol=1e-3))

    def test_learn_step_true_means_axes(self):
        a = Proposed(RotatedSaddle.plain(1.0), (0.05, 1e-8), learn_step=True)
        a.step()
        self.assertEqual(a.learn_step, "axes")
        self.assertIsNone(a.V)

    def test_tridiagonal_solver_gives_the_smallest_directions(self):
        rng = np.random.default_rng(1)
        B = np.diag(rng.normal(size=8)) + np.diag(rng.normal(size=7), 1)
        B = B + np.triu(B, 1).T
        V, W = smallest_directions(B, 3, 1), np.linalg.eigh(B)[1][:, :3]
        self.assertTrue(np.allclose(np.abs(V.T @ W), np.eye(3), atol=1e-8))

    def test_guessed_B_matches_the_hessian_near_the_saddle(self):
        # small moves: second-order Taylor is accurate, so the guessed entries are too
        P = RotatedSaddle.rotated_2d(0.5, 30)
        a = LDHGuessed(P, np.array([0.01, 0.005]))
        a.step()
        H = P.peek_hess(a.z)
        self.assertAlmostEqual(a.B_off[0], H[0, 1], delta=1e-2)
        self.assertTrue(np.allclose(a.B_diag, np.diag(H), atol=1e-2))

    def test_guessed_needs_per_axis_steps(self):
        with self.assertRaises(ValueError):
            Proposed(RotatedSaddle.plain(1.0), (0.0, 0.0), escape="guessed", learn_step=False)

    def test_ldh_and_guessed_reach_the_minimum_in_2d(self):
        for cls in (LDH, LDHGuessed):
            for P in (RotatedSaddle.plain(0.01), RotatedSaddle.rotated_2d(0.01, 45)):
                for start in ("far", "close", "on", "near_min"):
                    a = cls(P, P.start(start))
                    a.run()
                    self.assertTrue(at_min(a), (cls.name, P.label, start))

    def test_no_hessian_ever(self):
        self.assertEqual(run(RotatedSaddle.coupled(20, 0.1), "on", iters=200).counts[2], 0)


if __name__ == "__main__":
    unittest.main()
