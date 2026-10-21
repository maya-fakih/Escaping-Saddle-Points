"""
Checks for the shared pieces on main: the benchmark's math and the base class.

    python -m unittest tests.test_benchmark
"""
import unittest
import numpy as np

from saddle.benchmark import Benchmark, Star
from saddle.algorithms.base import Algorithm


class TestBenchmark(unittest.TestCase):
    def test_gradient_matches_finite_differences(self):
        P, h = Benchmark(0.1), 1e-6
        for z in [np.array([0.3, -0.7]), np.array([0.05, 1e-8]), np.array([-1.2, 0.4])]:
            num = [(P.peek_f(z + h * e) - P.peek_f(z - h * e)) / (2 * h) for e in np.eye(2)]
            np.testing.assert_allclose(P.peek_grad(z), num, atol=1e-8)

    def test_hessian_matches_finite_differences(self):
        P, h = Benchmark(1.0), 1e-5
        z = np.array([0.3, -0.7])
        num = np.column_stack([(P.peek_grad(z + h * e) - P.peek_grad(z - h * e)) / (2 * h)
                               for e in np.eye(2)])
        np.testing.assert_allclose(P.peek_hess(z), num, atol=1e-8)

    def test_stationary_points(self):
        P = Benchmark(0.1)
        np.testing.assert_allclose(P.peek_grad(P.saddle), 0)
        for m in P.minima:
            np.testing.assert_allclose(P.peek_grad(m), 0, atol=1e-15)
            self.assertAlmostEqual(P.peek_f(m), P.fstar)
        # saddle: one positive and one negative curvature
        lam = np.linalg.eigvalsh(P.peek_hess(P.saddle))
        self.assertTrue(lam[0] < 0 < lam[1])

    def test_counters(self):
        P = Benchmark(1.0)
        P.f([0, 0]); P.f([0, 0]); P.grad([0, 0]); P.peek_f([1, 1])
        self.assertEqual((P.nf, P.ng, P.nh), (2, 1, 0))


class TestStar(unittest.TestCase):
    def test_derivatives_match_finite_differences(self):
        P, h = Star(), 1e-6
        for z in [np.array([0.3, -0.7]), np.array([0.9, 0.2]), np.array([-0.4, 0.5])]:
            num_g = [(P.peek_f(z + h * e) - P.peek_f(z - h * e)) / (2 * h) for e in np.eye(2)]
            np.testing.assert_allclose(P.peek_grad(z), num_g, atol=1e-8)
            num_h = np.column_stack([(P.peek_grad(z + h * e) - P.peek_grad(z - h * e)) / (2 * h)
                                     for e in np.eye(2)])
            np.testing.assert_allclose(P.peek_hess(z), num_h, atol=1e-6)

    def test_minima_and_degenerate_saddle(self):
        P = Star()
        for m in P.minima:
            np.testing.assert_allclose(P.peek_grad(m), 0, atol=1e-12)
            self.assertAlmostEqual(P.peek_f(m), P.fstar)
        np.testing.assert_array_equal(P.peek_hess([0.0, 0.0]), 0)   # flat to 2nd order
        r = 1e-2                                                     # axis probes all go up
        for u in ([r, 0], [-r, 0], [0, r], [0, -r]):
            self.assertGreater(P.peek_f(np.array(u)), 0)


class Halfway(Algorithm):
    """A toy child just to test the parent: jump halfway to the origin each step."""
    name = "halfway"

    def update(self):
        self.z = self.z / 2
        self.fz = self.problem.f(self.z)
        return True


class TestBase(unittest.TestCase):
    def test_step_and_path(self):
        a = Halfway(Benchmark(1.0), (1.0, 1.0), max_iters=3)
        a.run()
        self.assertEqual(a.k, 3)
        self.assertTrue(a.done)
        self.assertEqual(len(a.path), 4)
        np.testing.assert_allclose(a.z, [0.125, 0.125])
        a.reset()
        self.assertEqual((a.k, len(a.path)), (0, 1))

    def test_backtracking_accepts_a_decrease(self):
        a = Halfway(Benchmark(1.0), (1.0, 0.0))
        g = a.problem.grad(a.z)
        z_new, f_new = a.backtrack(a.z, a.fz, -g, g @ g)
        self.assertLess(f_new, a.fz)


if __name__ == "__main__":
    unittest.main()
