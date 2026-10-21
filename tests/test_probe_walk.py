"""
Checks for Probe + walk only.

    python -m unittest tests.test_probe_walk
"""
import unittest
import numpy as np

from saddle.algorithms.probe_walk import ProbeWalk
from saddle.benchmark import Benchmark, Star
from saddle.cases import CASES


def run(problem, z0, iters=3000, **kw):
    a = ProbeWalk(problem, z0, max_iters=iters, **kw)
    a.run()
    return a


class TestProbeWalk(unittest.TestCase):
    def test_reaches_a_minimum_on_all_assignment_cases(self):
        for case, (gamma, z0) in CASES.items():
            a = run(Benchmark(gamma), z0)
            self.assertAlmostEqual(a.problem.peek_f(a.z), a.problem.fstar, places=8, msg=case)

    def test_star_saddle_is_flat_so_it_walks_then_escapes(self):
        # at (0, 0) of the star, the axis probes change f by only ~1e-9: "equal", so walk
        for seed in range(10):
            a = run(Star(), (0.0, 0.0), seed=seed)
            self.assertEqual(a.walk_steps[0], 0)
            self.assertAlmostEqual(a.problem.peek_f(a.z), a.problem.fstar, places=8)

    def test_star_from_the_axis(self):
        a = run(Star(), (0.5, 0.0))
        self.assertAlmostEqual(a.problem.peek_f(a.z), a.problem.fstar, places=8)

    def test_assignment_saddle_is_not_flat_so_no_walk(self):
        # f_gamma=1 drops by ~5e-5 along y at distance 0.01: a clear LOWER probe
        a = run(Benchmark(1.0), (0.0, 0.0), iters=50)
        self.assertEqual(a.walk_steps, [])
        self.assertEqual(a.jumps[0], 0)

    def test_gives_up_and_stays_at_a_minimum(self):
        P = Star()
        a = run(P, P.minima[0], iters=100)
        self.assertEqual(a.jumps, [])
        self.assertEqual(len(a.giveups), 1)
        np.testing.assert_allclose(a.z, P.minima[0])


if __name__ == "__main__":
    unittest.main()
