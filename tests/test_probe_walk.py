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

    def test_star_saddle_needs_a_walk_then_escapes(self):
        # at (0, 0) of the star, all 4 axis probes go up: the first round must walk
        for seed in range(10):
            a = run(Star(), (0.0, 0.0), seed=seed)
            self.assertEqual(a.walk_steps[0], 0)
            self.assertAlmostEqual(a.problem.peek_f(a.z), a.problem.fstar, places=8)

    def test_star_from_the_axis(self):
        a = run(Star(), (0.5, 0.0))
        self.assertAlmostEqual(a.problem.peek_f(a.z), a.problem.fstar, places=8)

    def test_gives_up_and_stays_at_a_minimum(self):
        P = Star()
        a = run(P, P.minima[0], iters=100)
        self.assertEqual(a.jumps, [])
        self.assertEqual(len(a.giveups), 1)
        np.testing.assert_allclose(a.z, P.minima[0])


if __name__ == "__main__":
    unittest.main()
