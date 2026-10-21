"""
Checks for Probe + Nesterov only.

    python -m unittest tests.test_probe_nesterov
"""
import unittest
import numpy as np

from saddle.algorithms.probe_nesterov import ProbeNesterov
from saddle.benchmark import Benchmark
from saddle.cases import CASES


def run(case, iters=3000, **kw):
    gamma, z0 = CASES[case]
    a = ProbeNesterov(Benchmark(gamma), z0, max_iters=iters, **kw)
    a.run()
    return a


class TestProbeNesterov(unittest.TestCase):
    def test_reaches_a_minimum_on_all_cases(self):
        for case in CASES:
            a = run(case)
            self.assertIsNone(a.failed_at, case)
            self.assertAlmostEqual(a.problem.peek_f(a.z), a.problem.fstar, places=8, msg=case)

    def test_f_never_goes_up(self):
        a = run("S3", 500)
        f = [a.problem.peek_f(z) for z in a.path]
        self.assertTrue(np.all(np.diff(f) <= 0))

    def test_escapes_exact_saddle_by_probing_S4(self):
        a = run("S4", 50)
        self.assertEqual(a.jumps[0], 0)

    def test_never_jumps_near_a_minimum_S5(self):
        self.assertEqual(run("S5").jumps, [])

    def test_no_hessian(self):
        self.assertEqual(run("S1", 100).counts[2], 0)


if __name__ == "__main__":
    unittest.main()
