"""
GD with a negative-curvature step (GD-NC), as in the assignment (page 3).

Each update:
  - if ||grad f(z)|| <= g_trig: compute the Hessian, take its smallest
    eigenvalue lam and its eigenvector v.
      if lam < -1e-4 (we really are near a saddle):
        p = v or -v, whichever points downhill (+v when the gradient is 0)
        backtrack along p: accept t when f(z + t p) <= f(z) + 1e-4 * t^2 * lam
  - otherwise (or if that did not apply): a normal GD step with backtracking.

It does not guess like P-GD: the Hessian tells it exactly which way the
surface bends down. The price is one Hessian (and an eigen-decomposition)
each time the gradient is small.
"""
import numpy as np

from saddle.algorithms.base import Algorithm, ARMIJO, MAX_TRIALS
from saddle.cases import G_TRIG

NEG_CURVATURE = -1e-4    # "really negative" threshold for the smallest eigenvalue


class GDNC(Algorithm):
    name = "GD-NC"
    uses_hessian = True

    def __init__(self, problem, z0, max_iters=3000, g_trig=G_TRIG):
        self.g_trig = g_trig
        super().__init__(problem, z0, max_iters)

    def setup(self):
        self.nc_steps = []     # iterations where the negative-curvature step was used

    def update(self):
        g = self.problem.grad(self.z)
        if np.linalg.norm(g) <= self.g_trig:
            lam, V = np.linalg.eigh(self.problem.hess(self.z))
            lam, v = lam[0], V[:, 0]          # eigh sorts ascending: [0] is the smallest
            if lam < NEG_CURVATURE:
                p = -v if g @ v > 0 else v
                return self._curvature_step(p, lam)
        z_new, f_new = self.backtrack(self.z, self.fz, -g, g @ g)
        if z_new is None:
            return False
        self.z, self.fz = z_new, f_new
        return True

    def _curvature_step(self, p, lam):
        t = 1.0
        for _ in range(MAX_TRIALS):
            z_try = self.z + t * p
            f_try = self.problem.f(z_try)
            if f_try <= self.fz + ARMIJO * t**2 * lam:
                self.z, self.fz = z_try, f_try
                self.nc_steps.append(self.k)
                return True
            t *= 0.5
        return False


ALGORITHM = GDNC
