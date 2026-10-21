"""
GD with a negative-curvature step (GD-NC), as in the assignment (page 3).

WHY WE HAVE IT
  GD only looks at the slope (first-order Taylor). At a saddle the slope is ~0, but
  the second-order term of Taylor is not:
      f(z + s) ~ f(z) + grad f . s + (1/2) s^T H s
  If H has a negative eigenvalue lam with eigenvector v, then moving along v makes
  the last term negative: f goes DOWN like (1/2) t^2 lam. So instead of guessing
  (P-GD's random kick), we ask the Hessian which way bends down and go there.

THE STEP
  - if ||grad f(z)|| <= g_trig: compute H, its smallest eigenvalue lam and its vector v.
      if lam < -1e-4 (really a saddle, not a flat minimum):
        p = v or -v, whichever does not go uphill (+v when the gradient is 0)
        backtrack along p: accept t when f(z + t p) <= f(z) + 1e-4 * t^2 * lam
  - otherwise: a normal GD step with backtracking.

WHY lam < -1e-4 AND NOT lam < 0
  Near a minimum the smallest eigenvalue is >= 0, so no curvature step happens (S5).
  The small margin keeps rounding noise from triggering it.

COST
  Like GD, plus 1 Hessian and an eigen-decomposition each time the gradient is small.
  Cheap in 2D, expensive in high dimension: that is the trade-off vs P-GD.
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
