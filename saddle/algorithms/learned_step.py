"""
Learned per-axis step sizes ALONE: a diagonal Hessian approximation from
"predicted vs actual" drops. No probes, no kicks, no momentum.

WHY WE HAVE IT
  The idea: each direction gets its own step size, learned by checking how much f
  really changed. Along a direction that curves UP (x here), the right step is
  about 1/curvature. Along a direction that curves DOWN (y near the saddle), there
  is no bottom, so the step keeps growing. So the point is pulled out along the
  downhill direction faster and faster, instead of crawling like GD, where the same
  small step is used for both directions.

  What it can NOT do: start exactly ON the saddle. There the gradient is exactly 0,
  so every predicted drop is 0, nothing is learned, and z - D*0 = z: it never moves.
  It needs at least a tiny y to amplify.

THE STEP
  1. For each axis i, try the move along that axis alone, s_i = -d_i g_i, and compare
         predicted drop  p_i   = d_i g_i^2                 (first-order Taylor)
         actual drop     a_i   = f(z) - f(z + s_i e_i)       (one real f evaluation)
         ratio           rho_i = a_i / p_i = 1 - d_i H_i / 2  (H_i = curvature along it)
     rho_i = 1/2 means d_i = 1/H_i: right at the bottom of the parabola.
         rho_i < 1  ->  d_i <- d_i * (1 / (2 (1 - rho_i)))^kappa   (move toward 1/H_i)
         rho_i >= 1 ->  d_i <- omega * d_i        (curving down: no bottom, grow)
     Skipped when p_i < 1e-12 (the subtraction would be rounding noise).
     The factor is kept in [0.1, 10] and d_i in [d_min, d_max].
  2. The step z_new = z - D g, D = diag(d_1, d_2), with Armijo backtracking
     (so f never goes up).

  This is exactly part 1+2 of the "probe" method with the probes switched off.

THE KNOBS
  kappa : how fast d_i moves toward 1/H_i (1 = jump straight there)
  omega : growth factor when the direction curves down
  d_min, d_max : limits on the step sizes

COST
  GD's cost + 1 f per axis per update. No Hessian.
"""
import numpy as np

from saddle.algorithms.base import Algorithm


class LearnedStep(Algorithm):
    name = "Learned step"

    def __init__(self, problem, z0, max_iters=3000, kappa=0.5, omega=2.0,
                 d_min=1e-6, d_max=100.0):
        self.kappa, self.omega, self.d_min, self.d_max = kappa, omega, d_min, d_max
        super().__init__(problem, z0, max_iters)

    def setup(self):
        self.g = self.problem.grad(self.z)
        self.d = np.ones(2)            # one step size per axis, start like GD (t = 1)

    def update(self):
        self._learn_step_sizes()
        step = -self.d * self.g
        z_new, f_new = self.backtrack(self.z, self.fz, step, self.g @ (self.d * self.g))
        if z_new is None:
            return False
        self.z, self.fz = z_new, f_new
        self.g = self.problem.grad(self.z)
        return True

    def _learn_step_sizes(self):
        """Compare predicted and actual drop along each axis and adjust d_i."""
        for i in range(2):
            p = self.d[i] * self.g[i]**2
            if p < 1e-12:
                continue
            zi = self.z.copy()
            zi[i] -= self.d[i] * self.g[i]
            rho = (self.fz - self.problem.f(zi)) / p
            if rho < 1:
                factor = np.clip((1.0 / (2 * (1 - rho)))**self.kappa, 0.1, 10.0)
            else:
                factor = self.omega               # curving down: grow
            self.d[i] = np.clip(self.d[i] * factor, self.d_min, self.d_max)


ALGORITHM = LearnedStep
