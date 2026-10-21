"""
My method: learned per-axis step sizes + axis probes to escape saddles.
(Same algorithm as probe_method in the project's methods.py, written as a class.)

1. Each axis i has its own step size d_i. A step is z_new = z - D g with
   D = diag(d_1, d_2), checked with backtracking (so f never goes up).

2. Learning d_i from predicted vs actual drop. Before the step, try the move
   along axis i alone, s_i = -d_i g_i, and measure
       predicted drop  p_i   = d_i g_i^2
       actual drop     a_i   = f(z) - f(z + s_i e_i)
       ratio           rho_i = a_i / p_i  = 1 - d_i H_i / 2   (H_i = curvature along the move)
   rho_i = 1/2 means d_i landed exactly on the bottom of the parabola.
       rho_i < 1  ->  d_i <- d_i * (1 / (2(1 - rho_i)))^kappa   (move toward 1/H_i)
       rho_i >= 1 ->  d_i <- omega * d_i        (bending down: no bottom, grow)
   The measured curvature H_i = 2(1 - rho_i)/d_i is kept to size the probes.
   Cost: one extra f per axis per update. Skipped when p_i < 1e-12 (rounding noise).

3. Probing (the escape). When ||g|| <= g_trig, evaluate f at z +- delta_i u_i
   along the two axes u_1, u_2, with delta_i = sqrt(2 eps_f / |H_i|)
   ("how far until f changes by about eps_f"), clipped to [delta_min, delta_max].
   No curvature known yet -> delta_i = delta_default (= P-GD's r, to be fair).
     - some probe is lower than f(z): jump there, then do the step from there
       (one update, like P-GD's kick + GD step);
     - no probe is lower: one failed try, rotate the axes by pi/(2R) for the next
       try. After R failed tries in a row: "probably a minimum", stop probing
       until ||g|| goes above g_trig again.

hessian_start=True : variant (a), one Hessian at z0 sets the first d_i and curvatures.
hessian_start=False: variant (b), the default, start at d_i = 1 and learn.
learn_d / probes   : switch off part 2 or part 3 (with both off this is exactly GD).
"""
import numpy as np

from saddle.algorithms.base import Algorithm
from saddle.cases import G_TRIG


class Probe(Algorithm):
    name = "Probe (proposed)"

    def __init__(self, problem, z0, max_iters=3000, g_trig=G_TRIG, hessian_start=False,
                 d_min=1e-6, d_max=100.0, kappa=0.5, omega=2.0,
                 eps_f=1e-6, delta_default=1e-2, delta_min=1e-4, delta_max=1e-1,
                 R=3, learn_d=True, probes=True):
        self.g_trig, self.hessian_start = g_trig, hessian_start
        self.d_min, self.d_max, self.kappa, self.omega = d_min, d_max, kappa, omega
        self.eps_f, self.delta_default = eps_f, delta_default
        self.delta_min, self.delta_max = delta_min, delta_max
        self.R, self.learn_d, self.probes = R, learn_d, probes
        self.uses_hessian = hessian_start         # only variant (a) touches the Hessian
        super().__init__(problem, z0, max_iters)

    def setup(self):
        self.g = self.problem.grad(self.z)
        self.d = np.ones(2)                       # step size per axis
        self.curv = np.full(2, np.nan)            # curvature estimate per axis (unknown yet)
        if self.hessian_start:
            h = np.diag(self.problem.hess(self.z)).copy()
            self.curv = h.copy()
            d = np.where(h > 0, 1.0 / np.maximum(h, 1e-12), self.d_max)
            self.d = np.clip(d, self.d_min, self.d_max)
        self.fails, self.stopped = 0, False
        self.jumps, self.giveups = [], []

    # ------------------------------------------------------------------
    def update(self):
        if np.linalg.norm(self.g) > self.g_trig:
            self.fails, self.stopped = 0, False   # left the flat region: allow probing again
        elif self.probes and not self.stopped:
            self._probe()
        if self.learn_d:
            self._learn_step_sizes()
        # 1. the scaled GD step, with backtracking
        step = -self.d * self.g
        z_new, f_new = self.backtrack(self.z, self.fz, step, self.g @ (self.d * self.g))
        if z_new is None:
            return False
        self.z, self.fz = z_new, f_new
        self.g = self.problem.grad(self.z)
        return True

    # ------------------------------------------------------------------
    def _probe(self):
        """3. Look a little to each side along two axes; jump to the lowest if it helps."""
        ang = self.fails * np.pi / (2 * self.R)   # rotate the probe axes after failures
        c, s = np.cos(ang), np.sin(ang)
        U = np.array([[c, -s], [s, c]])           # columns = probe axes
        best_f, best_z = self.fz, None
        for i in range(2):
            u = U[:, i]
            if np.all(np.isfinite(self.curv)):
                h_u = abs(u**2 @ self.curv)       # curvature along u from the axis estimates
                delta = np.sqrt(2 * self.eps_f / h_u) if h_u > 0 else self.delta_max
            else:
                delta = self.delta_default
            delta = np.clip(delta, self.delta_min, self.delta_max)
            for sign in (+1, -1):
                zp = self.z + sign * delta * u
                fp = self.problem.f(zp)
                if fp < best_f:
                    best_f, best_z = fp, zp
        if best_z is not None:
            self.z, self.fz = best_z, best_f
            self.g = self.problem.grad(self.z)
            self.jumps.append(self.k)
            self.fails = 0
        else:
            self.fails += 1
            if self.fails >= self.R:
                self.stopped = True
                self.giveups.append(self.k)

    def _learn_step_sizes(self):
        """2. Compare predicted and actual drop along each axis and adjust d_i."""
        for i in range(2):
            p = self.d[i] * self.g[i]**2
            if p < 1e-12:
                continue
            zi = self.z.copy()
            zi[i] -= self.d[i] * self.g[i]
            rho = (self.fz - self.problem.f(zi)) / p
            self.curv[i] = 2 * (1 - rho) / self.d[i]
            if rho < 1:
                factor = np.clip((1.0 / (2 * (1 - rho)))**self.kappa, 0.1, 10.0)
            else:
                factor = self.omega               # bending down: grow
            self.d[i] = np.clip(self.d[i] * factor, self.d_min, self.d_max)


ALGORITHM = Probe
