"""
Probe to escape, then Nesterov to run downhill.

WHY WE HAVE IT
  Away from the saddle, plain GD is fine but slow; Nesterov's momentum is the
  classic way to go faster with the same gradients. The only place momentum (or
  GD) cannot help is when the gradient is ~0: there is nothing to accelerate.
  So: ONLY when the gradient is too small, look around with probes and jump out;
  the rest of the time, run Nesterov normally. (Compared to the "probe" method,
  there are no learned per-axis step sizes here.)

THE STEP
  1. Look-ahead point: w = z + beta * (z - z_prev), beta = j / (j + 3), where j
     counts updates since the last restart (beta starts at 0, grows toward 1).
  2. If ||grad f(w)|| <= g_trig and we have not given up: probe.
       f at z +- r u_i along two axes u_1, u_2 (r = P-GD's kick size, to be fair).
       - a probe is lower than f(z): jump there, restart momentum.
       - none is lower: rotate the axes by pi/(2R) and try next time;
         after R failed tries in a row: "probably a minimum", stop probing until
         the gradient is large again.
  3. Otherwise a gradient step FROM w with backtracking (Armijo, as in GD).
     If the new point is higher than z, momentum overshot: restart it
     (j = 0, so w = z) and take a plain GD step from z instead. So f never goes up.

THE KNOBS
  g_trig : "gradient is small" threshold (same as the others)
  r      : probe length
  R      : failed probe rounds before giving up

COST
  1 gradient + backtracking per update; 4 f per probe round. No Hessian.
"""
import numpy as np

from saddle.algorithms.base import Algorithm
from saddle.cases import G_TRIG, R as R_DEFAULT


class ProbeNesterov(Algorithm):
    name = "Probe + Nesterov"

    def __init__(self, problem, z0, max_iters=3000, g_trig=G_TRIG, r=R_DEFAULT, R=3):
        self.g_trig, self.r, self.R = g_trig, r, R
        super().__init__(problem, z0, max_iters)

    def setup(self):
        self.z_prev = self.z.copy()   # previous point, for the momentum
        self.j = 0                    # updates since the last momentum restart
        self.fails, self.stopped = 0, False
        self.jumps, self.giveups, self.restarts = [], [], []

    # ------------------------------------------------------------------
    def update(self):
        beta = self.j / (self.j + 3)
        w = self.z + beta * (self.z - self.z_prev)
        g = self.problem.grad(w)

        if np.linalg.norm(g) > self.g_trig:
            self.fails, self.stopped = 0, False       # left the flat region: probing allowed again
        elif not self.stopped and self._probe():
            return True                               # jumped: that is this update

        # gradient step from the look-ahead point
        fw = self.fz if beta == 0 else self.problem.f(w)
        z_new, f_new = self.backtrack(w, fw, -g, g @ g)
        if z_new is None or f_new > self.fz:
            # momentum made things worse: restart it and do a plain GD step from z
            self.restarts.append(self.k)
            self.j = 0
            g = self.problem.grad(self.z)
            z_new, f_new = self.backtrack(self.z, self.fz, -g, g @ g)
            if z_new is None:
                return False
        else:
            self.j += 1
        self.z_prev, self.z, self.fz = self.z, z_new, f_new
        return True

    # ------------------------------------------------------------------
    def _probe(self):
        """Look a distance r to each side along two axes. Returns True if we jumped."""
        ang = self.fails * np.pi / (2 * self.R)       # rotate the axes after failures
        c, s = np.cos(ang), np.sin(ang)
        U = np.array([[c, -s], [s, c]])               # columns = probe axes
        best_f, best_z = self.fz, None
        for i in range(2):
            for sign in (+1, -1):
                zp = self.z + sign * self.r * U[:, i]
                fp = self.problem.f(zp)
                if fp < best_f:
                    best_f, best_z = fp, zp
        if best_z is None:
            self.fails += 1
            if self.fails >= self.R:
                self.stopped = True
                self.giveups.append(self.k)
            return False
        self.jumps.append(self.k)
        self.fails = 0
        self.z_prev = best_z.copy()                   # restart momentum at the new point
        self.j = 0
        self.z, self.fz = best_z, best_f
        return True


ALGORITHM = ProbeNesterov
