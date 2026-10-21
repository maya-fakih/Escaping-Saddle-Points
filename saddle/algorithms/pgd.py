"""
Perturbed gradient descent, the assignment's practical version (page 2-3).

Each update:
  1. if ||grad f(z)|| <= g_trig and at least T updates passed since the last kick:
       kick: w = z + (random point in a disk of radius r)
     otherwise w = z
  2. one GD step with backtracking FROM w.

The random kick pushes the point off the saddle, so the negative curvature
can take over. Without a kick, GD started exactly on the saddle never moves.
"""
import numpy as np

from saddle.algorithms.base import Algorithm
from saddle.cases import G_TRIG, R, T


class PGD(Algorithm):
    name = "P-GD"

    def __init__(self, problem, z0, max_iters=3000, seed=0, g_trig=G_TRIG, r=R, T=T):
        self.seed, self.g_trig, self.r, self.T = seed, g_trig, r, T
        super().__init__(problem, z0, max_iters)

    def setup(self):
        self.rng = np.random.default_rng(self.seed)   # same seed -> same kicks
        self.last_kick = -self.T                      # so a kick is allowed at k = 0
        self.kicks = []                               # iterations where we kicked
        self.g = self.problem.grad(self.z)

    def update(self):
        w, fw, gw = self.z, self.fz, self.g
        if np.linalg.norm(self.g) <= self.g_trig and self.k - self.last_kick >= self.T:
            # uniform point in a disk: random direction, radius r*sqrt(u)
            v = self.rng.standard_normal(2)
            u = self.rng.uniform()
            w = self.z + self.r * np.sqrt(u) * v / np.linalg.norm(v)
            fw, gw = self.problem.f(w), self.problem.grad(w)
            self.last_kick = self.k
            self.kicks.append(self.k)
        z_new, f_new = self.backtrack(w, fw, -gw, gw @ gw)
        if z_new is None:
            return False
        self.z, self.fz = z_new, f_new
        self.g = self.problem.grad(self.z)
        return True


ALGORITHM = PGD
