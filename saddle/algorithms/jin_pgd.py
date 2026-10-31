"""
Perturbed gradient descent as written in the paper:
C. Jin, R. Ge, P. Netrapalli, S. Kakade, M. Jordan, "How to Escape Saddle Points
Efficiently", ICML 2017, Algorithm 2.

WHY WE HAVE IT
  The assignment's P-GD is a "practical" simplification of this algorithm. The paper's
  version has one extra rule that matters a lot: after a kick it waits t_thres steps and
  checks whether f really went down. If it did not go down by at least f_thres, the paper
  concludes "this flat point is not a saddle, it is (approximately) a minimum" and STOPS,
  returning the point from before the kick. That stop rule is what gives the paper its
  guarantee, and it is also what lets us compare against the paper fairly: our method's
  worst case (no probe tells us anything, so we kick at random) is this algorithm.

THE STEP (one update)
  1. If we kicked exactly t_thres updates ago, check the kick:
       f(now) - f(before the kick) > -f_thres   ->  no real drop: go back to the point
                                                    before the kick and stop for good
       otherwise                                ->  the kick worked: carry on
  2. If ||grad f(z)|| <= g_thres and more than t_thres updates passed since the last kick:
       remember (z, f(z)) as "before the kick", then kick: z <- z + xi,
       xi uniform in a ball of radius r. Recompute f and the gradient there.
  3. A gradient step with backtracking (Armijo, as in GD).

DIFFERENCES FROM THE PAPER (on purpose, so only the escape rule is compared)
  - The paper uses a fixed step eta = c / l. We use the same backtracking as every
    other method here, so step-size choice is not what we are comparing.
  - The paper sets r, g_thres, t_thres, f_thres from constants (l, rho, epsilon, chi, c).
    We use the assignment's numbers where they exist: r = 1e-2, g_thres = g_trig = 1e-3,
    t_thres = T = 20, and f_thres = 1e-7 (small: "f has really changed").

COST
  Like GD, plus 1 f + 1 gradient at each kick. No Hessian. After stopping, no more work.
"""
import numpy as np

from saddle.algorithms.base import Algorithm
from saddle.cases import G_TRIG, R, T


class JinPGD(Algorithm):
    name = "Jin et al. PGD"

    def __init__(self, problem, z0, max_iters=3000, seed=0, g_trig=G_TRIG, r=R, T=T,
                 f_thres=1e-7):
        # g_trig and T are the assignment's names for the paper's g_thres and t_thres
        self.seed, self.g_thres, self.r, self.t_thres, self.f_thres = seed, g_trig, r, T, f_thres
        super().__init__(problem, z0, max_iters)

    def setup(self):
        self.rng = np.random.default_rng(self.seed)   # same seed -> same kicks
        self.g = self.problem.grad(self.z)
        self.t_noise = -np.inf                        # when we last kicked (none yet)
        self.before = None                            # (z, f) just before the last kick
        self.stopped = False                          # True once the paper's stop rule fires
        self.kicks = []

    def update(self):
        # once the paper has stopped, the point stays where it is (and costs nothing)
        if self.stopped:
            return True

        # 1. t_thres updates after a kick: did f really go down?
        if self.before is not None and self.k - self.t_noise == self.t_thres:
            z_before, f_before = self.before
            self.before = None
            if self.fz - f_before > -self.f_thres:
                # no real drop: the paper calls this point a minimum and returns it
                self.log_round("gave up")
                self.z, self.fz = z_before, f_before
                self.stopped = True
                return True
            self.log_round("escaped")

        # 2. small gradient and no recent kick: kick inside a ball of radius r
        if np.linalg.norm(self.g) <= self.g_thres and self.k - self.t_noise > self.t_thres:
            self.before = (self.z.copy(), self.fz)
            self.t_noise = self.k
            self.kicks.append(self.k)
            n = len(self.z)
            v = self.rng.standard_normal(n)
            u = self.rng.uniform()
            self.z = self.z + self.r * u**(1.0 / n) * v / np.linalg.norm(v)
            self.fz, self.g = self.problem.f(self.z), self.problem.grad(self.z)

        # 3. the gradient step
        z_new, f_new = self.backtrack(self.z, self.fz, -self.g, self.g @ self.g)
        if z_new is None:
            return False
        self.z, self.fz = z_new, f_new
        self.g = self.problem.grad(self.z)
        return True


ALGORITHM = JinPGD
