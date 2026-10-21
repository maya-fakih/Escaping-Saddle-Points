"""
Plain gradient descent with backtracking (assignment, page 2).

WHY WE HAVE IT
  It is the baseline: the simplest method, and the one the other three try to fix.
  Near z, f looks like its tangent plane (first-order Taylor):
      f(z + s) ~ f(z) + grad f(z) . s
  The plane drops fastest along -grad f(z), so we step that way.

WHAT GOES WRONG AT A SADDLE
  The plane only sees the slope, not the curvature. Next to a saddle the slope is
  almost zero, so the steps are almost zero: GD crawls (the y part grows only like
  (1 + t*gamma)^k, very slowly when gamma is small, e.g. S3). Exactly ON the saddle
  (S4) the gradient is exactly zero and GD never moves. GD alone cannot tell
  "flat because saddle" from "flat because minimum".

THE STEP
  z_{k+1} = z_k - t * grad f(z_k), with t from Armijo backtracking:
  try t = 1, 1/2, 1/4, ... (at most 40 times) and keep the first t that lowers f
  by at least 1e-4 * t * ||grad||^2. So f never goes up.

COST PER UPDATE
  1 gradient + (number of backtracking tries) function values. No Hessian.
"""
from saddle.algorithms.base import Algorithm


class GD(Algorithm):
    name = "GD"

    def update(self):
        g = self.problem.grad(self.z)
        z_new, f_new = self.backtrack(self.z, self.fz, -g, g @ g)
        if z_new is None:
            return False
        self.z, self.fz = z_new, f_new
        return True


ALGORITHM = GD
