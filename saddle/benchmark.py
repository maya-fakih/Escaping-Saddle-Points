"""
The benchmark function from the assignment:

    f(x, y) = x^2/2 - gamma*y^2/2 + y^4/4

  - (0, 0) is a saddle point (curves up along x, down along y)
  - (0, +sqrt(gamma)) and (0, -sqrt(gamma)) are the two minima, f* = -gamma^2/4

Every call to f / grad / hess is counted (Nf, Ng, NH), because the assignment
asks us to report the work each method does. The "peek_*" versions are NOT
counted: the simulator uses them to show values on screen without making the
algorithm look more expensive than it is.
"""
import numpy as np


class Benchmark:
    def __init__(self, gamma=1.0):
        self.gamma = gamma
        self.reset_counts()

    def reset_counts(self):
        self.nf = self.ng = self.nh = 0

    # ---- the math (not counted) ----
    def peek_f(self, z):
        x, y = z
        return 0.5 * x**2 - 0.5 * self.gamma * y**2 + 0.25 * y**4

    def peek_grad(self, z):
        x, y = z
        return np.array([x, -self.gamma * y + y**3])

    def peek_hess(self, z):
        # f is a sum of a function of x and a function of y, so the Hessian is diagonal
        return np.diag([1.0, -self.gamma + 3 * z[1]**2])

    # ---- what the algorithms call (counted) ----
    def f(self, z):
        self.nf += 1
        return self.peek_f(z)

    def grad(self, z):
        self.ng += 1
        return self.peek_grad(z)

    def hess(self, z):
        self.nh += 1
        return self.peek_hess(z)

    # ---- known answers, handy for tests and plots ----
    @property
    def fstar(self):
        """Lowest value of f (at the two minima)."""
        return -self.gamma**2 / 4

    @property
    def minima(self):
        r = np.sqrt(self.gamma)
        return [np.array([0.0, r]), np.array([0.0, -r])]

    saddle = np.array([0.0, 0.0])
