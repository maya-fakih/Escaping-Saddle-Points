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

    @property
    def scale(self):
        """Distance from the saddle to a minimum (the simulator sizes its box with it)."""
        return np.sqrt(self.gamma)

    @property
    def label(self):
        return f"gamma = {self.gamma}"

    # the simulator hides the surface above this height (None = show everything)
    zcap = None


class Star(Benchmark):
    """
    A worse saddle: a 4-pointed star.

        f(x, y) = c (x^4 - 6 x^2 y^2 + y^4) + (x^2 + y^2)^3 / 6,   c = 1/4

    In polar coordinates (x = r cos(th), y = r sin(th)) the first part is
    c r^4 cos(4 th): it goes UP along the axes (th = 0, 90, 180, 270 degrees)
    and DOWN along the diagonals (45, 135, ...). The r^6 part keeps f bounded,
    so there are 4 minima on the diagonals at distance 1, f* = -1/12.

    Why it is hard:
      - at (0, 0) the gradient AND the Hessian are zero (a "degenerate" saddle),
        so GD-NC sees no negative eigenvalue and cannot escape either;
      - near the saddle f changes like r^4, so tiny pushes barely move GD;
      - the 4 axis probes all go UP: only a probe off the axes finds the way down;
      - a start on an axis (y = 0) slides straight into the saddle.
    """

    C = 0.25

    def __init__(self, gamma=None):
        super().__init__(gamma=0.0)        # gamma is not used here

    def peek_f(self, z):
        x, y = z
        s = x**2 + y**2
        return self.C * (x**4 - 6 * x**2 * y**2 + y**4) + s**3 / 6

    def peek_grad(self, z):
        x, y = z
        s = x**2 + y**2
        return np.array([self.C * (4 * x**3 - 12 * x * y**2) + s**2 * x,
                         self.C * (4 * y**3 - 12 * x**2 * y) + s**2 * y])

    def peek_hess(self, z):
        x, y = z
        s = x**2 + y**2
        fxx = self.C * (12 * x**2 - 12 * y**2) + s**2 + 4 * x**2 * s
        fyy = self.C * (12 * y**2 - 12 * x**2) + s**2 + 4 * y**2 * s
        fxy = -24 * self.C * x * y + 4 * x * y * s
        return np.array([[fxx, fxy], [fxy, fyy]])

    @property
    def fstar(self):
        return -1.0 / 12

    @property
    def minima(self):
        a = 1 / np.sqrt(2)
        return [np.array([sx * a, sy * a]) for sx in (1, -1) for sy in (1, -1)]

    @property
    def scale(self):
        return 0.75             # box of +-1.2: the minima at distance 1 fit inside

    # the r^6 part shoots up at the corners and would squash the saddle flat
    zcap = 0.25

    @property
    def label(self):
        return "star saddle"


FUNCTIONS = {"assignment": Benchmark, "star": Star}
