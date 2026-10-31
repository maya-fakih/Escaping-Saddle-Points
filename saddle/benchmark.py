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


class RotatedSaddle(Benchmark):
    """
    The benchmark's saddle in any dimension N, seen through a rotation Q.

        u = Q z                                   (turn the coordinates)
        f(z) = 1/2 * (u_1^2 + ... + u_{N-1}^2)    (N-1 directions that curve up)
               - gamma/2 * u_N^2 + 1/4 * u_N^4    (one direction that curves down, then up)

    WHY WE HAVE IT
      It is the assignment's function made bigger and turned. With N = 2 and Q = I
      it IS f_gamma(x, y) (x = u_1 curves up, y = u_2 is the saddle direction).
      Turning it does not change the landscape at all (same saddle, same minima,
      same curvatures); it only changes how the landscape lines up with the axes.
      That is exactly what we want to test: methods that look along the axes
      (probes, per-axis step sizes, a banded Hessian) should care, and methods
      that do not (GD, P-GD, GD-NC) should not.

    THE FOUR VERSIONS WE TEST (built with the class methods below)
      plain(gamma)              2D, no turn: the assignment's function
      rotated_2d(gamma, angle)  2D, turned by `angle` (45 degrees in the tests)
      coupled(N, gamma)         N dims, each pair (1,2), (3,4), ... turned by 45 degrees:
                                every coordinate only talks to its neighbour, so the
                                Hessian is banded (band 1). A banded estimate should win.
      random(N, gamma, seed)    N dims, a random rotation: every coordinate talks to
                                every other one, the Hessian is full. A band should lose.

    Known answers: saddle at z = 0 with f = 0; minima at u = (0, ..., 0, +-sqrt(gamma)),
    f* = -gamma^2/4, the same as the assignment.
    """

    def __init__(self, gamma, Q, label="rotated saddle"):
        super().__init__(gamma)
        self.Q = np.array(Q, dtype=float)
        self.n = self.Q.shape[0]
        self._label = label

    # ---- the four versions ----
    @classmethod
    def plain(cls, gamma):
        return cls(gamma, np.eye(2), f"gamma = {gamma}")

    @classmethod
    def rotated_2d(cls, gamma, angle_deg):
        a = np.radians(angle_deg)
        Q = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return cls(gamma, Q, f"gamma = {gamma}, turned {angle_deg:g} deg")

    @classmethod
    def coupled(cls, n, gamma):
        # one 45-degree turn inside each pair of neighbours: a block-diagonal Q
        c = np.sqrt(0.5)
        Q = np.zeros((n, n))
        for i in range(0, n - 1, 2):
            Q[i:i + 2, i:i + 2] = [[c, -c], [c, c]]
        if n % 2:
            Q[-1, -1] = 1.0
        return cls(gamma, Q, f"{n}D neighbours coupled, gamma = {gamma}")

    @classmethod
    def random(cls, n, gamma, seed=0):
        # QR of a random Gaussian matrix gives a random rotation
        A = np.random.default_rng(seed).standard_normal((n, n))
        Q, R = np.linalg.qr(A)
        Q = Q * np.sign(np.diag(R))            # make the choice unique
        return cls(gamma, Q, f"{n}D random rotation, gamma = {gamma}")

    # ---- the math (not counted) ----
    def _u(self, z):
        # works for one point (shape N) and for a grid of points (shape N x ...)
        return np.tensordot(self.Q, np.asarray(z, dtype=float), axes=1)

    def peek_f(self, z):
        u = self._u(z)
        up, down = u[:-1], u[-1]
        return 0.5 * np.sum(up**2, axis=0) - 0.5 * self.gamma * down**2 + 0.25 * down**4

    def peek_grad(self, z):
        u = self._u(z)
        du = u.copy()                                 # d f / d u_i = u_i for the up directions
        du[-1] = -self.gamma * u[-1] + u[-1]**3       # and this for the saddle direction
        return self.Q.T @ du                          # back to z coordinates (chain rule)

    def peek_hess(self, z):
        u = self._u(z)
        d = np.ones(self.n)
        d[-1] = -self.gamma + 3 * u[-1]**2
        return self.Q.T @ (d[:, None] * self.Q)       # Q^T diag(d) Q

    # ---- known answers ----
    @property
    def minima(self):
        r = np.sqrt(self.gamma)
        e = np.zeros(self.n); e[-1] = r
        return [self.Q.T @ e, -(self.Q.T @ e)]

    @property
    def saddle(self):
        return np.zeros(self.n)

    @property
    def label(self):
        return self._label

    def start(self, kind):
        """The four test starts, written in u (the function's own coordinates) and
        turned into z, so 'far' means the same thing in every version.
            far           1 away along the up directions, 0.5 along the saddle direction
            close         0.05 along the up directions, 1e-8 along the saddle direction
                          (the assignment's (0.05, 1e-8): the gradient is tiny)
            on            exactly the saddle, 0
            near_min      0.05 along the up directions, sqrt(gamma) + 1e-4 along the
                          saddle direction (the assignment's S5 when gamma = 1)
        """
        size, along = {"far": (1.0, 0.5), "close": (0.05, 1e-8), "on": (0.0, 0.0),
                       "near_min": (0.05, np.sqrt(self.gamma) + 1e-4)}[kind]
        u = np.zeros(self.n)
        u[:-1] = size / np.sqrt(self.n - 1)            # spread "size" over the up directions
        u[-1] = along
        return self.Q.T @ u


FUNCTIONS = {"assignment": Benchmark, "star": Star}
