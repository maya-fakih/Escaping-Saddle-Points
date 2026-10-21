"""
The parent class every method inherits from.

The idea: an algorithm is something that sits at a point z and, each time you
call step(), moves z once. The parent class does the bookkeeping (iteration
counter, the path so far, stopping) so each child only writes ONE method,
update(), which says how to get from the current z to the next one.

Because everything moves one step at a time, the simulator can play it,
pause it, or advance it by hand.
"""
import numpy as np

ARMIJO = 1e-4      # sufficient-decrease constant (assignment)
MAX_TRIALS = 40    # backtracking gives up after 40 halvings (assignment)


class Algorithm:
    name = "base"
    uses_hessian = False      # children that use the Hessian set this to True

    def __init__(self, problem, z0, max_iters=3000):
        self.problem = problem
        self.z0 = np.array(z0, dtype=float)
        self.max_iters = max_iters
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        """Go back to the start point and forget everything."""
        self.problem.reset_counts()
        self.z = self.z0.copy()
        self.fz = self.problem.f(self.z)
        self.k = 0                    # number of updates done
        self.path = [self.z.copy()]   # every point visited, z_0 ... z_k
        self.done = False
        self.failed_at = None         # iteration where the line search failed, if ever
        self.setup()

    def setup(self):
        """Children put their own starting state here (seeds, counters, ...)."""

    def update(self):
        """Children: move self.z (and self.fz) once.
        Return True if it worked, False if the line search failed."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    def step(self):
        """Do one update (if we are not finished) and return the new point."""
        if self.done:
            return self.z
        if self.update():
            self.k += 1
            self.path.append(self.z.copy())
            if self.k >= self.max_iters:
                self.done = True
        else:
            self.failed_at = self.k
            self.done = True
        return self.z

    def run(self):
        """Run until max_iters (or a failure). Returns the path as an array."""
        while not self.done:
            self.step()
        return np.array(self.path)

    @property
    def counts(self):
        """Work done so far: (Nf, Ng, NH)."""
        p = self.problem
        return p.nf, p.ng, p.nh

    # ------------------------------------------------------------------
    def backtrack(self, z, fz, direction, slope):
        """Armijo backtracking, exactly the assignment's rule.

        Try t = 1, 1/2, 1/4, ... (at most 40 times) and accept the first t with
            f(z + t*direction) <= f(z) - 1e-4 * t * slope
        For plain GD: direction = -grad and slope = ||grad||^2.

        Returns (new point, f at it), or (None, None) if all 40 tries failed.
        """
        t = 1.0
        for _ in range(MAX_TRIALS):
            z_new = z + t * direction
            f_new = self.problem.f(z_new)
            if f_new <= fz - ARMIJO * t * slope:
                return z_new, f_new
            t *= 0.5
        return None, None
