"""
Plain gradient descent with backtracking (assignment, page 2).

    z_{k+1} = z_k - t * grad f(z_k)

with t chosen by Armijo backtracking (start at 1, halve, at most 40 tries).
Near a saddle the gradient is tiny, so GD crawls; exactly on it, it never moves.
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
