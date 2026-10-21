"""
Probe, and if no probe is better, walk anywhere and probe again. Otherwise plain GD.

WHY WE HAVE IT
  The axis probes can be unlucky: on the star saddle (FUNC=star) all four axis
  directions go UP, and only the diagonals go down. Instead of being clever about
  which direction to try next, just move: take a short step in ANY direction,
  and probe +-x, +-y again from that new point. The new point sees the saddle
  from a different angle, so sooner or later one probe lands in a valley.
  Once we are out, nothing special is needed: plain gradient descent finishes.

THE STEP
  - Normal mode: one GD step with backtracking (exactly GD).
  - When ||grad f(z)|| <= g_trig (and we have not given up): remember where we are
    (the "anchor" z_a, with f(z_a)) and start a search. Each update of the search:
      1. probe f at the current point +- r along x and along y;
      2. if a probe is LOWER THAN f(z_a): jump there, the search is over,
         back to normal mode (the next GD steps run down the valley);
      3. if not: walk a distance r in a random direction ("any direction") and
         probe again from there on the next update.
    Comparing with f(z_a), not with the current walk point, is what stops us from
    wandering at a minimum: around a minimum nothing is lower than the minimum.
    After max_walks walks with no success: "probably a minimum", go back to z_a
    and stop searching until the gradient is large again.

  While walking, f can go up a little (the walk is a free move, not a descent step).

THE KNOBS
  g_trig    : "gradient is small" threshold
  r         : probe length and walk length
  max_walks : how many walks before we decide it is a minimum
  seed      : the random walk directions; same seed -> same run

COST
  GD's cost + 4 f per probe round. No Hessian.
"""
import numpy as np

from saddle.algorithms.base import Algorithm
from saddle.cases import G_TRIG, R as R_DEFAULT


class ProbeWalk(Algorithm):
    name = "Probe + walk"

    def __init__(self, problem, z0, max_iters=3000, seed=0, g_trig=G_TRIG, r=R_DEFAULT,
                 max_walks=10):
        self.seed, self.g_trig, self.r, self.max_walks = seed, g_trig, r, max_walks
        super().__init__(problem, z0, max_iters)

    def setup(self):
        self.rng = np.random.default_rng(self.seed)
        self.g = self.problem.grad(self.z)
        self.searching, self.stopped = False, False
        self.anchor, self.f_anchor, self.walks = None, None, 0
        self.jumps, self.walk_steps, self.giveups = [], [], []

    # ------------------------------------------------------------------
    def update(self):
        small = np.linalg.norm(self.g) <= self.g_trig
        if not small and not self.searching:
            self.stopped = False                  # left the flat region: allow searching again
        if self.searching or (small and not self.stopped):
            return self._search()
        return self._gd_step()

    def _gd_step(self):
        z_new, f_new = self.backtrack(self.z, self.fz, -self.g, self.g @ self.g)
        if z_new is None:
            return False
        self._move(z_new, f_new)
        return True

    def _move(self, z, fz):
        self.z, self.fz = z, fz
        self.g = self.problem.grad(z)

    # ------------------------------------------------------------------
    def _search(self):
        """One round: probe from the current point; jump, walk, or give up."""
        if not self.searching:                    # start a new search here
            self.searching = True
            self.anchor, self.f_anchor, self.walks = self.z.copy(), self.fz, 0

        best_f, best_z = self.f_anchor, None
        for u in ([1.0, 0.0], [0.0, 1.0]):
            for sign in (+1, -1):
                zp = self.z + sign * self.r * np.array(u)
                fp = self.problem.f(zp)
                if fp < best_f:
                    best_f, best_z = fp, zp

        if best_z is not None:                    # found a way down: done searching
            self.searching = False
            self.jumps.append(self.k)
            self._move(best_z, best_f)
        elif self.walks < self.max_walks:         # nothing better: walk anywhere
            v = self.rng.standard_normal(2)
            z_walk = self.z + self.r * v / np.linalg.norm(v)
            self.walks += 1
            self.walk_steps.append(self.k)
            self._move(z_walk, self.problem.f(z_walk))
        else:                                     # nothing anywhere: probably a minimum
            self.searching, self.stopped = False, True
            self.giveups.append(self.k)
            self._move(self.anchor, self.f_anchor)
        return True


ALGORITHM = ProbeWalk
