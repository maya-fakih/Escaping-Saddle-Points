"""
Probe; if every probe looks the same (flat), step anywhere and look further.
Otherwise plain GD.

WHY WE HAVE IT
  A probe compares f(z + delta u) with f(z). Three things can happen:
    - some probe is LOWER          -> there is a way down: take it.
    - all probes are clearly HIGHER -> we are in a bowl: a minimum, stop.
    - all probes are about EQUAL   -> we cannot tell yet: around here f is so flat
      (e.g. the star saddle, where f changes like r^4) that at this distance the
      change is below what we can trust.
  In the third case we do not do math to find a direction. We just check:
  move to any of the probe points and look again, a bit further each time
  (delta doubles). That is a trust region decided by measurement: we keep
  growing the look-around distance until f really changes, and the measured
  change tells us which case we are in. Once we are out, plain GD finishes.

THE STEP
  - Normal mode: one GD step with backtracking (exactly GD).
  - When ||grad f(z)|| <= g_trig (and we have not given up): remember the point
    (the "anchor" z_a, with f(z_a)) and search. Each update of the search:
      1. probe f at the current point +- delta along x and along y;
         each probe is LOWER if f_probe < f(z_a) - eps_f,
                        HIGHER if f_probe > f(z_a) + eps_f, else EQUAL;
      2. any LOWER: jump to the lowest probe, search over, delta back to r,
         normal GD from there;
      3. else any EQUAL (flat, cannot tell): move to a random EQUAL probe point
         and double delta (up to delta_max), then probe again next update;
      4. else (all HIGHER): maybe a minimum, maybe our 4 directions just missed
         a narrow valley (the star: every axis is a ridge). So turn the probe
         axes by pi/(2R) and probe again from the same point; only after R turns
         that are all HIGHER: a minimum. Go back to z_a, stop searching until the
         gradient is large again.
    Comparing with f(z_a) (not with the current point) is what stops us from
    drifting: we only leave for good if we find something lower than where we got stuck.
    max_walks caps the number of flat moves (then: treat as a minimum).

THE KNOBS
  g_trig    : "gradient is small" threshold
  r         : first probe length (then it doubles while flat)
  eps_f     : how big a change in f must be to count (smaller = "equal")
  delta_max : largest probe length
  max_walks : most flat moves in one search
  R         : turns of the probe axes before we call it a minimum
  seed      : which EQUAL probe point we move to; same seed -> same run

COST
  GD's cost + 4 f per probe round. No Hessian.
"""
import numpy as np

from saddle.algorithms.base import Algorithm
from saddle.cases import G_TRIG, R as R_DEFAULT


class ProbeWalk(Algorithm):
    name = "Probe + walk"

    def __init__(self, problem, z0, max_iters=3000, seed=0, g_trig=G_TRIG, r=R_DEFAULT,
                 eps_f=1e-6, delta_max=0.5, max_walks=20, R=3):
        self.seed, self.g_trig, self.r = seed, g_trig, r
        self.eps_f, self.delta_max, self.max_walks, self.R = eps_f, delta_max, max_walks, R
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
        """One round: probe from the current point, then jump, walk, or stop."""
        if not self.searching:                    # start a new search here
            self.searching = True
            self.anchor, self.f_anchor, self.walks = self.z.copy(), self.fz, 0
            self.delta = self.r
            self.turns = 0                        # how many times the axes were turned

        ang = self.turns * np.pi / (2 * self.R)
        c, s = np.cos(ang), np.sin(ang)
        axes = ([c, s], [-s, c])                  # x and y axes turned by ang

        lower, equal = [], []
        for u in axes:
            for sign in (+1, -1):
                zp = self.z + sign * self.delta * np.array(u)
                fp = self.problem.f(zp)
                if fp < self.f_anchor - self.eps_f:
                    lower.append((fp, zp))
                elif fp <= self.f_anchor + self.eps_f:
                    equal.append((fp, zp))

        if lower:                                 # a way down: take the lowest, done
            fp, zp = min(lower, key=lambda pair: pair[0])
            self.searching = False
            self.jumps.append(self.k)
            self._move(zp, fp)
        elif equal and self.walks < self.max_walks:
            # flat: we cannot tell at this distance. Go to any equal point, look further.
            fp, zp = equal[self.rng.integers(len(equal))]
            self.walks += 1
            self.walk_steps.append(self.k)
            self.delta = min(2 * self.delta, self.delta_max)
            self._move(zp, fp)
        elif not equal and self.turns + 1 < self.R:
            self.turns += 1                       # all higher: turn the axes, look again (stay here)
        else:                                     # all higher after R turns: a minimum
            self.searching, self.stopped = False, True
            self.giveups.append(self.k)
            self._move(self.anchor, self.f_anchor)
        return True


ALGORITHM = ProbeWalk
