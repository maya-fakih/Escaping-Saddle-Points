"""
The proposed method: measured (banded) curvature + a guided bag of probe directions
+ a capped uphill kick + a fading penalty on directions already used. GD otherwise.

THE STORY (why each part exists)
  - GD stalls at a saddle because its model of f is a plane (or the descent-lemma
    parabola that curves UP in every direction): it never sees a direction that
    curves DOWN. Near the saddle the gradient is tiny, so GD crawls; exactly on it GD
    never moves.
  - GD-NC fixes this with the exact Hessian and its eigenvectors, but a full Hessian
    is expensive in many dimensions. P-GD fixes it with a random kick, but the kick is
    blind: in many dimensions most random directions barely point downhill.
  - Our idea: MEASURE the curvature cheaply, use it to pick a few promising directions
    (the "bag"), CHECK them with real function values, and only if nothing helps,
    fall back to a kick, which is the paper's method. So the worst case is the paper,
    and every bit of curvature we measure can only narrow the search.

THE PARTS (each one can be switched on or off, so we can test what each part buys)
  1. learn_step   per-axis step sizes learned from "actual drop / predicted drop"
                  (rho), our first idea. Used in every GD step when on.
  2. band b       a banded Hessian estimate B, measured from gradient differences
                  (Curtis, Powell & Reid 1974): with band b (only entries |i - j| <= b
                  are kept) it costs min(2b + 1, N) extra gradients, whatever N is.
                  b = 0 keeps only the diagonal, b = N - 1 gives the full matrix.
  3. escape       what to do when the gradient is small ("stuck"):
                    "none"       nothing, plain GD (then it is GD or the learned step)
                    "curvature"  step along B's most negative direction, exactly like
                                 GD-NC but with the MEASURED B instead of the true Hessian
                    "axes"       probe +-delta along every coordinate axis (2N probes)
                    "bag"        probe +-delta along the bag_size directions where B curves
                                 down the most (B's eigenvectors with the smallest eigenvalues)
  4. kick         if every probe is worse, do not just give up: take one direction from
                  the bag and step UPHILL a little, capped inside a ball of radius r
                  (the same radius as the paper's perturbation) and capped in how much f
                  may rise. From the new point the next probes see the saddle from a
                  different side.
  5. penalty      one number p_j per coordinate. When we leave a stuck point along a
                  direction v, p_j grows by p0 * v_j^2 (how much of v lies along axis j).
                  Every update all p_j shrink by the factor beta, so old choices are
                  forgiven. Probes are ranked by  (change in f) + (penalty of the direction),
                  so a direction we just used (and that GD may drag us back from) is
                  tried last, and the search explores something new.

ONE UPDATE
  - Shrink the penalties: p <- beta * p.
  - If learn_step: update the per-axis step sizes d_i (see _learn_step_sizes).
  - If the gradient is small, we are allowed to search, and escape != "none":
      * "curvature": measure B; if its smallest eigenvalue lam < -1e-4, step along the
        matching eigenvector with GD-NC's line search; otherwise call it a minimum.
      * probing ("axes" or "bag"): probe +-delta along each direction, score each
        probe by its change in f plus its penalty, take the best.
          - best change in f is a real drop (below f where we got stuck)
                                             -> jump there ("escaped")
          - nothing drops, kick is on        -> kick uphill along one bag direction
          - nothing drops, kick is off       -> count a failed round
        After R rounds in a row with no drop: "very likely a minimum" ("gave up"):
        go back to the best point of this search and stop searching until the
        gradient is large again (we have left the flat region).
  - Then one GD step with backtracking (scaled by d if learn_step is on).
  A jump or a kick followed by the GD step counts as ONE update, like P-GD.

COST (all counted in Nf / Ng / NH; no Hessian is ever computed)
  - measuring B: min(2b + 1, N) gradients per stuck round
  - probing: 2 f per direction (2N for "axes", 2 * bag_size for "bag")
  - kick: 1 f per halving of the kick length
  - learned step: 1 f per axis per update
"""
import numpy as np

from saddle.algorithms.base import Algorithm, ARMIJO, MAX_TRIALS
from saddle.cases import G_TRIG, R as R_DEFAULT

NEG_CURVATURE = -1e-4    # the same "really negative" threshold as GD-NC


class Proposed(Algorithm):
    name = "Proposed"

    def __init__(self, problem, z0, max_iters=3000, seed=0, g_trig=G_TRIG,
                 learn_step=False, band=1, escape="bag", bag_size=4, kick=True, penalty=True,
                 delta=1e-2, r=R_DEFAULT, R=3, h=1e-5, p0=1e-4, beta=0.9, eps_f=1e-12,
                 kappa=0.5, omega=2.0, d_min=1e-6, d_max=100.0):
        # which parts are on
        self.learn_step, self.escape, self.kick, self.penalty = learn_step, escape, kick, penalty
        self.band = band                  # None = never measured; "full" or N-1 = full matrix
        self.bag_size = bag_size
        # sizes and thresholds
        self.seed, self.g_trig, self.delta, self.r, self.R = seed, g_trig, delta, r, R
        self.h, self.p0, self.beta, self.eps_f = h, p0, beta, eps_f
        self.kappa, self.omega, self.d_min, self.d_max = kappa, omega, d_min, d_max
        super().__init__(problem, z0, max_iters)

    # ------------------------------------------------------------------
    def setup(self):
        """Fresh state for a new run."""
        n = len(self.z)
        self.rng = np.random.default_rng(self.seed)   # used only to pick a kick direction
        self.g = self.problem.grad(self.z)
        self.d = np.ones(n)                           # per-axis step sizes (learn_step)
        self.p = np.zeros(n)                          # per-axis penalties (penalty)
        self.fails = 0                                # search rounds in a row with no drop
        self.anchor = None                            # best point of the current search
        self.f_start = None                           # f where the current search started
        self.stopped = False                          # "very likely a minimum" flag
        self.jumps, self.kicks, self.giveups = [], [], []

    # ------------------------------------------------------------------
    def update(self):
        """One update: (search if stuck) then one GD step."""
        self.p *= self.beta                           # old penalties fade a little

        if self.learn_step:
            self._learn_step_sizes()

        # leave the "probably a minimum" state only once we have left the flat region
        # (the gradient got big again): sitting at a minimum, we never search again
        stuck = np.linalg.norm(self.g) <= self.g_trig
        if self.stopped and not stuck:
            self.stopped = False
        if stuck and not self.stopped and self.escape != "none":
            if self.escape == "curvature":
                return self._curvature_step()
            self._search()

        return self._gd_step()

    # ------------------------------------------------------------------
    # the GD step (plain, or scaled per axis when learn_step is on)
    def _gd_step(self):
        step = -self.d * self.g
        z_new, f_new = self.backtrack(self.z, self.fz, step, self.g @ (self.d * self.g))
        if z_new is None:
            return False
        self._move(z_new, f_new)
        return True

    def _move(self, z, fz):
        """Go to z (whose f we already know) and get the gradient there."""
        self.z, self.fz = z, fz
        self.g = self.problem.grad(z)

    # ------------------------------------------------------------------
    # part 2: the banded Hessian, measured
    def _band_width(self):
        n = len(self.z)
        return n - 1 if self.band in ("full", None) else min(int(self.band), n - 1)

    def measure_curvature(self):
        """Banded estimate B of the Hessian at z, from gradient differences.

        HOW (Curtis, Powell & Reid 1974): H times a small step s is about
        grad(z + s) - grad(z). If B only has entries within b of the diagonal, then
        columns j and j' that are at least 2b + 1 apart never touch the same row, so
        we can push ALL of them at once with one step and still read each column off
        separately. That needs only G = min(2b + 1, N) gradient differences.
        With b = 0 it is one gradient difference (it assumes no coupling at all);
        with b = N - 1 it is N differences: the full matrix.
        """
        n, b = len(self.z), self._band_width()
        groups = min(2 * b + 1, n)
        B = np.zeros((n, n))
        for c in range(groups):
            cols = np.arange(c, n, groups)            # columns pushed together
            s = np.zeros(n)
            s[cols] = self.h
            y = (self.problem.grad(self.z + s) - self.g) / self.h   # about H @ (s / h)
            for j in cols:                            # read each column inside its band
                lo, hi = max(0, j - b), min(n, j + b + 1)
                B[lo:hi, j] = y[lo:hi]
        return 0.5 * (B + B.T)                        # a Hessian is symmetric

    # ------------------------------------------------------------------
    # escape = "curvature": GD-NC, but with the measured B
    def _curvature_step(self):
        lam, V = np.linalg.eigh(self.measure_curvature())
        lam, v = lam[0], V[:, 0]                      # smallest eigenvalue and its direction
        if lam >= NEG_CURVATURE:
            # B sees no direction that curves down: call it a minimum, take the GD step
            self._give_up()
            return self._gd_step()
        p = -v if self.g @ v > 0 else v               # the downhill sign (+v if g . v = 0)
        t = 1.0
        for _ in range(MAX_TRIALS):                   # GD-NC's own line search
            z_try = self.z + t * p
            f_try = self.problem.f(z_try)
            if f_try <= self.fz + ARMIJO * t**2 * lam:
                self._remember_direction(p)
                self.jumps.append(self.k)
                self.log_round("escaped")
                self._move(z_try, f_try)
                return True
            t *= 0.5
        return False

    # ------------------------------------------------------------------
    # escape = "axes" or "bag": probe, then jump / kick / count a failure
    def _directions(self):
        """The directions to probe, one per column."""
        n = len(self.z)
        if self.escape == "axes":
            return np.eye(n)                          # every coordinate axis: 2N probes
        lam, V = np.linalg.eigh(self.measure_curvature())
        return V[:, :min(self.bag_size, n)]           # where B curves down the most

    def _penalty_of(self, v):
        """How much we already used direction v: sum of p_j weighted by v_j^2."""
        return float(self.p @ v**2) if self.penalty else 0.0

    def _search(self):
        """One search round from the current point."""
        if self.fails == 0:                           # a new search starts here
            self.anchor = (self.z.copy(), self.fz)    # best point seen in this search
            self.f_start = self.fz                    # f where we got stuck
        dirs = self._directions()

        # probe +-delta along each direction; score = change in f + penalty.
        # The change is measured from f where we got STUCK (f_start), not from the
        # current point: after an uphill kick almost anything looks "lower" than the
        # kicked point, but only a point lower than where we got stuck is real progress.
        best = None                                   # (score, change in f, point, direction)
        for v in dirs.T:
            for sign in (+1, -1):
                zp = self.z + sign * self.delta * v
                fp = self.problem.f(zp)
                change = fp - self.f_start
                score = change + self._penalty_of(v)
                if best is None or score < best[0]:
                    best = (score, change, zp, fp, sign * v)
                if fp < self.anchor[1]:               # keep the best point seen in this search
                    self.anchor = (zp.copy(), fp)

        _, change, zp, fp, v = best
        if change < -self.eps_f:                      # a real drop: take it
            self.fails = 0
            self._remember_direction(v)
            self.jumps.append(self.k)
            self.log_round("escaped")
            self._move(zp, fp)
            return

        self.fails += 1                               # nothing went down this round
        if self.fails >= self.R:
            self._give_up()
        elif self.kick:
            self._kick(dirs)
        else:
            self.log_round("failed")

    def _kick(self, dirs):
        """Every probe was worse: step a little UPHILL along one bag direction.
        The step is at most r long (inside the paper's ball) and f may rise by at most
        r^2; if it would rise more, the step is halved. From the new point the next
        probes see the saddle from another side."""
        if self.penalty:                              # least-used direction, ties at random
            pens = np.array([self._penalty_of(v) for v in dirs.T])
            choices = np.flatnonzero(pens <= pens.min() + 1e-15)
        else:
            choices = np.arange(dirs.shape[1])
        v = dirs[:, self.rng.choice(choices)] * self.rng.choice([-1.0, 1.0])
        t = self.r
        for _ in range(MAX_TRIALS):
            zk = self.z + t * v
            fk = self.problem.f(zk)
            if fk - self.fz <= self.r**2:             # cap on how much f may rise
                break
            t *= 0.5
        self._remember_direction(v)
        self.kicks.append(self.k)
        self.log_round("kicked")
        self._move(zk, fk)

    def _give_up(self):
        """R rounds and nothing went down: very likely a minimum. Go back to the best
        point of this search and stop searching until the gradient is large again."""
        self.giveups.append(self.k)
        self.log_round("gave up")
        if self.anchor is not None and self.anchor[1] < self.fz:
            self._move(*self.anchor)
        self.stopped, self.fails, self.anchor = True, 0, None

    def _remember_direction(self, v):
        """Part 5: we left a stuck point along v, so make v expensive to pick again."""
        if self.penalty:
            v = v / np.linalg.norm(v)
            self.p = np.maximum(self.p, self.p0 * v**2)

    # ------------------------------------------------------------------
    # part 1: the learned per-axis step sizes (same rule as the learned-step method)
    def _learn_step_sizes(self):
        """For each axis: predicted drop d_i g_i^2 vs the actual drop of the move along
        that axis alone; rho = actual / predicted = 1 - d_i H_i / 2. Move d_i toward
        1 / H_i, or grow it when the axis curves down (rho >= 1)."""
        for i in range(len(self.z)):
            pred = self.d[i] * self.g[i]**2
            if pred < 1e-12:                          # too small to measure: rounding noise
                continue
            zi = self.z.copy()
            zi[i] -= self.d[i] * self.g[i]
            rho = (self.fz - self.problem.f(zi)) / pred
            factor = np.clip((1.0 / (2 * (1 - rho)))**self.kappa, 0.1, 10.0) if rho < 1 \
                else self.omega
            self.d[i] = np.clip(self.d[i] * factor, self.d_min, self.d_max)


ALGORITHM = Proposed
