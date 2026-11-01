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
  1. learn_step   step sizes learned from "actual drop / predicted drop" (rho), our
                  first idea. Used in every GD step when on. Two versions:
                    "axes"       one step size per coordinate axis (the first version).
                                 Fine when the function's valleys line up with the
                                 axes; when the function is turned, a step that fits
                                 one axis overshoots across the diagonal valley and
                                 the path zigzags.
                    "curvature"  one step size per eigen-direction of the measured B
                                 (part 2). These directions turn with the function,
                                 so the steps fit the valley in any rotation. Each
                                 step size starts at 1 / |lambda| (the curvature B
                                 measured along it; a direction that curves down gets
                                 a big step, and rho then grows it further), and rho
                                 corrects it from there. B is measured again every
                                 basis_every updates, because the curvature changes
                                 as we move.
                  True means "axes" (kept so the older settings still work).
                  The default is "axes" (the LDH method); "curvature" is kept only as a
                  switch for comparison.
  2. band b       a banded Hessian estimate B, measured from gradient differences
                  (Curtis, Powell & Reid 1974): with band b (only entries |i - j| <= b
                  are kept) it costs min(2b + 1, N) extra gradients, whatever N is.
                  b = 0 keeps only the diagonal, b = N - 1 gives the full matrix.
  3. escape       what to do when the gradient is small ("stuck"):
                    "none"       nothing, plain GD (then it is GD or the learned step)
                    "curvature"  step along B's most negative direction, exactly like
                                 GD-NC but with the MEASURED B instead of the true Hessian
                    "axes"       probe +-delta along every coordinate axis (2N probes)
                    "bag"        probe +-delta along the bag_size directions where the
                                 MEASURED B curves down the most (B's eigenvectors with the
                                 smallest eigenvalues). For a tridiagonal B (band 1) only those
                                 few eigenvectors are computed, with a tridiagonal solver: about
                                 linear in N, the upward directions are never computed.
                    "guessed"    the same probes, but aimed by a GUESSED B that is never
                                 measured: it is learned for free while learning the step sizes
                                 (see _learn_step_sizes). Diagonal: H_i = 2 (1 - rho_i) / d_i.
                                 Off-diagonal (neighbours i, i+1): one extra f per pair, the two
                                 axis moves made together. Directions: for each neighbour pair,
                                 the 2 x 2 block of B and its eigenvectors, solved by hand (a
                                 formula, no solver). Needs learn_step = "axes".
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
        After R = max(3, ceil(sqrt(N))) rounds in a row with no drop (unless overridden): "very likely a minimum" ("gave up"):
        go back to the best point of this search and stop searching until the
        gradient is large again (we have left the flat region).
  - Then one GD step with backtracking (scaled by d if learn_step is on).
  A jump or a kick followed by the GD step counts as ONE update, like P-GD.

COST (all counted in Nf / Ng / NH; no Hessian is ever computed)
  - measuring B: min(2b + 1, N) gradients per stuck round
  - probing: 2 f per direction (2N for "axes", 2 * bag_size for "bag")
  - kick: 1 f per halving of the kick length
  - learned step: 1 f per direction per update; "curvature" also measures B every
    basis_every updates (min(2b + 1, N) gradients)
"""
import numpy as np

from saddle.algorithms.base import Algorithm, ARMIJO, MAX_TRIALS
from saddle.cases import G_TRIG, R as R_DEFAULT

NEG_CURVATURE = -1e-4    # the same "really negative" threshold as GD-NC


class Proposed(Algorithm):
    name = "Proposed"

    def __init__(self, problem, z0, max_iters=3000, seed=0, g_trig=G_TRIG,
                 learn_step="axes", band=1, escape="bag", bag_size=4, kick=True, penalty=True,
                 delta=1e-2, r=R_DEFAULT, R=None, h=1e-5, p0=1e-4, beta=0.9, eps_f=1e-12,
                 kappa=0.5, omega=2.0, d_min=1e-6, d_max=100.0, basis_every=20, L=1.0,
                 best_probe=False):
        # which parts are on
        if learn_step is True:                   # the older setting meant per-axis steps
            learn_step = "axes"
        self.learn_step, self.escape, self.kick, self.penalty = learn_step, escape, kick, penalty
        self.best_probe = best_probe
        self.basis_every = basis_every            # how often B is measured again for the steps
        self.L = L                                # first guess of every curvature (guessed B)
        if escape == "guessed" and learn_step != "axes":
            raise ValueError('escape="guessed" learns B while learning per-axis steps: '
                             'use learn_step="axes"')
        self.band = band                  # None = never measured; "full" or N-1 = full matrix
        self.bag_size = bag_size
        # sizes and thresholds
        # Default failed-search budget grows with dimension; explicit R overrides it.
        rounds = max(3, int(np.ceil(np.sqrt(len(z0))))) if R is None else R
        self.seed, self.g_trig, self.delta, self.r, self.R = seed, g_trig, delta, r, rounds
        self.h, self.p0, self.beta, self.eps_f = h, p0, beta, eps_f
        self.kappa, self.omega, self.d_min, self.d_max = kappa, omega, d_min, d_max
        super().__init__(problem, z0, max_iters)

    # ------------------------------------------------------------------
    def setup(self):
        """Fresh state for a new run."""
        n = len(self.z)
        self.rng = np.random.default_rng(self.seed)   # used only to pick a kick direction
        self.g = self.problem.grad(self.z)
        self.d = np.ones(n)                           # one step size per direction (learn_step)
        self.V = None                                 # the step directions: None = the axes,
                                                      # else the columns are B's eigenvectors
        self.basis_k = None                           # update number when B was last measured
        # the guessed B (escape="guessed"): its diagonal and its one off-diagonal.
        # Before anything is learned every direction is guessed to curve up by L, no coupling.
        self.B_diag = np.full(n, float(self.L))
        self.B_off = np.zeros(max(n - 1, 0))
        self.p = np.zeros(n)                          # per-axis penalties (penalty)
        self.fails = 0                                # search rounds in a row with no drop
        self.anchor = None                            # best point of the current search
        self.f_start = None                           # f where the current search started
        self.stopped = False                          # "very likely a minimum" flag
        self.jumps, self.kicks, self.giveups = [], [], []
        self.walks = []

    # ------------------------------------------------------------------
    def update(self):
        """One update: (search if stuck) then one GD step."""
        self.p *= self.beta                           # old penalties fade a little

        if self.learn_step == "curvature" and (
                self.basis_k is None or self.k - self.basis_k >= self.basis_every):
            self._set_basis(self.measure_curvature())  # the curvature changed: re-measure
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
    # the GD step (plain, or scaled per direction when learn_step is on)
    def _gd_step(self):
        if self.V is None:                            # directions = the axes
            step, slope = -self.d * self.g, self.g @ (self.d * self.g)
        else:
            # write g in B's directions (gb_i = how much of g lies along direction i),
            # scale each part by its own step size, then turn back to x, y, ...
            gb = self.V.T @ self.g
            step, slope = -self.V @ (self.d * gb), gb @ (self.d * gb)
        z_new, f_new = self.backtrack(self.z, self.fz, step, slope)
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

    def _set_basis(self, B):
        """learn_step = "curvature": make B's eigenvectors the step directions.
        Each step size starts at 1 / |lambda|: along a direction that curves up with
        curvature lambda, 1 / lambda lands on the bottom of that parabola (Newton's
        step); along one that curves down, 1 / |lambda| is a big step away from the
        saddle, and rho keeps growing it while f keeps dropping faster than predicted.
        Nearly flat directions (|lambda| tiny) start at the largest allowed step."""
        lam, V = np.linalg.eigh(B)
        self.V = V
        self.d = np.clip(1.0 / np.maximum(np.abs(lam), 1.0 / self.d_max), self.d_min, self.d_max)
        self.basis_k = self.k

    # ------------------------------------------------------------------
    # escape = "curvature": GD-NC, but with the measured B
    def _curvature_step(self):
        B = self.measure_curvature()
        if self.learn_step == "curvature":
            self._set_basis(B)                        # measured anyway: reuse it for the steps
        lam, V = np.linalg.eigh(B)
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
        m = min(self.bag_size, n)
        if self.escape == "axes":
            return np.eye(n)                          # every coordinate axis: 2N probes
        if self.escape == "guessed":
            return self._guessed_directions(m)
        B = self.measure_curvature()
        if self.learn_step == "curvature":
            self._set_basis(B)                        # measured anyway: reuse it for the steps
            return self.V[:, :m]
        return smallest_directions(B, m, self._band_width())

    def _guessed_directions(self, m):
        """The m directions where the GUESSED B curves down the most, without any solver.
        Each neighbour pair (i, i+1) has a 2 x 2 block [[a, b], [b, c]] of B. Its smallest
        eigenvalue and direction have a formula:
            lam = (a + c)/2 - sqrt(((a - c)/2)^2 + b^2),  direction (b, lam - a)
        (or the axis with the smaller of a, c when b = 0). The other eigenvalue is
        (a + c)/2 + sqrt(...) with the perpendicular direction. Rank all of them, keep the m
        most negative, each written as a full N-vector (zero outside its pair)."""
        n = len(self.z)
        a, c, b = self.B_diag[:-1], self.B_diag[1:], self.B_off
        mid, rad = (a + c) / 2, np.sqrt(((a - c) / 2)**2 + b**2)
        found = []                                    # (eigenvalue, direction)
        for i in range(n - 1):
            lo = mid[i] - rad[i]
            if abs(b[i]) > 1e-14:
                u = np.array([b[i], lo - a[i]])
            else:
                u = np.array([1.0, 0.0]) if a[i] <= c[i] else np.array([0.0, 1.0])
            u /= np.linalg.norm(u)
            for lam, w in ((lo, u), (mid[i] + rad[i], np.array([-u[1], u[0]]))):
                v = np.zeros(n)
                v[i:i + 2] = w
                found.append((lam, v))
        found.sort(key=lambda t: t[0])
        dirs, seen = [], []
        for lam, v in found:                          # skip repeats (one axis can come twice)
            if all(abs(v @ s) < 1 - 1e-9 for s in seen):
                dirs.append(v)
                seen.append(v)
            if len(dirs) == m:
                break
        return np.array(dirs).T           # where B curves down the most

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
        elif self.best_probe:
            # Deterministic exploration: reuse the best scored evaluated probe,
            # even uphill. There is no new sampled kick or enclosing trust region.
            self._remember_direction(v)
            self.walks.append(self.k)
            self.log_round("walked")
            self._move(zp, fp)
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
    # part 1: the learned step sizes (same rule as the learned-step method)
    def _learn_step_sizes(self):
        """For each direction v_i (an axis, or one of B's eigenvectors): c_i = g . v_i
        is how much of the gradient lies along it. Predicted drop of the move along v_i
        alone: d_i c_i^2; compare with the actual drop; rho = actual / predicted =
        1 - d_i H_i / 2, where H_i is the curvature along v_i. Move d_i toward 1 / H_i,
        or grow it when the direction curves down (rho >= 1)."""
        n = len(self.z)
        guess = self.escape == "guessed"
        moves, f_moves = np.zeros(n), np.full(n, np.nan)   # each axis move and its f (guessed B)
        for i in range(n):
            c = self.g[i] if self.V is None else self.V[:, i] @ self.g
            pred = self.d[i] * c**2
            if pred < 1e-12:                          # too small to measure: rounding noise
                continue
            if self.V is None:
                zi = self.z.copy()
                zi[i] -= self.d[i] * c
            else:
                zi = self.z - self.d[i] * c * self.V[:, i]
            fi = self.problem.f(zi)
            rho = (self.fz - fi) / pred
            if guess:
                # the same rho tells the curvature along this axis: rho = 1 - d H / 2
                self.B_diag[i] = 2 * (1 - rho) / self.d[i]
                moves[i], f_moves[i] = -self.d[i] * c, fi
            factor = np.clip((1.0 / (2 * (1 - rho)))**self.kappa, 0.1, 10.0) if rho < 1 \
                else self.omega
            self.d[i] = np.clip(self.d[i] * factor, self.d_min, self.d_max)
        if guess:
            self._learn_coupling(moves, f_moves)

    def _learn_coupling(self, w, fw):
        """Guessed B, off-diagonal: make the moves of axes i and i+1 TOGETHER (one extra f).
        Taylor to second order, the plain parts cancel and only the coupling is left:
            f(z + w_i e_i + w_j e_j) - f(z + w_i e_i) - f(z + w_j e_j) + f(z) = w_i w_j H_ij
        (exact when f is quadratic). Skipped when the moves are too small to measure."""
        for i in range(len(self.z) - 1):
            j = i + 1
            if np.isnan(fw[i]) or np.isnan(fw[j]) or abs(w[i] * w[j]) < 1e-12:
                continue
            zij = self.z.copy()
            zij[i] += w[i]
            zij[j] += w[j]
            both = self.problem.f(zij)
            self.B_off[i] = (both - fw[i] - fw[j] + self.fz) / (w[i] * w[j])


def smallest_directions(B, m, band):
    """The m eigenvectors of B with the smallest eigenvalues (where B curves down the most).
    Tridiagonal B (band <= 1): a tridiagonal solver that computes ONLY those m, about
    linear in N; the upward directions are never computed. Wider band: a full solve."""
    n = len(B)
    if band <= 1 and n > 2:
        from scipy.linalg import eigh_tridiagonal
        _, V = eigh_tridiagonal(np.diag(B).copy(), np.diag(B, 1).copy(),
                                select="i", select_range=(0, m - 1))
        return V
    return np.linalg.eigh(B)[1][:, :m]


ALGORITHM = Proposed
