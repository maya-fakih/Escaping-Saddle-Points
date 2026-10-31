"""
The numbers we report for one run (same definitions as the report's tables).

With f_s = 0 (f at the saddle, for both functions) and f* (f at a minimum), let
Delta = f_s - f*  (how far down the minimum is below the saddle).

  gap q_k        = (f(z_k) - f*) / Delta     1 at the saddle, 0 at a minimum
  escape k_esc   = first k with f(z_k) <= f_s - 0.1*Delta
                   ("we went 10% of the way down": we have left the saddle)
  success k_succ = first k with q_k <= 1e-4 AND ||grad|| <= 1e-5 AND
                   smallest Hessian eigenvalue >= -1e-4
                   ("we are at a real minimum, not just somewhere flat")

These checks use the uncounted peek_* functions, so they do not add to Nf/Ng/NH.

For methods with an escape move (probes, kicks, curvature steps; see Algorithm.log_round),
each try is one "round", numbered 1, 2, 3, ...:
  rounds         = how many rounds in the whole run
  escape round   = how many rounds it took to escape: the rounds done before k_esc
                   (works the same for jumps, kicks and curvature steps)
  stuck round    = round number of the first "gave up" (called it a minimum)
  rounds to min  = how many rounds were done before the point first reached a minimum
"""
import numpy as np


def at_minimum(P, z):
    """True when z passes the success test: gap <= 1e-4, ||grad|| <= 1e-5 and
    no negative curvature. (A small gradient alone is NOT enough: it is small at
    the saddle too.)"""
    q = (P.peek_f(z) - P.fstar) / (0.0 - P.fstar)
    if q > 1e-4 or np.linalg.norm(P.peek_grad(z)) > 1e-5:
        return False
    return np.linalg.eigvalsh(P.peek_hess(z))[0] >= -1e-4


def probe_rounds(log, k_succ=None, k_esc=None):
    """Summarize a round log [(k, outcome), ...]. Round numbers start at 1; None = never.
    Without k_esc (the live window), the escape round is the first "escaped" round."""
    first = lambda outcome: next((i + 1 for i, (_, o) in enumerate(log) if o == outcome), None)
    rounds_before = lambda k_end: None if k_end is None or not log else \
        sum(1 for k, _ in log if k < k_end)
    escape = first("escaped") if k_esc is None else rounds_before(k_esc)
    return dict(rounds=len(log), escape_round=escape, stuck_round=first("gave up"),
                rounds_to_min=rounds_before(k_succ))


def evaluate(algo):
    P = algo.problem
    Z = np.array(algo.path)
    fs, fstar = 0.0, P.fstar
    delta = fs - fstar
    f = np.array([P.peek_f(z) for z in Z])
    q = (f - fstar) / delta

    hit = np.nonzero(f <= fs - 0.1 * delta)[0]
    k_esc = int(hit[0]) if len(hit) else None

    k_succ = next((int(k) for k in np.nonzero(q <= 1e-4)[0] if at_minimum(P, Z[k])), None)

    nf, ng, nh = algo.counts
    # work and time spent up to the first success (None if it never succeeded)
    w = algo.work[k_succ] if k_succ is not None else (None, None, None)
    t_succ = algo.times[k_succ] if k_succ is not None else None
    return dict(updates=algo.k, k_esc=k_esc, k_succ=k_succ, z=Z[-1], f=f[-1], q=q[-1],
                nf=nf, ng=ng, nh=nh, failed_at=algo.failed_at,
                nf_succ=w[0], ng_succ=w[1], nh_succ=w[2], t_succ=t_succ,
                **probe_rounds(algo.probe_log, k_succ, k_esc))
