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
"""
import numpy as np


def evaluate(algo):
    P = algo.problem
    Z = np.array(algo.path)
    fs, fstar = 0.0, P.fstar
    delta = fs - fstar
    f = np.array([P.peek_f(z) for z in Z])
    q = (f - fstar) / delta

    hit = np.nonzero(f <= fs - 0.1 * delta)[0]
    k_esc = int(hit[0]) if len(hit) else None

    k_succ = None
    for k in np.nonzero(q <= 1e-4)[0]:
        g = P.peek_grad(Z[k])
        lam_min = np.linalg.eigvalsh(P.peek_hess(Z[k]))[0]
        if np.linalg.norm(g) <= 1e-5 and lam_min >= -1e-4:
            k_succ = int(k)
            break

    nf, ng, nh = algo.counts
    return dict(updates=algo.k, k_esc=k_esc, k_succ=k_succ, z=Z[-1], f=f[-1], q=q[-1],
                nf=nf, ng=ng, nh=nh, failed_at=algo.failed_at)
