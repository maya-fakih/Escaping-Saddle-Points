"""
LDH (Learned Diagonal Hessian), the proposed method as submitted.

    step sizes   one per dimension, learned from rho = actual drop / predicted drop
    when stuck   probe +-delta along the m directions where the MEASURED tridiagonal B
                 curves down the most (only those m eigenvectors are computed)
    no drop      capped uphill kick; after R = max(3, ceil(sqrt(N))) rounds with no drop: very likely a minimum
    penalty      directions already used are tried last (fades every update)

All the work is in proposed.py (one class, a switch per part); this file only picks the
switches, so the window and the scripts can name it.
"""
from saddle.algorithms.proposed import Proposed


class LDH(Proposed):
    name = "LDH"

    def __init__(self, problem, z0, **kwargs):
        kwargs = {"learn_step": "axes", "escape": "bag", "band": 1, **kwargs}
        super().__init__(problem, z0, **kwargs)


ALGORITHM = LDH
