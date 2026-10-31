"""
LDH with a GUESSED B: the same method as ldh.py, but B is never measured and no
eigen-solver is ever called.

    B's diagonal      learned for free from the same rho as the step sizes:
                      rho = 1 - d_i H_i / 2  ->  H_i = 2 (1 - rho) / d_i
    B's off-diagonal  one extra f per neighbour pair: the two axis moves made together
    probe directions  each neighbour pair's 2 x 2 block of B, solved by a formula

Everything else (kick, give-up, penalty) is as in LDH. Code in proposed.py.
"""
from saddle.algorithms.proposed import Proposed


class LDHGuessed(Proposed):
    name = "LDH, guessed B"

    def __init__(self, problem, z0, **kwargs):
        kwargs = {"learn_step": "axes", "escape": "guessed", "band": 1, **kwargs}
        super().__init__(problem, z0, **kwargs)


ALGORITHM = LDHGuessed
