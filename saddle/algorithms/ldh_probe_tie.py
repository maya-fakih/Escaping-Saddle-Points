"""LDH variant: walk to an evaluated probe only when all probe values tie.

This is a separate variant so archived random-kick results remain reproducible.
No new random perturbation is sampled. Ties use a fixed, penalty-ranked order.
"""
from saddle.algorithms.ldh import LDH


class LDHProbeTie(LDH):
    name = 'LDH-GD (probe ties only)'

    def __init__(self, problem, z0, tie_tol=1e-12, **kwargs):
        kwargs['kick'] = False
        self.tie_tol = tie_tol
        super().__init__(problem, z0, **kwargs)

    def setup(self):
        super().setup()
        self.tie_walks = []

    def _search(self):
        if self.fails == 0:
            self.anchor = (self.z.copy(), self.fz)
            self.f_start = self.fz
        candidates = []
        for direction in self._directions().T:
            for sign in (+1, -1):
                v = sign * direction
                point = self.z + self.delta * v
                value = self.problem.f(point)
                change = value - self.f_start
                candidates.append((change + self._penalty_of(v), change, point, value, v))
                if value < self.anchor[1]:
                    self.anchor = (point.copy(), value)
        # Preserve the existing penalty ranking and decrease test.
        _, change, point, value, v = min(candidates, key=lambda item: item[0])
        if change < -self.eps_f:
            self.fails = 0
            self._remember_direction(v)
            self.jumps.append(self.k)
            self.log_round('escaped')
            self._move(point, value)
            return
        self.fails += 1
        if self.fails >= self.R:
            self._give_up()
            return
        values = [candidate[3] for candidate in candidates]
        if max(values) - min(values) <= self.tie_tol:
            # Move to an ALREADY EVALUATED point: no sampled direction or kick trial.
            self._remember_direction(v)
            self.tie_walks.append(self.k)
            self.log_round('walked')
            self._move(point, value)
        else:
            self.log_round('failed')


ALGORITHM = LDHProbeTie
