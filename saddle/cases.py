"""The five settings from the assignment (page 4): name -> (gamma, start point)."""

CASES = {
    "S1": (1.0, (0.05, 1e-8)),        # strong negative curvature, start next to the saddle
    "S2": (0.1, (0.05, 1e-8)),        # weaker curvature
    "S3": (0.01, (0.05, 1e-8)),       # very weak curvature: the hard one
    "S4": (1.0, (0.0, 0.0)),          # start exactly ON the saddle
    "S5": (1.0, (0.05, 1 + 1e-4)),    # start next to a minimum (should NOT run away)
}

# defaults used by every method unless you change them (assignment page 3)
K = 3000          # number of updates per run
G_TRIG = 1e-3     # "gradient is small" threshold
R = 1e-2          # P-GD perturbation radius
T = 20            # P-GD: wait at least T updates between two kicks
