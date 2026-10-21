# Escaping Saddle Points

ENMG 616 Advanced Optimization, Fall 2026.

A study of how three first/second-order methods behave around a saddle point, plus one algorithmic modification of my own.

## The problem

Benchmark function, for γ > 0:

    f_γ(x, y) = ½x² − (γ/2)y² + ¼y⁴

γ controls how strong the negative curvature is at the saddle (0, 0).

## Methods compared

| Method | Idea |
|--------|------|
| GD | Plain gradient descent, backtracking line search |
| P-GD | GD + small random perturbation when the gradient is tiny |
| GD-NC | GD + a step along the negative-curvature direction of the Hessian |
| Proposed | My modification (to be designed after Questions 1-4) |

## Research question

Can my modification improve escape from a saddle? When does it help, when does it fail, and what extra computation does it cost?

## Project stages

1. **Prepare**: analyze the geometry of f_γ (A1) and read the two papers (A2)
2. **Experiment**: Questions 1-4 on settings S1-S5, then develop the proposed method
3. **Explain**: LaTeX report and presentation

## Repository layout

    .
    ├── README.md
    ├── requirements.txt
    ├── simulate.py               # opens the 3D simulator: python simulate.py <method>
    ├── saddle/
    │   ├── benchmark.py          # f, gradient, Hessian, with Nf/Ng/NH counters
    │   ├── cases.py              # S1-S5 and the default parameters
    │   ├── simulator.py          # the 3D window + dashboard
    │   └── algorithms/
    │       ├── base.py           # Algorithm: the parent class (step, run, backtracking)
    │       └── <method>.py       # one child per method, on its own branch
    └── tests/
        ├── test_benchmark.py     # checks the shared pieces (on main)
        └── test_<method>.py      # checks only that branch's method

Every method is a child of `Algorithm` and only writes `update()`: how to go from the
current point to the next one. The parent does the rest (counting iterations, keeping the
path, stopping, Armijo backtracking). The simulator takes any child, so a new idea only
needs a new file.

## Branches

`main` holds the shared pieces. Each method lives on its own branch, made from `main`
and **not merged back**; each branch has a test that checks only its method.

| Branch | Adds | Test |
|--------|------|------|
| `main` | benchmark, base class, simulator | `python -m unittest tests.test_benchmark` |
| `gd` | plain GD with backtracking | `python -m unittest tests.test_gd` |
| `pgd` | P-GD (random kick in a disk of radius r when the gradient is small) | `python -m unittest tests.test_pgd` |
| `gdnc` | GD-NC (step along the most negative Hessian direction) | `python -m unittest tests.test_gdnc` |
| `probe` | my method: learned per-axis step sizes + axis probes | `python -m unittest tests.test_probe` |

## Running

    pip install -r requirements.txt
    git checkout gd                      # or pgd, gdnc, probe
    python -m unittest discover tests    # that branch's test + the shared ones
    python simulate.py gd --case S1      # the 3D simulator

Simulator options: `--case S1..S5`, `--speed <steps per second>`, `--iters <max updates>`,
`--frames 0 10 100` (save PNG pictures at those iterations instead of opening a window).

In the window: **Play/Pause** runs it at the slider's speed, **Step** moves it one update,
**Reset** goes back to the start. The panel shows k, the point, f, f*, the gradient and its
norm, the Hessian (or n/a when the method does not use it), and the work Nf / Ng / NH.
The values shown on screen are not counted as work.

## Experimental setup (fixed)

- K = 3000 updates per run, no early stopping on small gradient
- Randomized methods: 10 runs, seeds 0-9 (same seeds when changing parameters)
- Line search: start at 1, halve on rejection, 40 trials, report failure otherwise
- Counted: N_f, N_g, N_H (including rejected trials); diagnostics excluded

Default parameters: g_trig = 1e-3, r = 1e-2, T = 20.

## Settings

| Case | γ | Start (x0, y0) |
|------|---|----------------|
| S1 | 1 | (0.05, 1e-8) |
| S2 | 0.1 | (0.05, 1e-8) |
| S3 | 0.01 | (0.05, 1e-8) |
| S4 | 1 | (0, 0) |
| S5 | 1 | (0.05, 1 + 1e-4) |

## Metrics

Normalized gap q_k = (f(z_k) − f*) / Δ, with Δ = f_s − f*. Reported: escape iteration, success iteration, time to success, final gap, and work (N_f / N_g / N_H).

## Deadlines

- **October 30, 2026**: report + code
- **November 4, 2026**: presentation and discussion

## References

- C. Jin et al., *How to Escape Saddle Points Efficiently*, ICML 2017
- J. D. Lee et al., *Gradient Descent Only Converges to Minimizers*, COLT 2016