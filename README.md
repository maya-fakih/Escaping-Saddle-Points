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
    ├── Makefile                  # make install / make run visual / make run headless / make test
    ├── main.py                   # what make run calls (visual or headless)
    ├── saddle/
    │   ├── benchmark.py          # f, gradient, Hessian, with Nf/Ng/NH counters
    │   ├── cases.py              # default parameters (+ S1-S5, used by the tests)
    │   ├── metrics.py            # escape / success iteration, gap, work for one run
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
| `probe` | earlier variant: learned per-axis step sizes + axis probes | `python -m unittest tests.test_probe` |
| `probe-nesterov` | probes only when the gradient is tiny, then Nesterov momentum | `python -m unittest tests.test_probe_nesterov` |
| `probe-walk` | probes when stuck; if all probes look equal (flat) move to one and look further (probe length doubles); turn the axes before calling it a minimum | `python -m unittest tests.test_probe_walk` |
| `learned-step` | only the per-axis step sizes learned from predicted vs actual drop (no probes) | `python -m unittest tests.test_learned_step` |
| `jin-pgd` | Jin et al. (2017) perturbed GD: kick in a ball, stop if f did not drop by f_thres after t_thres steps | `python -m unittest tests.test_jin_pgd` |
| `proposed` | LDH (Learned Diagonal Hessian): per-dimension step sizes learned from ρ; when stuck, probes along the m most negative directions of a tridiagonal B (measured, or guessed from ρ), capped kick, give-up after R rounds, fading penalty. One class with a switch per part (`proposed.py`), plus `ldh.py` and `ldh_guessed.py` | `python -m unittest tests.test_proposed` |
| `all-methods` | every branch merged, plus the experiment scripts in `experiments/` | all of the above |

## Running

    make install                    # once: venv in ../venv + numpy, matplotlib
    git checkout all-methods        # or any single method branch
    make test                       # the tests on this branch
    make run visual                 # 3D window, pick the method in the window
    make run headless ALG=all       # no window, prints a results table

Settings (all optional): `ALG` (gd, pgd, gdnc, jin_pgd, probe_nesterov, ldh, ldh_guessed, any other method file on the branch, or all), `FUNC` (assignment or star), `GAMMA` (> 0, default 1;
headless takes a list like `GAMMA="1 0.1 0.01"`), `X0`, `Y0` (start, default (0.05, 1e-8)), `ANGLE` (turn the 2D function by this many degrees, default 0), `K` (max updates), `SEED`, `G_TRIG`, `R`, `T`, `SPEED`
(steps per second in the window). Example:

    make run headless ALG=pgd GAMMA=0.01 SEED=4 K=500
    make run headless GAMMA="1 0.1 0.01 0.001"
    make run visual ALG=gdnc GAMMA=0.5 X0=0.3 Y0=0 SPEED=20
    make run headless FUNC=star X0=0 Y0=0

`FUNC=star` is a harder saddle: f = (x^4 - 6x^2y^2 + y^4)/4 + (x^2+y^2)^3/6. At (0, 0) the
gradient and the Hessian are both zero, the four axes are ridges and the valleys are on the
diagonals (minima at distance 1, f* = -1/12). Details in `saddle/benchmark.py`.

In the window: drag to **rotate** the 3D view, scroll to zoom. On the right: the
**Algorithm** picker, **Play/Pause** (runs at the slider speed), **Step** (one update),
**Reset**, and the numbers at the current point: k, z, f, f*, the gradient and its norm,
the Hessian (n/a when the method does not use it) and the work Nf / Ng / NH. The numbers
on screen are computed without counting, so they do not change the work.

In the window the gamma, x0, y0 and angle boxes change the function and the start; press
**Apply**. **Start at (0,0)** puts the start exactly on the saddle.

## Experiments (branch all-methods)

    python experiments/run_hw.py                    # the assignment: S1-S5 (Q1-Q3) + Question 4 on S2
    python experiments/run_hw.py --pgd T=50 --gdnc g_trig=1e-4   # + S1-S5 rows for your Q4 choices
    python experiments/hw_figures.py                # S2 contours, S2 gap vs k, one Q4 plot
    python experiments/run_cases.py                 # the 27 cases below, 7 methods

`run_cases.py`: 2D, 2D turned 45°, 100D with a random rotation × γ = 1, 0.1, 0.01 ×
starts near the saddle, on it, near a minimum. Methods: GD, P-GD, GD-NC, Jin et al. PGD,
Probe + Nesterov, LDH (the proposed method), LDH with a guessed B. Every finished run is
appended to `results/cases_runs.csv` (a stopped run resumes); tables in
`results/cases_results.xlsx` with merged headers (function, then γ, then start).
Everything goes into `results/`.

Headless columns: `k_esc` = first k that is 10% of the way down from the saddle,
`k_succ` = first k at a real minimum (gap <= 1e-4, |grad| <= 1e-5, no negative curvature),
gap q = (f - f*) / (f_saddle - f*). Definitions in `saddle/metrics.py`.

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
## Dimension-dependent LDH-GD search budget

LDH-GD now defaults to `R(n) = max(3, ceil(sqrt(n)))`: 3 failed search
rounds in 2D, 10 in 100D. Explicit constructor overrides such as `R=3`
remain supported. See [the budget rule and reproduction notes](docs/search_budget.md).
Archived experiment tables predate this default change.
