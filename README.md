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

## The methods, one by one

Every picture is one run, drawn by `experiments/run_cases.py` (randomized methods: seed 0). Left: the contours of f, the path, × where the method perturbed or probed. Right: the method, its parameters and how it did. Numbers over all seeds are in `results/cases_results.xlsx`.

### 1. Gradient descent (GD)

*Assignment baseline. Backtracking line search from t = 1.*

**Idea.** Step against the gradient. GD only sees the slope, so near the saddle, where the slope is tiny, it crawls, and exactly on the saddle it never moves.

**Parameters.** Nothing to tune (backtracking: start 1, halve, Armijo 1e-4, 40 trials).

**What we see.** TODO

<details><summary>plain 2D function: 9 runs</summary>

| γ | near the saddle | on the saddle | near a minimum |
|---|---|---|---|
| 1 | ![](docs/runs/2d/gamma1_close_gd.png) | ![](docs/runs/2d/gamma1_on_gd.png) | ![](docs/runs/2d/gamma1_near_min_gd.png) |
| 0.1 | ![](docs/runs/2d/gamma0.1_close_gd.png) | ![](docs/runs/2d/gamma0.1_on_gd.png) | ![](docs/runs/2d/gamma0.1_near_min_gd.png) |
| 0.01 | ![](docs/runs/2d/gamma0.01_close_gd.png) | ![](docs/runs/2d/gamma0.01_on_gd.png) | ![](docs/runs/2d/gamma0.01_near_min_gd.png) |

</details>

<details><summary>2D function turned 45°: 9 runs</summary>

| γ | near the saddle | on the saddle | near a minimum |
|---|---|---|---|
| 1 | ![](docs/runs/2d_rot/gamma1_close_gd.png) | ![](docs/runs/2d_rot/gamma1_on_gd.png) | ![](docs/runs/2d_rot/gamma1_near_min_gd.png) |
| 0.1 | ![](docs/runs/2d_rot/gamma0.1_close_gd.png) | ![](docs/runs/2d_rot/gamma0.1_on_gd.png) | ![](docs/runs/2d_rot/gamma0.1_near_min_gd.png) |
| 0.01 | ![](docs/runs/2d_rot/gamma0.01_close_gd.png) | ![](docs/runs/2d_rot/gamma0.01_on_gd.png) | ![](docs/runs/2d_rot/gamma0.01_near_min_gd.png) |

</details>

### 2. Perturbed GD (P-GD)

*Assignment baseline.*

**Idea.** When the gradient is small (below g_trig), kick the point randomly inside a ball of radius r, at most once every T updates, then continue with GD. Part of the kick points downhill and GD grows it. The kick is blind: it does not know which way is down.

**Parameters.** g_trig = 1e-3, r = 1e-2, T = 20 (defaults; Question 4 varies each one on S2).

**What we see.** TODO

<details><summary>plain 2D function: 9 runs</summary>

| γ | near the saddle | on the saddle | near a minimum |
|---|---|---|---|
| 1 | ![](docs/runs/2d/gamma1_close_pgd.png) | ![](docs/runs/2d/gamma1_on_pgd.png) | ![](docs/runs/2d/gamma1_near_min_pgd.png) |
| 0.1 | ![](docs/runs/2d/gamma0.1_close_pgd.png) | ![](docs/runs/2d/gamma0.1_on_pgd.png) | ![](docs/runs/2d/gamma0.1_near_min_pgd.png) |
| 0.01 | ![](docs/runs/2d/gamma0.01_close_pgd.png) | ![](docs/runs/2d/gamma0.01_on_pgd.png) | ![](docs/runs/2d/gamma0.01_near_min_pgd.png) |

</details>

<details><summary>2D function turned 45°: 9 runs</summary>

| γ | near the saddle | on the saddle | near a minimum |
|---|---|---|---|
| 1 | ![](docs/runs/2d_rot/gamma1_close_pgd.png) | ![](docs/runs/2d_rot/gamma1_on_pgd.png) | ![](docs/runs/2d_rot/gamma1_near_min_pgd.png) |
| 0.1 | ![](docs/runs/2d_rot/gamma0.1_close_pgd.png) | ![](docs/runs/2d_rot/gamma0.1_on_pgd.png) | ![](docs/runs/2d_rot/gamma0.1_near_min_pgd.png) |
| 0.01 | ![](docs/runs/2d_rot/gamma0.01_close_pgd.png) | ![](docs/runs/2d_rot/gamma0.01_on_pgd.png) | ![](docs/runs/2d_rot/gamma0.01_near_min_pgd.png) |

</details>

### 3. GD with negative curvature (GD-NC)

*Assignment baseline.*

**Idea.** When the gradient is small, compute the full Hessian and step along its most negative eigenvector, the direction that curves down the most. Exact, but the Hessian costs N² numbers.

**Parameters.** g_trig = 1e-3 (default; Question 4 varies it on S2).

**What we see.** TODO

<details><summary>plain 2D function: 9 runs</summary>

| γ | near the saddle | on the saddle | near a minimum |
|---|---|---|---|
| 1 | ![](docs/runs/2d/gamma1_close_gdnc.png) | ![](docs/runs/2d/gamma1_on_gdnc.png) | ![](docs/runs/2d/gamma1_near_min_gdnc.png) |
| 0.1 | ![](docs/runs/2d/gamma0.1_close_gdnc.png) | ![](docs/runs/2d/gamma0.1_on_gdnc.png) | ![](docs/runs/2d/gamma0.1_near_min_gdnc.png) |
| 0.01 | ![](docs/runs/2d/gamma0.01_close_gdnc.png) | ![](docs/runs/2d/gamma0.01_on_gdnc.png) | ![](docs/runs/2d/gamma0.01_near_min_gdnc.png) |

</details>

<details><summary>2D function turned 45°: 9 runs</summary>

| γ | near the saddle | on the saddle | near a minimum |
|---|---|---|---|
| 1 | ![](docs/runs/2d_rot/gamma1_close_gdnc.png) | ![](docs/runs/2d_rot/gamma1_on_gdnc.png) | ![](docs/runs/2d_rot/gamma1_near_min_gdnc.png) |
| 0.1 | ![](docs/runs/2d_rot/gamma0.1_close_gdnc.png) | ![](docs/runs/2d_rot/gamma0.1_on_gdnc.png) | ![](docs/runs/2d_rot/gamma0.1_near_min_gdnc.png) |
| 0.01 | ![](docs/runs/2d_rot/gamma0.01_close_gdnc.png) | ![](docs/runs/2d_rot/gamma0.01_on_gdnc.png) | ![](docs/runs/2d_rot/gamma0.01_near_min_gdnc.png) |

</details>

### 4. Perturbed GD, Jin et al. 2017 (Algorithm 2)

*From the paper we read.*

**Idea.** P-GD plus a stop rule: after a kick, if f has not dropped by f_thres within t_thres updates, call the point a minimum and stop. When γ is small the drop after a kick is tiny, so this rule can stop at the saddle.

**Parameters.** r = 1e-2, g_thres = 1e-3, t_thres = 20, f_thres = 1e-7.

**What we see.** TODO

<details><summary>plain 2D function: 9 runs</summary>

| γ | near the saddle | on the saddle | near a minimum |
|---|---|---|---|
| 1 | ![](docs/runs/2d/gamma1_close_jin_pgd.png) | ![](docs/runs/2d/gamma1_on_jin_pgd.png) | ![](docs/runs/2d/gamma1_near_min_jin_pgd.png) |
| 0.1 | ![](docs/runs/2d/gamma0.1_close_jin_pgd.png) | ![](docs/runs/2d/gamma0.1_on_jin_pgd.png) | ![](docs/runs/2d/gamma0.1_near_min_jin_pgd.png) |
| 0.01 | ![](docs/runs/2d/gamma0.01_close_jin_pgd.png) | ![](docs/runs/2d/gamma0.01_on_jin_pgd.png) | ![](docs/runs/2d/gamma0.01_near_min_jin_pgd.png) |

</details>

<details><summary>2D function turned 45°: 9 runs</summary>

| γ | near the saddle | on the saddle | near a minimum |
|---|---|---|---|
| 1 | ![](docs/runs/2d_rot/gamma1_close_jin_pgd.png) | ![](docs/runs/2d_rot/gamma1_on_jin_pgd.png) | ![](docs/runs/2d_rot/gamma1_near_min_jin_pgd.png) |
| 0.1 | ![](docs/runs/2d_rot/gamma0.1_close_jin_pgd.png) | ![](docs/runs/2d_rot/gamma0.1_on_jin_pgd.png) | ![](docs/runs/2d_rot/gamma0.1_near_min_jin_pgd.png) |
| 0.01 | ![](docs/runs/2d_rot/gamma0.01_close_jin_pgd.png) | ![](docs/runs/2d_rot/gamma0.01_on_jin_pgd.png) | ![](docs/runs/2d_rot/gamma0.01_near_min_jin_pgd.png) |

</details>

### 5. Probe + Nesterov (earlier variant)

*An earlier version of the proposed idea.*

**Idea.** When stuck, probe along the axes and jump to a lower probe; otherwise move with Nesterov momentum (restarted when f goes up).

**Parameters.** Probe length δ = 1e-2, R = 3 rounds.

**What we see.** TODO

<details><summary>plain 2D function: 9 runs</summary>

| γ | near the saddle | on the saddle | near a minimum |
|---|---|---|---|
| 1 | ![](docs/runs/2d/gamma1_close_probe_nesterov.png) | ![](docs/runs/2d/gamma1_on_probe_nesterov.png) | ![](docs/runs/2d/gamma1_near_min_probe_nesterov.png) |
| 0.1 | ![](docs/runs/2d/gamma0.1_close_probe_nesterov.png) | ![](docs/runs/2d/gamma0.1_on_probe_nesterov.png) | ![](docs/runs/2d/gamma0.1_near_min_probe_nesterov.png) |
| 0.01 | ![](docs/runs/2d/gamma0.01_close_probe_nesterov.png) | ![](docs/runs/2d/gamma0.01_on_probe_nesterov.png) | ![](docs/runs/2d/gamma0.01_near_min_probe_nesterov.png) |

</details>

<details><summary>2D function turned 45°: 9 runs</summary>

| γ | near the saddle | on the saddle | near a minimum |
|---|---|---|---|
| 1 | ![](docs/runs/2d_rot/gamma1_close_probe_nesterov.png) | ![](docs/runs/2d_rot/gamma1_on_probe_nesterov.png) | ![](docs/runs/2d_rot/gamma1_near_min_probe_nesterov.png) |
| 0.1 | ![](docs/runs/2d_rot/gamma0.1_close_probe_nesterov.png) | ![](docs/runs/2d_rot/gamma0.1_on_probe_nesterov.png) | ![](docs/runs/2d_rot/gamma0.1_near_min_probe_nesterov.png) |
| 0.01 | ![](docs/runs/2d_rot/gamma0.01_close_probe_nesterov.png) | ![](docs/runs/2d_rot/gamma0.01_on_probe_nesterov.png) | ![](docs/runs/2d_rot/gamma0.01_near_min_probe_nesterov.png) |

</details>

### 6. LDH: Learned Diagonal Hessian (the proposed method)

*The proposed method.*

**Idea.** Two ideas. (1) Each dimension gets its own step size, learned from ρ = actual drop / predicted drop: ρ = 1 − d·H/2, so ρ tells whether the step was too short or too long, and grows the step when a direction curves down. (2) When stuck, do not search all 2N directions: build a tridiagonal approximation B of the Hessian (3 gradients) and probe ±δ only along its m most negative directions (only those m are computed). Take a probe that really lowers f; if none does, a small capped uphill kick; after R rounds with no drop, call it a minimum. A fading penalty makes directions already used the last to be tried.

**Parameters.** m = 4, δ = 1e-2, kick r = 1e-2 (f may rise at most r²), R = 3, κ = 0.5, ω = 2, penalty p0 = 1e-4 fading by β = 0.9.

**What we see.** TODO

<details><summary>plain 2D function: 9 runs</summary>

| γ | near the saddle | on the saddle | near a minimum |
|---|---|---|---|
| 1 | ![](docs/runs/2d/gamma1_close_ldh.png) | ![](docs/runs/2d/gamma1_on_ldh.png) | ![](docs/runs/2d/gamma1_near_min_ldh.png) |
| 0.1 | ![](docs/runs/2d/gamma0.1_close_ldh.png) | ![](docs/runs/2d/gamma0.1_on_ldh.png) | ![](docs/runs/2d/gamma0.1_near_min_ldh.png) |
| 0.01 | ![](docs/runs/2d/gamma0.01_close_ldh.png) | ![](docs/runs/2d/gamma0.01_on_ldh.png) | ![](docs/runs/2d/gamma0.01_near_min_ldh.png) |

</details>

<details><summary>2D function turned 45°: 9 runs</summary>

| γ | near the saddle | on the saddle | near a minimum |
|---|---|---|---|
| 1 | ![](docs/runs/2d_rot/gamma1_close_ldh.png) | ![](docs/runs/2d_rot/gamma1_on_ldh.png) | ![](docs/runs/2d_rot/gamma1_near_min_ldh.png) |
| 0.1 | ![](docs/runs/2d_rot/gamma0.1_close_ldh.png) | ![](docs/runs/2d_rot/gamma0.1_on_ldh.png) | ![](docs/runs/2d_rot/gamma0.1_near_min_ldh.png) |
| 0.01 | ![](docs/runs/2d_rot/gamma0.01_close_ldh.png) | ![](docs/runs/2d_rot/gamma0.01_on_ldh.png) | ![](docs/runs/2d_rot/gamma0.01_near_min_ldh.png) |

</details>

### 7. LDH with a guessed B

*The proposed method, no solver.*

**Idea.** Same as LDH, but B is never measured. Its diagonal comes for free from the same ρ (H_i = 2(1 − ρ)/d_i); each off-diagonal costs one extra f (the two neighbouring axis moves made together); the probe directions come from each neighbour pair's 2 × 2 block, solved by a formula.

**Parameters.** As LDH; B starts as the identity (curvature 1, no coupling).

**What we see.** TODO

<details><summary>plain 2D function: 9 runs</summary>

| γ | near the saddle | on the saddle | near a minimum |
|---|---|---|---|
| 1 | ![](docs/runs/2d/gamma1_close_ldh_guessed.png) | ![](docs/runs/2d/gamma1_on_ldh_guessed.png) | ![](docs/runs/2d/gamma1_near_min_ldh_guessed.png) |
| 0.1 | ![](docs/runs/2d/gamma0.1_close_ldh_guessed.png) | ![](docs/runs/2d/gamma0.1_on_ldh_guessed.png) | ![](docs/runs/2d/gamma0.1_near_min_ldh_guessed.png) |
| 0.01 | ![](docs/runs/2d/gamma0.01_close_ldh_guessed.png) | ![](docs/runs/2d/gamma0.01_on_ldh_guessed.png) | ![](docs/runs/2d/gamma0.01_near_min_ldh_guessed.png) |

</details>

<details><summary>2D function turned 45°: 9 runs</summary>

| γ | near the saddle | on the saddle | near a minimum |
|---|---|---|---|
| 1 | ![](docs/runs/2d_rot/gamma1_close_ldh_guessed.png) | ![](docs/runs/2d_rot/gamma1_on_ldh_guessed.png) | ![](docs/runs/2d_rot/gamma1_near_min_ldh_guessed.png) |
| 0.1 | ![](docs/runs/2d_rot/gamma0.1_close_ldh_guessed.png) | ![](docs/runs/2d_rot/gamma0.1_on_ldh_guessed.png) | ![](docs/runs/2d_rot/gamma0.1_near_min_ldh_guessed.png) |
| 0.01 | ![](docs/runs/2d_rot/gamma0.01_close_ldh_guessed.png) | ![](docs/runs/2d_rot/gamma0.01_on_ldh_guessed.png) | ![](docs/runs/2d_rot/gamma0.01_near_min_ldh_guessed.png) |

</details>

### 100D (random rotation)

No pictures in 100D; see the 100D columns of `results/cases_results.xlsx`.

**What we see.** TODO

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
- A. R. Curtis, M. J. D. Powell, J. K. Reid, *On the estimation of sparse Jacobian matrices*, IMA J. Appl. Math. 1974 (how B is measured with 3 gradients)