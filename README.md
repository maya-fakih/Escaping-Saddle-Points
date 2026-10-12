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
    ├── src/            # function, derivatives, optimizers, metrics
    ├── experiments/    # one script per setting / question
    ├── results/        # raw runs (z_0..z_K, timings, counts), tables
    ├── figures/        # contours, gap-vs-updates, parameter plots
    ├── notes/          # derivations and paper notes
    └── report/         # LaTeX source

## Branch workflow

`main` stays clean: only merged, verified work lands here.

| Branch | Content |
|--------|---------|
| `prep-a1-geometry` | Gradient, Hessian, stationary points, numerical checks, GD analysis, Newton |
| `prep-a2-papers` | Notes on Jin et al. 2017 and Lee et al. 2016 |
| `q1-curvature` | S1-S3 |
| `q2-exact-saddle` | S4 |
| `q3-near-minimum` | S5 |
| `q4-parameters` | Parameter sweeps on S2 |
| `proposed-method` | Design, S1-S5 comparison, S2 ablation |
| `report` | LaTeX report |

Each branch is merged into `main` through a pull request once its results are checked.

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

## Running

    pip install numpy scipy matplotlib
    python experiments/run_all.py

(Update this section as the code takes shape.)

## References

- C. Jin et al., *How to Escape Saddle Points Efficiently*, ICML 2017
- J. D. Lee et al., *Gradient Descent Only Converges to Minimizers*, COLT 2016