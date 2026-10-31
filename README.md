# Escaping Saddle Points

ENMG 616 Advanced Optimization, Fall 2026.

How gradient descent and three classical fixes (P-GD, GD-NC, and the perturbed GD of
Jin et al. 2017) behave around a saddle point, and a proposed method, **LDH (Learned
Diagonal Hessian)**: step sizes learned per dimension from measured function decrease,
and, when stuck, a few probes aimed by a cheap tridiagonal approximation of the Hessian.
No Hessian is ever computed.

> **Headline result.** TODO (one sentence with the main number, e.g. steps to the
> minimum on S2 for LDH vs P-GD vs GD-NC, and the work it took).

**Contents:**
[The problem](#the-problem) ·
[Quick start](#quick-start) ·
[The assignment: setup](#the-assignment-setup) ·
[Results: Questions 1-4](#results-questions-1-4) ·
[The proposed method: LDH](#the-proposed-method-ldh) ·
[Results: the proposed method](#results-the-proposed-method) ·
[Beyond the assignment: 27 cases](#beyond-the-assignment-27-cases) ·
[Every method, run by run](#every-method-run-by-run) ·
[Limitations](#limitations) ·
[Using the code](#using-the-code) ·
[Repository layout and branches](#repository-layout-and-branches) ·
[References](#references)

## The problem

For $\gamma > 0$:

$$f_\gamma(x, y) = \tfrac12 x^2 - \tfrac{\gamma}{2} y^2 + \tfrac14 y^4$$

| Point | What it is | $f$ | Hessian |
|---|---|---|---|
| $(0, 0)$ | saddle | $0$ | $\mathrm{diag}(1, -\gamma)$ |
| $(0, \pm\sqrt\gamma)$ | the two minima | $f^* = -\gamma^2/4$ | $\mathrm{diag}(1, 2\gamma)$ |

$\gamma$ is how strongly the saddle curves down along $y$. The smaller $\gamma$, the flatter
the way out and the shallower the minima, so the harder it is to notice that $(0,0)$ is
not a minimum. Near the saddle GD moves $y$ by a factor $(1 + t\gamma)$ per step, so the
escape takes about $\ln(\text{distance}/|y_0|)/\ln(1 + t\gamma)$ steps: slow when $\gamma$
is small and $y_0$ is tiny, and never when $y_0 = 0$.

## Quick start

    make install                  # once: a venv in ../venv with numpy, matplotlib, openpyxl, scipy
    git checkout all-methods      # every method in one place
    make test                     # all unit tests
    make run visual               # the 3D window: pick a method, change γ, start, angle
    make experiments              # every table, figure and picture in this README

`make experiments` is `make hw` (the assignment's tables), `make figures` (its three
figures) and `make cases` (the 27 extra cases and one picture per 2D run).

## The assignment: setup

| Case | $\gamma$ | Start $(x_0, y_0)$ | What changes |
|---|---|---|---|
| S1 | 1 | (0.05, 1e-8) | near the saddle |
| S2 | 0.1 | (0.05, 1e-8) | weaker negative curvature |
| S3 | 0.01 | (0.05, 1e-8) | still weaker |
| S4 | 1 | (0, 0) | exactly on the saddle |
| S5 | 1 | (0.05, 1 + 1e-4) | near a minimum |

- $K = 3000$ updates per run, no early stop on a small gradient.
- Deterministic methods run once; randomized ones 10 times, seeds 0-9 (same seeds when a
  parameter changes). Tables give mean ± sample standard deviation.
- Backtracking line search: start at $t = 1$, halve, Armijo constant $10^{-4}$, 40 trials.
- Counted work: $N_f$, $N_g$, $N_H$ (rejected trials included, diagnostics excluded).
- Defaults: $g_{trig} = 10^{-3}$, $r = 10^{-2}$, $T = 20$.

**Measures** (`saddle/metrics.py`). With $\Delta = f_s - f^*$ and gap
$q_k = (f(z_k) - f^*)/\Delta$:
escape $k_{esc}$ = first $k$ with $f(z_k) \le f_s - 0.1\Delta$;
success $k_{succ}$ = first $k$ with $q_k \le 10^{-4}$, $\|\nabla f\| \le 10^{-5}$ and no
negative curvature; time to success; final gap $q_K$; work $N_f / N_g / N_H$.

## Results: Questions 1-4

All numbers: `make hw` → `results/hw_results.xlsx` (the assignment's one-table format).

> TODO: paste the table for each question from `results/hw_results.xlsx`.

### Question 1. How does curvature affect escape? (S1-S3)

**Table.** TODO (S1, S2, S3 rows for GD, P-GD, GD-NC)

**What we see.** TODO (link the trend to the escape estimate in [The problem](#the-problem))

### Question 2. What happens exactly at the saddle? (S4 vs S1)

**Table.** TODO

**What we see.** TODO (which update creates motion away from the saddle; who fails)

### Question 3. What happens near a minimum? (S5)

**Table.** TODO (no escape column, as asked)

**What we see.** TODO (unnecessary perturbations, final gap, extra work)

### Question 4. Effect of the parameters (S2)

One parameter at a time, the others at their defaults: P-GD $g_{trig} \in \{10^{-4},
10^{-3}, 10^{-2}\}$, $r \in \{10^{-3}, 10^{-2}, 5\cdot10^{-2}\}$, $T \in \{10, 20, 50\}$;
GD-NC $g_{trig} \in \{10^{-4}, 10^{-3}, 10^{-2}\}$.

**Table.** TODO

![Effect of T](docs/figures/q4_parameter.png)

**What we see.** TODO

**Chosen configurations.** P-GD: TODO. GD-NC: TODO. Why: TODO.
(Re-run with them: `make hw PGD="T=..." GDNC="g_trig=..."`.)

### S2 figures

![S2 contours and paths](docs/figures/s2_contours.png)

![S2 normalized gap](docs/figures/s2_gaps.png)

## The proposed method: LDH

**Goal.** From a point near or on a saddle, reach a minimum in few updates using only
values of $f$ and $\nabla f$, with the extra work small and visible in the counts.

| Sub-goal | Part of LDH |
|---|---|
| the right step length in every direction | step sizes learned from $\rho$ |
| know cheaply which directions curve down | a tridiagonal approximation $B$ of the Hessian |
| when stuck, find a way down and check it | probes along $B$'s $m$ most negative directions |
| do not give up after one failed look | a small capped uphill kick |
| decide when it is a minimum | give up after $R$ rounds with no drop |
| do not leave the same way twice | a fading penalty on used directions |

### 1. Step sizes learned from $\rho$

Move along axis $i$ alone by $d_i g_i$. Second-order Taylor, with $H_i$ the curvature
along that axis:

$$f(z - d_i g_i e_i) \approx f(z) - d_i g_i^2 + \tfrac12 d_i^2 g_i^2 H_i$$

The first-order part predicts a drop of $d_i g_i^2$. Divide the real drop by it:

$$\rho_i = \frac{f(z) - f(z - d_i g_i e_i)}{d_i g_i^2} \approx 1 - \frac{d_i H_i}{2}$$

- $\rho_i = \tfrac12$ means $d_i = 1/H_i$: the step lands on the bottom of that direction.
- $\rho_i < 1$ (curves up): multiply $d_i$ by $\big(1/(2(1-\rho_i))\big)^\kappa = (1/(d_i H_i))^\kappa$,
  half way (in log scale) towards $1/H_i$ with $\kappa = 0.5$, clipped to $[0.1, 10]$.
- $\rho_i \ge 1$ (flat or curves down): grow $d_i$ by $\omega = 2$. Once the way down is
  found, the step keeps doubling: this is what makes the escape fast.

The GD step is then $z \leftarrow z - t\,D\,\nabla f(z)$ with $D = \mathrm{diag}(d_i)$ and
backtracking on $t$. Cost: one $f$ per dimension per update.

### 2. A tridiagonal $B$, and probes along its most negative directions

When the gradient is small ($\|\nabla f\| \le g_{trig}$), searching all $2N$ axis
directions is blind and expensive. Instead keep only the main diagonal of the Hessian
and its two neighbours. Two versions:

- **LDH (measured $B$).** $\nabla f(z + hs) - \nabla f(z) \approx h\,Hs$. In a tridiagonal
  matrix, row $i$ touches only columns $i-1, i, i+1$, so columns $1, 4, 7, \dots$ can be
  pushed in one gradient and read off separately (Curtis, Powell & Reid 1974): **3
  gradients** give all of $B$, whatever $N$ is. Only the $m$ smallest eigenvalues and their
  directions are computed (a tridiagonal solver); the upward directions never are.
- **LDH, guessed $B$.** Nothing is measured. The diagonal comes for free from the same
  $\rho$: $H_i \approx 2(1-\rho_i)/d_i$. Each neighbour coupling costs one extra $f$:
  making the two axis moves $w_i, w_j$ together,
  $$f(z + w_i e_i + w_j e_j) - f(z + w_i e_i) - f(z + w_j e_j) + f(z) = w_i w_j H_{ij}$$
  (exact for a quadratic). The probe directions come from each neighbour pair's
  $2\times2$ block, solved by a formula: no solver at all.

Probe $z \pm \delta v$ along the $m = 4$ chosen directions ($\delta = 10^{-2}$), score each
by its change in $f$ from where we got stuck plus its penalty, and jump to the best one
if it really lowers $f$. Real function values decide, so a rough $B$ cannot fool the method.

### 3. Kick, give-up, penalty

- **Kick.** No probe lower: step $r = 10^{-2}$ uphill along one bag direction (the
  least used), halving until $f$ rises by at most $r^2$. Probe again from there.
- **Give-up.** After $R = 3$ rounds with no drop: very likely a minimum. Go back to the
  best point seen and do not search again until the gradient is large again.
- **Penalty.** Leaving along $v$ sets $p_j \leftarrow \max(p_j, p_0 v_j^2)$; a probe along
  $u$ costs $\sum_j p_j u_j^2$ extra; every update $p \leftarrow \beta p$ ($p_0 = 10^{-4}$,
  $\beta = 0.9$). Directions just used are tried last.

### Cost per update

| Work | When | Cost |
|---|---|---|
| GD step | every update | 1 gradient + 1 $f$ per backtracking trial |
| learning $\rho$ | every update | $N$ values of $f$ (+ $N-1$ for the guessed $B$) |
| measuring $B$ | each search round (measured version) | 3 gradients |
| probes | each search round | $2m$ values of $f$ |
| kick | failed rounds | 1 $f$ per halving |

$N_H = 0$ in every run.

### Closest published work

- Saddle-free Newton (Dauphin et al. 2014) steps along the Hessian's eigenvectors with
  size $1/|\lambda|$. LDH never forms the Hessian and uses curvature only to aim a few probes.
- Perturbed GD (Jin et al. 2017): LDH's kick is the same ball, used only after a guided
  search fails.
- Learning the step from actual vs predicted decrease is the ratio test of
  trust-region methods (Nocedal & Wright, ch. 4), applied per dimension.

## Results: the proposed method

As asked in section 4 of the assignment: LDH on S1-S5 against GD and the two
configurations chosen in Question 4, all parameters fixed across the five cases.

**Table.** TODO (from `make hw PGD=... GDNC=...`)

**Effect of the modification (S2, same seeds and parameters).** TODO: benefits, extra
work ($N_f$), limitations.

## Beyond the assignment: 27 cases

Does it hold when the axes do not line up, and in many dimensions? `make cases` runs
GD, P-GD, GD-NC, Jin et al., Probe + Nesterov, LDH and LDH with a guessed $B$ on

- **functions:** the 2D function, the 2D function turned 45° (worst case for anything
  per axis), and 100D with a random rotation ($f$ written in turned coordinates
  $u = Qz$: $\tfrac12\sum_{i<N} u_i^2 - \tfrac\gamma2 u_N^2 + \tfrac14 u_N^4$),
- **$\gamma$:** 1, 0.1, 0.01,
- **starts:** near the saddle, exactly on it, near a minimum,

one run per cell (189 runs). Tables: `results/cases_results.xlsx`, merged headers
function → $\gamma$ → start.

**Summary.** TODO (one small table: steps to the minimum per method, worst case of each
function)

## Every method, run by run

Every picture is one run, drawn by `experiments/run_cases.py` (randomized methods: seed 0). Left: the contours of f, the path, × where the method perturbed or probed. Right: the method, its parameters and how it did. All numbers are in `results/cases_results.xlsx`.

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
## Limitations

- **Per-axis steps on a turned function.** The learned $d_i$ belong to the axes. When
  the valley runs diagonally, a step fitted to one axis overshoots across it, so the path
  zigzags along the valley (see the 45° pictures). The probes still escape.
- **Guessed $B$ needs movement to learn.** Exactly on the saddle the gradient is zero, so
  nothing is learned yet: the first probes use the starting guess ($B = I$) and the
  method may need a kick before $B$ means anything.
- **Cost of learning $\rho$.** $N$ values of $f$ per update: nothing in 2D, 100 per update in
  100D.
- **Band 1 on a dense Hessian.** On the random 100D rotation the true Hessian is dense;
  a tridiagonal $B$ is only an approximation there. The probes check it with real $f$
  values.
- TODO (anything the results show)

## Using the code

### The window

`make run visual` opens a 3D view of $f$ (drag to rotate, scroll to zoom) and a panel:
the method picker (GD, P-GD, GD-NC, Jin et al., Probe + Nesterov, LDH, LDH guessed $B$),
boxes for $\gamma$, $x_0$, $y_0$ and the angle (press **Apply**), **Start at (0,0)**,
**Play/Pause**, **Step**, **Reset**, a speed slider, and the numbers at the current
point (k, z, f, f*, gradient, Hessian when the method uses it, work, probe rounds).
Numbers on screen are computed without counting, so they do not change the work.

### Settings

    make run visual ALG=ldh GAMMA=0.01 X0=0 Y0=0 ANGLE=45
    make run headless ALG=all GAMMA="1 0.1 0.01"       # one table per gamma, no window

`ALG` (a method file name or `all`), `FUNC` (`assignment` or `star`), `GAMMA`, `X0`, `Y0`,
`ANGLE`, `K`, `SEED`, `G_TRIG`, `R`, `T`, `SPEED`. `FUNC=star` is a harder, degenerate
saddle (gradient and Hessian both zero at the origin); see `saddle/benchmark.py`.

### Experiments

    make hw                                   # S1-S5 + Question 4 tables      -> results/hw_results.xlsx
    make hw PGD="T=50" GDNC="g_trig=1e-4"     # + S1-S5 rows for your Q4 choices
    make figures                              # the 3 assignment figures       -> docs/figures/
    make cases                                # 27 cases + 2D pictures         -> results/cases_results.xlsx, docs/runs/

A stopped `make cases` resumes from `results/cases_runs.csv`.

## Repository layout and branches

    .
    ├── Makefile, main.py, requirements.txt
    ├── saddle/
    │   ├── benchmark.py        # f, gradient, Hessian with Nf/Ng/NH counters; turned and 100D versions
    │   ├── metrics.py          # escape / success / gap / work of one run
    │   ├── simulator.py        # the 3D window
    │   └── algorithms/         # base.py (the parent class) + one file per method
    ├── experiments/            # run_hw.py, hw_figures.py, run_cases.py, pictures.py
    ├── docs/                   # figures/ and runs/: the pictures in this README
    └── tests/

Every method is a child of `Algorithm` and only writes `update()`; the parent counts,
keeps the path and does the backtracking. `main` holds the shared code; each method lives
on its own branch made from `main` and never merged back; `all-methods` merges them all.

| Branch | Adds |
|---|---|
| `main` | benchmark, base class, simulator, metrics |
| `gd`, `pgd`, `gdnc` | the assignment's three methods |
| `jin-pgd` | Jin et al. 2017, Algorithm 2 |
| `learned-step`, `probe`, `probe-nesterov`, `probe-walk` | earlier steps towards LDH |
| `proposed` | LDH: `proposed.py` (one class, a switch per part), `ldh.py`, `ldh_guessed.py` |
| `all-methods` | everything merged + `experiments/` |

Each branch's tests: `make test`.

## References

- C. Jin, R. Ge, P. Netrapalli, S. M. Kakade, M. I. Jordan, *How to Escape Saddle Points
  Efficiently*, ICML 2017.
- J. D. Lee, M. Simchowitz, M. I. Jordan, B. Recht, *Gradient Descent Only Converges to
  Minimizers*, COLT 2016.
- Y. Dauphin et al., *Identifying and attacking the saddle point problem in
  high-dimensional non-convex optimization*, NeurIPS 2014.
- A. R. Curtis, M. J. D. Powell, J. K. Reid, *On the estimation of sparse Jacobian
  matrices*, IMA J. Appl. Math. 13, 1974.
- J. Nocedal, S. J. Wright, *Numerical Optimization*, 2nd ed., Springer 2006 (ch. 4,
  trust regions).
