"""
One picture per 2D run, drawn by run_cases.py right after the run (no second run):
the function's contours, the path the method took, and on the side the method, its
parameters and how it did (steps to escape, steps to the minimum, final gap, work).
These are the README's pictures.

Only the 2D functions (plain and turned 45 degrees), and randomized methods only for
seed 0 (written on the picture).

WHERE
  docs/runs/<function>/gamma<g>_<start>_<method>.png
  e.g. docs/runs/2d_rot/gamma0.1_on_ldh.png
"""
import os
import textwrap

import matplotlib
matplotlib.use("Agg")                               # no screen needed, only files
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402

FUNCS_2D = ["2d", "2d_rot"]
SEED = 0
START_NAMES = {"far": "Far", "close": "Near saddle", "on": "On saddle", "near_min": "Near minimum"}


def fmt(v, spec="{:.0f}", none="never"):
    return none if v is None else spec.format(v)


def picture_path(out, fkey, gamma, start, module):
    """Where the picture of this run goes."""
    return os.path.join(out, fkey, f"gamma{gamma:g}_{start}_{module}.png")


def draw(algo, m, row_info, function_title, fkey, gamma, start, out):
    """Save the picture of one finished run. row_info = one METHODS row of run_cases.py,
    m = evaluate(algo)."""
    family, method, params, module, settings, random = row_info
    problem = algo.problem
    path = np.array(algo.path)
    z0 = path[0]

    # the window: the saddle, both minima, and the whole path, with a margin
    pts = np.vstack([path, problem.minima, np.zeros((1, 2))])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.15 * max(hi - lo) + 1e-3
    lo, hi = lo - pad, hi + pad
    X, Y = np.meshgrid(np.linspace(lo[0], hi[0], 250), np.linspace(lo[1], hi[1], 250))
    Z = problem.peek_f(np.stack([X, Y]))

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_axes([0.06, 0.09, 0.56, 0.82])
    ax.contourf(X, Y, Z, levels=40, cmap="viridis", alpha=0.85)
    ax.contour(X, Y, Z, levels=[0.0], colors="white", linewidths=0.8, linestyles="--")
    ax.plot(path[:, 0], path[:, 1], "-", color="black", lw=1.3)
    ax.plot(path[:, 0], path[:, 1], ".", color="black", ms=3)
    marks = [k for k, _ in algo.probe_log if k < len(path)]
    if marks:
        ax.plot(path[marks, 0], path[marks, 1], "x", color="#ff3b30", ms=9, mew=2,
                label="perturbation / probe round")
    ax.plot(*z0, "o", color="white", mec="black", ms=8, label="start")
    ax.plot(*path[-1], "o", color="#ff3b30", ms=8, label="end")
    ax.plot(0, 0, "*", color="white", mec="black", ms=14, label="saddle")
    mins = np.array(problem.minima)
    ax.plot(mins[:, 0], mins[:, 1], "D", color="gold", mec="black", ms=7, label="minima")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=7.5, loc="best", framealpha=0.8)
    ax.set_title(f"{method.split(':')[0]}  on  {function_title},  γ = {gamma:g}")

    rounds = f"{m.get('rounds', 0) or 0}" if algo.probe_log else "none"
    lines = [
        ("Method", method),
        ("Family", family),
        ("Parameters", params),
        ("Seed", str(SEED) if random else "deterministic"),
        ("", ""),
        ("Function", function_title),
        ("γ", f"{gamma:g}"),
        ("Start", f"{START_NAMES[start]}  ({z0[0]:.3g}, {z0[1]:.3g})"),
        ("", ""),
        ("Steps to escape  k_esc", "–" if start == "near_min" else fmt(m["k_esc"])),
        ("Steps to minimum  k_succ", fmt(m["k_succ"])),
        ("Final gap  q", f"{m['q']:.1e}"),
        ("Work  Nf / Ng / NH", f"{m['nf']} / {m['ng']} / {m['nh']}"),
        ("Probe / kick rounds", rounds),
        ("Ended at", f"({path[-1][0]:.3g}, {path[-1][1]:.3g})"),
    ]
    y = 0.95
    for key, val in lines:
        if not key:
            y -= 0.015                                # a small gap between groups
            continue
        fig.text(0.66, y, key, fontsize=8.5, color="#555555", va="top")
        wrapped = textwrap.wrap(val, 46) or [""]
        fig.text(0.66, y - 0.024, "\n".join(wrapped), fontsize=9.5, va="top", linespacing=1.2)
        y -= 0.024 + 0.031 * len(wrapped) + 0.006
    target = picture_path(out, fkey, gamma, start, module)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    fig.savefig(target, dpi=110)
    plt.close(fig)
    return target
