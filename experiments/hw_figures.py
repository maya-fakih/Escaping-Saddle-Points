"""
The figures the assignment asks for (page 4, after Question 4), all on S2
(gamma = 0.1, start (0.05, 1e-8)):

    (saved in docs/figures/, so the README shows them)
    1. s2_contours.png    contours of f with every method's path; the points where a
                          method perturbed / probed / took a curvature step are marked
    2. s2_gaps.png        normalized gap q_k = (f(z_k) - f*) / (f_s - f*) against k
    3. q4_parameter.png   one parameter effect from Question 4: P-GD's gap curves for
                          T = 10, 20, 50 (seed 0)

Randomized runs use seed 0 (written in the legend, as the assignment asks).

    python experiments/hw_figures.py
    python experiments/hw_figures.py --pgd T=50 --gdnc g_trig=1e-4   # your Question 4 choices
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")                               # no screen needed, only files
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from saddle.algorithms import load                  # noqa: E402
from saddle.benchmark import RotatedSaddle          # noqa: E402
from run_hw import parse_settings                   # noqa: E402

GAMMA, START, SEED = 0.1, "close", 0

# (label, file in saddle/algorithms, random?)
METHODS = [("GD", "gd", False), ("P-GD", "pgd", True), ("GD-NC", "gdnc", False),
           ("LDH (proposed)", "ldh", True), ("LDH, guessed B", "ldh_guessed", True)]
COLORS = {"GD": "#4c4c4c", "P-GD": "#d62728", "GD-NC": "#1f77b4",
          "LDH (proposed)": "#2ca02c", "LDH, guessed B": "#9467bd"}


def run(module, settings, random):
    """Run one method on S2 and return the finished algorithm (path, probe log, ...)."""
    problem = RotatedSaddle.plain(GAMMA)
    kwargs = dict(settings)
    if random:
        kwargs["seed"] = SEED
    algo = load(module)(problem, problem.start(START), **kwargs)
    algo.run()
    return algo


def gaps(algo):
    """q_k for every visited point: how far from the minimum, in units of the saddle's depth."""
    P = algo.problem
    f = np.array([P.peek_f(z) for z in algo.path])
    return (f - P.fstar) / (0.0 - P.fstar)


def label(name, random, settings):
    extra = ", ".join(f"{k}={v:g}" for k, v in settings.items())
    text = name + (f" ({extra})" if extra else "")
    return text + (f", seed {SEED}" if random else "")


def contours(runs, path):
    """Figure 1: the paths on the contour map; markers where a method perturbed or probed."""
    P = RotatedSaddle.plain(GAMMA)
    s = np.sqrt(GAMMA)
    xs, ys = np.linspace(-0.08, 0.08, 300), np.linspace(-1.6 * s, 1.6 * s, 300)
    X, Y = np.meshgrid(xs, ys)
    Z = P.peek_f(np.stack([X, Y]))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contour(X, Y, Z, levels=30, cmap="Greys", linewidths=0.6)
    for name, algo, text in runs:
        pts = np.array(algo.path)
        dash = "--" if "guessed" in name else "-"
        ax.plot(pts[:, 0], pts[:, 1], dash, color=COLORS[name], lw=1.4, label=text)
        marks = [k for k, _ in algo.probe_log if k < len(pts)]
        if marks:
            ax.plot(pts[marks, 0], pts[marks, 1], "x", color=COLORS[name], ms=8, mew=2)
    ax.plot(0, 0, "k*", ms=12, label="saddle (0, 0)")
    ax.plot([0, 0], [s, -s], "ko", ms=6, label="minima (0, ±√γ)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"S2 (γ = {GAMMA}), start (0.05, 1e-8); x = perturbation / probe round")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def gap_plot(curves, path, title):
    """q_k against k, log scale; curves = [(text, q, color)]."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for text, q, color in curves:
        dash = "--" if "guessed" in text else "-"      # LDH and its guessed-B twin can overlap
        ax.semilogy(np.arange(len(q)), np.maximum(q, 1e-16), dash, color=color, lw=1.4, label=text)
    ax.axhline(1e-4, color="k", ls=":", lw=0.8, label="success gap 1e-4")
    ax.set_xlabel("update k")
    ax.set_ylabel("normalized gap q_k")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(0, None)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pgd", nargs="+", metavar="KEY=VALUE", help="chosen P-GD configuration")
    ap.add_argument("--gdnc", nargs="+", metavar="KEY=VALUE", help="chosen GD-NC configuration")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "docs", "figures"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    chosen = {"P-GD": parse_settings(args.pgd), "GD-NC": parse_settings(args.gdnc)}

    runs = []
    for name, module, random in METHODS:
        settings = chosen.get(name, {})
        runs.append((name, run(module, settings, random), label(name, random, settings)))

    contours(runs, os.path.join(args.out, "s2_contours.png"))
    gap_plot([(text, gaps(a), COLORS[n]) for n, a, text in runs],
             os.path.join(args.out, "s2_gaps.png"), f"S2 (γ = {GAMMA}): normalized gap")
    shades = ["#f4a582", "#d6604d", "#67001f"]
    q4 = [(label("P-GD", True, {"T": T}), gaps(run("pgd", {"T": T}, True)), c)
          for T, c in zip((10, 20, 50), shades)]
    gap_plot(q4, os.path.join(args.out, "q4_parameter.png"),
             "Question 4 on S2: effect of the P-GD spacing T")
    print("wrote s2_contours.png, s2_gaps.png and q4_parameter.png in", args.out)


if __name__ == "__main__":
    main()
