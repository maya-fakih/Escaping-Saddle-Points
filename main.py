"""
Entry point. Usually you call it through the Makefile (make run ...), but it works alone:

    python main.py visual   --alg gd   --case S1
    python main.py headless --alg all  --case S3 --k 3000

visual   : opens the 3D simulator; pick the method in the window.
headless : no window; runs the method(s) to the end and prints a results table.

Every setting is a parameter. --case fills in gamma and the start point from the
assignment; --gamma / --x0 / --y0 override it. Each method only receives the
parameters it actually has (P-GD takes seed, r, T; GD takes none of those).
"""
import argparse
import inspect
import time

from saddle.algorithms import available, load
from saddle.benchmark import Benchmark
from saddle.cases import CASES, K, G_TRIG, R, T
from saddle.metrics import evaluate


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["visual", "headless"])
    ap.add_argument("--alg", default="all", help="a method on this branch, or 'all'")
    ap.add_argument("--case", default="S1", choices=CASES)
    ap.add_argument("--gamma", type=float, help="overrides the case's gamma")
    ap.add_argument("--x0", type=float, help="overrides the case's start x")
    ap.add_argument("--y0", type=float, help="overrides the case's start y")
    ap.add_argument("--k", type=int, default=K, help="max number of updates")
    ap.add_argument("--seed", type=int, default=0, help="random seed (P-GD)")
    ap.add_argument("--g-trig", type=float, default=G_TRIG, help="'gradient is small' threshold")
    ap.add_argument("--r", type=float, default=R, help="P-GD kick radius")
    ap.add_argument("--t", type=int, default=T, help="P-GD: updates between kicks")
    ap.add_argument("--speed", type=int, default=5, help="visual: steps per second")
    ap.add_argument("--frames", type=int, nargs="*",
                    help="visual: save PNGs at these iterations instead of opening a window")
    return ap.parse_args()


def builder(args):
    """Returns build(name): makes a fresh Algorithm with all the chosen settings."""
    gamma, (x0, y0) = CASES[args.case]
    gamma = args.gamma if args.gamma is not None else gamma
    z0 = (args.x0 if args.x0 is not None else x0, args.y0 if args.y0 is not None else y0)
    settings = dict(max_iters=args.k, seed=args.seed, g_trig=args.g_trig, r=args.r, T=args.t)

    def build(name):
        cls = load(name)
        accepted = inspect.signature(cls.__init__).parameters
        kwargs = {key: val for key, val in settings.items() if key in accepted}
        return cls(Benchmark(gamma), z0, **kwargs)

    return build, gamma, z0


def headless(build, names, gamma, z0):
    print(f"gamma = {gamma}, start = {z0}\n")
    head = f"{'method':<18}{'updates':>8}{'k_esc':>8}{'k_succ':>8}{'final f':>14}" \
           f"{'gap q':>11}{'Nf':>7}{'Ng':>7}{'NH':>6}{'time (s)':>10}   final z"
    print(head)
    print("-" * (len(head) + 20))
    for name in names:
        algo = build(name)
        t0 = time.perf_counter()
        algo.run()
        sec = time.perf_counter() - t0
        m = evaluate(algo)
        show = lambda v: "-" if v is None else str(v)
        print(f"{algo.name:<18}{m['updates']:>8}{show(m['k_esc']):>8}{show(m['k_succ']):>8}"
              f"{m['f']:>14.6e}{m['q']:>11.1e}{m['nf']:>7}{m['ng']:>7}{m['nh']:>6}{sec:>10.4f}"
              f"   ({m['z'][0]:.3e}, {m['z'][1]:.3e})"
              + (f"   line search failed at k={m['failed_at']}" if m['failed_at'] is not None else ""))
    print("\nk_esc: first k that is 10% of the way down from the saddle;"
          " k_succ: first k at a real minimum (see saddle/metrics.py)")


def visual(args, build, names):
    if args.frames is not None:
        import matplotlib
        matplotlib.use("Agg")                # no screen needed for saving pictures
    from saddle.simulator import Simulator

    first = names[0]
    sim = Simulator(build, available(), first, speed=args.speed)
    if args.frames is None:
        sim.show()
        return
    for target in sorted(args.frames):
        while sim.algo.k < target and not sim.algo.done:
            sim.algo.step()                  # step without redrawing: much faster
        sim.redraw()
        out = f"frame_{first}_{args.case}_{sim.algo.k}.png"
        sim.save(out)
        print("saved", out)


def main():
    args = parse()
    names = available() if args.alg == "all" else [args.alg]
    if not names:
        raise SystemExit("No methods on this branch. Try: git checkout all-methods")
    for n in names:
        load(n)                              # clear error message if a name is wrong
    build, gamma, z0 = builder(args)
    if args.mode == "headless":
        headless(build, names, gamma, z0)
    else:
        visual(args, build, names)


if __name__ == "__main__":
    main()
