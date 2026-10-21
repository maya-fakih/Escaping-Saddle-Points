"""
Open the 3D simulator for one method on one of the assignment's cases.

    python simulate.py gd                 # GD on S1
    python simulate.py pgd --case S3 --speed 20
    python simulate.py gdnc --case S4

The method must exist on the branch you are on (git checkout <method>).
Use --frames to save pictures instead of opening a window, e.g.
    python simulate.py gd --frames 0 5 50
"""
import argparse
import sys

from saddle.algorithms import load
from saddle.benchmark import Benchmark
from saddle.cases import CASES, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", help="file name in saddle/algorithms/, e.g. gd, pgd, gdnc, probe")
    ap.add_argument("--case", default="S1", choices=CASES)
    ap.add_argument("--speed", type=int, default=5, help="steps per second when playing")
    ap.add_argument("--iters", type=int, default=K, help="max number of updates")
    ap.add_argument("--frames", type=int, nargs="*",
                    help="save frame_<k>.png at these iterations instead of opening a window")
    args = ap.parse_args()

    if args.frames is not None:
        import matplotlib
        matplotlib.use("Agg")          # no screen needed
    from saddle.simulator import Simulator

    gamma, z0 = CASES[args.case]
    algo = load(args.method)(Benchmark(gamma), z0, max_iters=args.iters)
    sim = Simulator(algo, speed=args.speed, title=f"{algo.name} on {args.case} (gamma = {gamma})")

    if args.frames is None:
        sim.show()
        return
    for target in sorted(args.frames):
        while algo.k < target and not algo.done:
            sim.advance()
        name = f"frame_{args.method}_{args.case}_{algo.k}.png"
        sim.save(name)
        print("saved", name)


if __name__ == "__main__":
    sys.exit(main())
