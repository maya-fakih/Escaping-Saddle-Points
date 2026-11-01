"""Compare LDH-GD, assignment P-GD and Jin-inspired P-GD in 100 dimensions.

Run from the repository root:
    ../venv/bin/python experiments/compare_100d.py

Defaults: one fixed dense rotation (seed 0), gamma 1/0.1/0.01, near/exact
saddle starts, optimizer seeds 0-9, and 3000 updates: 180 runs. No plain GD.
The Jin baseline is the repository's backtracking/fixed-threshold adaptation,
not the theoretically calibrated fixed-step Algorithm 2 from the paper.
"""
import os
# Avoid tiny dense matrix operations spawning many BLAS threads. Set before NumPy.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from saddle.algorithms.ldh import LDH
from saddle.algorithms.pgd import PGD
from saddle.algorithms.jin_pgd import JinPGD
from saddle.benchmark import RotatedSaddle
from saddle.metrics import evaluate

METHODS = {"LDH-GD": LDH, "Assignment P-GD": PGD,
           "Jin-inspired P-GD": JinPGD}
FIELDS = ["gamma", "start", "method", "seed", "updates", "k_esc", "k_succ",
          "t_succ", "q", "nf", "ng", "nh", "nf_succ", "ng_succ", "nh_succ",
          "failed_at", "search_rounds", "kick_count", "gave_up", "optimizer_seconds"]


def mean_sd(values, scientific=False):
    values = [float(v) for v in values if v not in (None, "")]
    if not values:
        return "NR"
    fmt = ".3e" if scientific else ".1f"
    text = format(statistics.mean(values), fmt)
    if len(values) > 1:
        text += " +/- " + format(statistics.stdev(values), fmt)
    return text


def summary(rows, args):
    text = ["# 100D dense-rotation comparison", "",
            "Three methods only; no plain GD. Rotation seed: " + str(args.rotation_seed),
            "LDH-GD R=10 (dimension rule); other parameters at repository defaults.",
            "P-GD: g_trig=1e-3, r=1e-2, T=20. Jin-inspired: same plus f_thres=1e-7.",
            "Jin-inspired uses backtracking and practical thresholds, not the paper's theorem parameters.",
            "Mean +/- sample SD. Escape uses escaping runs; success/time use successful runs.",
            "Final gap/full work use all runs. NR = target not reached.",
            "Jin's stop rule freezes its returned point; the loop still reaches the update budget.", "",
            "| gamma | Start | Method | Success | Escape (count) | Success update | Time to success (s) | Final gap | Full Nf / Ng / NH | Work to success Nf / Ng / NH |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for gamma in args.gammas:
        for start in args.starts:
            for method in METHODS:
                group = [r for r in rows if float(r["gamma"]) == gamma
                         and r["start"] == start and r["method"] == method]
                if not group:
                    continue
                values = lambda key: [r[key] for r in group]
                successes = sum(r["k_succ"] not in (None, "") for r in group)
                escapes = sum(r["k_esc"] not in (None, "") for r in group)
                cells = [str(gamma), start, method, f"{successes}/{len(group)}",
                         mean_sd(values("k_esc")) + f" ({escapes}/{len(group)})",
                         mean_sd(values("k_succ")), mean_sd(values("t_succ"), True),
                         mean_sd(values("q"), True),
                         " / ".join(mean_sd(values(k)) for k in ("nf", "ng", "nh")),
                         " / ".join(mean_sd(values(k)) for k in
                                    ("nf_succ", "ng_succ", "nh_succ"))]
                text.append("| " + " | ".join(cells) + " |")
    text += ["", "Timing excludes diagnostics and initialization; see optimizer_seconds for full update time.",
             "Individual values are in runs.csv; complete paths, per-iterate times and work are in trajectories/.",
             "Results apply to one rotation and this benchmark, not arbitrary 100D functions."]
    return "\n".join(text) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gammas", nargs="+", type=float, default=[1.0, 0.1, 0.01])
    parser.add_argument("--starts", nargs="+", choices=["close", "on"], default=["close", "on"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--rotation-seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=3000)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "comparison_100d")
    parser.add_argument("--resume", action="store_true", help="Reuse completed rows from this exact configuration")
    args = parser.parse_args()
    if args.k < 1 or any(g <= 0 for g in args.gammas):
        parser.error("k and all gamma values must be positive")
    if len(args.seeds) != len(set(args.seeds)) or len(args.gammas) != len(set(args.gammas)) or len(args.starts) != len(set(args.starts)):
        parser.error("Do not repeat seeds, gammas, or starts")
    args.out.mkdir(parents=True, exist_ok=True)
    history = args.out / "trajectories"
    history.mkdir(exist_ok=True)
    config = {"dimension": 100, "rotation_seed": args.rotation_seed,
              "gammas": args.gammas, "starts": args.starts, "seeds": args.seeds,
              "K": args.k, "LDH_R": 10, "g_trig": 1e-3, "r": 1e-2, "T": 20,
              "jin_f_thres": 1e-7}
    sources = ["benchmark.py", "metrics.py", "algorithms/base.py", "algorithms/proposed.py",
               "algorithms/ldh.py", "algorithms/pgd.py", "algorithms/jin_pgd.py"]
    config["source_sha256"] = {p: hashlib.sha256((ROOT / "saddle" / p).read_bytes()).hexdigest()
                               for p in sources}
    config_path = args.out / "config.json"
    csv_path = args.out / "runs.csv"
    rows = []
    if csv_path.exists():
        if not args.resume:
            parser.error("Results already exist. Use --resume or choose a different --out directory.")
        if not config_path.exists() or json.loads(config_path.read_text()) != config:
            parser.error("Saved configuration or algorithm source differs; use a new --out directory.")
        with csv_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    key = lambda row: (float(row["gamma"]), row["start"], row["method"], int(row["seed"]))
    completed = {key(row) for row in rows}
    jobs = [(g, start, name, seed) for g in args.gammas for start in args.starts
            for name in METHODS for seed in args.seeds]
    print(f"{len(jobs)} runs; {len(completed)} already completed. Output: {args.out}", flush=True)
    with csv_path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if not rows:
            writer.writeheader()
        for index, (gamma, start, name, seed) in enumerate(jobs, 1):
            if (gamma, start, name, seed) in completed:
                continue
            problem = RotatedSaddle.random(100, gamma, seed=args.rotation_seed)
            # Pass seed explicitly, including through LDH's **kwargs wrapper.
            algorithm = METHODS[name](problem, problem.start(start), seed=seed, max_iters=args.k)
            if name == "LDH-GD" and algorithm.R != 10:
                raise RuntimeError("LDH must use the new dimension-dependent budget R=10; pull all-methods.")
            algorithm.run()
            metrics = evaluate(algorithm)  # Post-run diagnostics: not optimizer work/time.
            row = {k: metrics.get(k) for k in FIELDS}
            row.update(gamma=gamma, start=start, method=name, seed=seed,
                       search_rounds=len(algorithm.probe_log),
                       kick_count=len(getattr(algorithm, "kicks", [])),
                       gave_up=any(outcome == "gave up" for _, outcome in algorithm.probe_log),
                       optimizer_seconds=algorithm.clock)
            filename = f"g{gamma:g}_{start}_{name.split()[0]}_seed{seed}.npz"
            np.savez_compressed(history / filename, z=np.asarray(algorithm.path),
                                times=np.asarray(algorithm.times), work=np.asarray(algorithm.work))
            writer.writerow(row)
            stream.flush()  # Keep each completed run if interrupted.
            rows.append(row)
            (args.out / "summary.md").write_text(summary(rows, args))
            print(f"[{index}/{len(jobs)}] gamma={gamma:g} {start}, {name}, seed={seed}: "
                  f"escape={row['k_esc']} success={row['k_succ']} "
                  f"Nf={row['nf']} Ng={row['ng']}", flush=True)
    (args.out / "summary.md").write_text(summary(rows, args))
    print("Done. Share summary.md and runs.csv; keep config.json and trajectories for reproducibility.")
    print("The repository ignores results/. We can explicitly stage the CSV, summary and config after review.")


if __name__ == "__main__":
    main()
