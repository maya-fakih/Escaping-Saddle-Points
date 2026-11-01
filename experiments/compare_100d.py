"""Compare LDH-GD, assignment P-GD and Jin-inspired P-GD in 100 dimensions.

Run from the repository root:
    ../venv/bin/python experiments/compare_100d.py

Defaults: one fixed dense rotation (seed 0), gamma 1/0.1/0.01, near/exact
saddle starts, optimizer seeds 0 and 1, and 3000 updates: 36 runs. No plain GD.
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



def write_excel(rows, config, output_path):
    """Export numeric results and grouped sample statistics to an Excel workbook."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    book = Workbook()
    summary_sheet = book.active
    summary_sheet.title = "Summary"
    metrics = FIELDS[4:]
    headers = ["gamma", "start", "method", "runs", "successes", "escapes"]
    for metric in metrics:
        headers.extend([metric + "_n", metric + "_mean", metric + "_sample_sd"])
    summary_sheet.append(headers)
    def numbers(group, metric):
        return [float(r[metric]) for r in group
                if r.get(metric) not in (None, "")]
    groups = {}
    for row in rows:
        groups.setdefault((float(row["gamma"]), row["start"], row["method"]), []).append(row)
    for (gamma, start, method), group in groups.items():
        line = [gamma, start, method, len(group),
                len(numbers(group, "k_succ")), len(numbers(group, "k_esc"))]
        for metric in metrics:
            values = numbers(group, metric)
            line.extend([len(values), statistics.mean(values) if values else None,
                         statistics.stdev(values) if len(values) > 1 else None])
        summary_sheet.append(line)

    runs_sheet = book.create_sheet("Individual runs")
    runs_sheet.append(FIELDS)
    for row in rows:
        line = []
        for field in FIELDS:
            value = row.get(field)
            if value in (None, ""):
                value = None
            elif field not in ("method", "start"):
                if field == "gave_up":
                    value = value if isinstance(value, bool) else str(value).lower() == "true"
                else:
                    value = float(value)
            line.append(value)
        runs_sheet.append(line)

    settings = book.create_sheet("Settings and notes")
    settings.append(["Setting", "Value"])
    for key, value in config.items():
        settings.append([key, json.dumps(value) if isinstance(value, (list, dict)) else value])
    for note in [
        "Blank target values mean not reached; they are never counted as zero.",
        "Each summary metric has its own sample count, mean and sample standard deviation.",
        "Escape/success statistics use only runs reaching that target.",
        "Final gap and full work statistics use all completed runs.",
        "Sample standard deviation is blank when fewer than two observations exist.",
        "gave_up mean is the fraction of completed runs that gave up probing.",
        "Two seeds provide descriptive results only, not a reliable variability estimate.",
        "Times are seconds. q is the final normalized objective gap.",
        "Jin-inspired P-GD is the repository adaptation, not the paper's exact configuration.",
        "This workbook is a snapshot of completed runs; rerun with --resume to refresh.",
    ]:
        settings.append(["Note", note])
    for sheet in book:
        sheet.freeze_panes = "D2" if sheet == summary_sheet else "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="203864")
            cell.alignment = Alignment(wrap_text=True, vertical="center")
        sheet.row_dimensions[1].height = 44
        for column in sheet.columns:
            title = str(column[0].value)
            width = min(28, max(14, len(title) + 2))
            sheet.column_dimensions[get_column_letter(column[0].column)].width = width
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = "0.000000E+00" if (0 < abs(cell.value) < 0.001) else "0.000"
    settings.column_dimensions["A"].width = 26
    settings.column_dimensions["B"].width = 100
    for row in settings.iter_rows(min_row=2):
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
        settings.row_dimensions[row[0].row].height = 32
    temporary = output_path.with_name(output_path.stem + ".tmp.xlsx")
    book.save(temporary)
    temporary.replace(output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gammas", nargs="+", type=float, default=[1.0, 0.1, 0.01])
    parser.add_argument("--starts", nargs="+", choices=["close", "on"], default=["close", "on"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
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
    write_excel(rows, config, args.out / "comparison_100d.xlsx")
    print("Excel workbook: " + str(args.out / "comparison_100d.xlsx"))
    print("Done. Share summary.md and runs.csv; keep config.json and trajectories for reproducibility.")
    print("The repository ignores results/. We can explicitly stage the CSV, summary and config after review.")


if __name__ == "__main__":
    main()
