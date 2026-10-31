"""
The 27 test cases: every compared method on 3 functions x 3 gammas x 3 starts.

    python experiments/run_cases.py                 # everything (2D is fast, 100D takes a while)
    python experiments/run_cases.py --quick         # a tiny slice, to check that it runs
    python experiments/run_cases.py --workers 8     # use 8 CPU cores
    python experiments/run_cases.py --functions 2d 2d_rot      # only part of it

WHAT IT RUNS
  Columns, nested:  function  ->  gamma  ->  start
      functions: 2D benchmark, 2D turned 45 degrees (the worst case for per-axis steps),
                 100D with a random rotation    (see RotatedSaddle in saddle/benchmark.py)
      gammas:    1, 0.1, 0.01                   (the assignment's S1, S2, S3 values)
      starts:    near the saddle (0.05, 1e-8), exactly on it, near a minimum
  Rows: GD, P-GD, GD-NC (assignment defaults), Jin et al. PGD, Probe + Nesterov,
        LDH (the proposed method), LDH with a guessed B.
  (100d_band and the "far" start are still available with --functions / --starts.)
  The assignment's own tables (S1-S5, Question 4) are made by run_hw.py.

THE RULES (the assignment's, page 3)
  K = 3000 updates, no early stop; ONE run per cell (seed 0; --seeds N for more),
  deterministic ones once; mean +- sample standard deviation; "NR" when a target is
  never met; "-" for the escape iteration at the near-minimum start; escape and success
  numbers are averaged over the runs that reach them, final gap and work over all runs.

WHAT IT WRITES (into results/)
  cases_runs.csv      one line per run (every number, so nothing is lost)
  cases_results.xlsx  one sheet per measure, merged headers function -> gamma -> start:
                      Success, Escape k_esc, Success k_succ, Time to success, Final gap,
                      Work Nf-Ng-NH, Work to success, Rounds
  docs/runs/<function>/gamma<g>_<start>_<method>.png   (in docs/, so the README shows them)
                      one picture per 2D run (seed 0 for random methods), drawn right
                      after that run: contours, the path, and the method's numbers on
                      the side (see pictures.py). --no-pictures skips them.
  If cases_runs.csv already has some runs, they are reused (so a stopped run can resume);
  a 2D run whose picture is missing is run again, only to draw it (2D runs are fast).
"""
import argparse
import csv
import inspect
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # so "saddle" can be imported

from saddle.algorithms import load                 # noqa: E402
from saddle.benchmark import RotatedSaddle         # noqa: E402
from saddle.metrics import evaluate                # noqa: E402
import pictures                                    # noqa: E402

N_BIG = 100                                        # size of the high-dimensional tests
K = 3000
SEEDS = list(range(10))

# ----------------------------------------------------------------------------
# The columns
# ----------------------------------------------------------------------------
FUNCTIONS = {   # key: (title in the table, how to build it for a gamma)
    "2d":        ("2D benchmark f_γ(x, y)",              lambda g: RotatedSaddle.plain(g)),
    "2d_rot":    ("2D benchmark turned 45°",             lambda g: RotatedSaddle.rotated_2d(g, 45)),
    "100d_band": ("100D, neighbouring coordinates coupled", lambda g: RotatedSaddle.coupled(N_BIG, g)),
    "100d_rand": ("100D, random rotation",               lambda g: RotatedSaddle.random(N_BIG, g, seed=0)),
}
GAMMAS = [1.0, 0.1, 0.01]
STARTS = {"far": "Far", "close": "Near saddle", "on": "On saddle", "near_min": "Near minimum"}
DEFAULT_FUNCTIONS = ["2d", "2d_rot", "100d_rand"]
DEFAULT_STARTS = ["close", "on", "near_min"]

# ----------------------------------------------------------------------------
# The rows: (family, method, parameters shown, file in saddle/algorithms, settings, random?)
# "Proposed method" rows all use the same class with some parts switched off.
# ----------------------------------------------------------------------------
METHODS = [
    ("Assignment baselines", "GD", "backtracking", "gd", {}, False),
    ("Assignment baselines", "P-GD", "default: g_trig=1e-3, r=1e-2, T=20", "pgd", {}, True),
    ("Assignment baselines", "GD-NC", "default: g_trig=1e-3", "gdnc", {}, False),
    ("Literature", "Perturbed GD, Jin et al. 2017 (Algorithm 2)",
     "r=1e-2, g_thres=1e-3, t_thres=20, f_thres=1e-7", "jin_pgd", {}, True),
    ("Earlier variant", "Probe + Nesterov", "probe when stuck, Nesterov otherwise",
     "probe_nesterov", {}, False),
    ("Proposed method", "LDH: learned per-dimension steps + probes along B's m most negative directions",
     "B measured (3 gradients), m = 4", "ldh", {}, True),
    ("Proposed method", "LDH, guessed B: B learned from rho + neighbour steps, no eigen-solver",
     "2 x 2 blocks, m = 4", "ldh_guessed", {}, True),
]

FIELDS = ["row", "function", "gamma", "start", "seed", "updates", "k_esc", "k_succ", "t_succ",
          "q", "nf", "ng", "nh", "nf_succ", "ng_succ", "nh_succ", "rounds", "escape_round",
          "stuck_round", "rounds_to_min", "failed_at", "seconds"]


# ----------------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------------
def one_run(job):
    """Run one method once on one test and return its numbers (one CSV line)."""
    row, fkey, gamma, start, seed, iters, pic_dir = job
    _, _, _, module, settings, _ = METHODS[row]
    problem = FUNCTIONS[fkey][1](gamma)
    cls = load(module)
    kwargs = dict(settings, max_iters=iters)
    if "seed" in inspect.signature(cls.__init__).parameters:
        kwargs["seed"] = seed
    t0 = time.perf_counter()
    algo = cls(problem, problem.start(start), **kwargs)
    algo.run()
    m = evaluate(algo)
    if pic_dir and wants_picture(fkey, seed):          # drawn now, from this same run
        pictures.draw(algo, m, METHODS[row], FUNCTIONS[fkey][0], fkey, gamma, start, pic_dir)
    m.update(row=row, function=fkey, gamma=gamma, start=start, seed=seed,
             seconds=round(time.perf_counter() - t0, 3))
    return {k: m.get(k) for k in FIELDS}


def wants_picture(fkey, seed):
    """Pictures only for the 2D functions, and only one run per cell (seed 0)."""
    return fkey in pictures.FUNCS_2D and seed == pictures.SEED


def make_jobs(rows, functions, gammas, starts, seeds, iters, pic_dir=None):
    """Every (method, test, seed) to run. Deterministic methods get one seed only."""
    jobs = []
    for row in rows:
        random = METHODS[row][5]
        for fkey in functions:
            for gamma in gammas:
                for start in starts:
                    for seed in (seeds if random else seeds[:1]):
                        jobs.append((row, fkey, gamma, start, seed, iters, pic_dir))
    return jobs


def load_done(path):
    """Runs already saved in the CSV (so a stopped grid can carry on)."""
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        done = {}
        for line in csv.DictReader(fh):
            key = (int(line["row"]), line["function"], float(line["gamma"]), line["start"],
                   int(line["seed"]))
            done[key] = line
        return done


# ----------------------------------------------------------------------------
# Summarizing, in the assignment's style
# ----------------------------------------------------------------------------
def num(v):
    return None if v in (None, "", "None") else float(v)


def mean_sd(values, fmt="{:.0f}"):
    """'mean ± sd' over the values that exist; 'NR' if none; no sd if only one."""
    vals = [v for v in values if v is not None]
    if not vals:
        return "NR"
    if len(vals) == 1:
        return fmt.format(vals[0])
    return f"{fmt.format(np.mean(vals))} ± {fmt.format(np.std(vals, ddof=1))}"


def summarize(runs, start):
    """Every measure for one cell (one method on one test), as text."""
    n = len(runs)
    get = lambda key: [num(r[key]) for r in runs]
    succ = sum(v is not None for v in get("k_succ"))
    esc = [v for v in get("k_esc") if v is not None]
    escape = "–" if start == "near_min" else mean_sd(esc)
    if start != "near_min" and 0 < len(esc) < n:
        escape += f" ({len(esc)}/{n} escape)"
    work = " / ".join(mean_sd(get(k)) for k in ("nf", "ng", "nh"))
    work_succ = " / ".join(mean_sd(get(k)) for k in ("nf_succ", "ng_succ", "nh_succ"))
    rounds = " / ".join(mean_sd(get(k)) for k in ("escape_round", "stuck_round", "rounds_to_min"))
    failed = [r for r in runs if num(r["failed_at"]) is not None]
    if failed:
        work += f" (line search failed in {len(failed)} run(s))"
    return {
        "Success": f"{succ}/{n}",
        "Escape k_esc": escape,
        "Success k_succ": mean_sd(get("k_succ")),
        "Time to success": mean_sd(get("t_succ"), "{:.4f}"),
        "Final gap": mean_sd(get("q"), "{:.1e}"),
        "Work Nf-Ng-NH": work,
        "Work to success": work_succ,
        "Rounds": rounds,
    }


SHEET_NOTES = {
    "Success": "runs that reach a real minimum at least once / runs",
    "Escape k_esc": "first update 10% of the way down from the saddle (over runs that escape)",
    "Success k_succ": "first update at a real minimum (over successful runs)",
    "Time to success": "optimizer seconds at k_succ (over successful runs)",
    "Final gap": "q_K = (f(z_K) - f*) / (f_saddle - f*) after all updates (over all runs)",
    "Work Nf-Ng-NH": "f / gradient / Hessian evaluations in the whole run (over all runs)",
    "Work to success": "f / gradient / Hessian evaluations up to k_succ",
    "Rounds": "escape round / gave-up round / rounds before the minimum",
}


def write_excel(path, done, functions, gammas, starts):
    """One sheet per measure, in the test-plan layout (merged, nested headers)."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    cells = {}                                         # (row, f, g, s) -> summary dict
    groups = {}
    for (row, f, g, s, _), line in done.items():
        groups.setdefault((row, f, g, s), []).append(line)
    for key, runs in groups.items():
        cells[key] = summarize(runs, key[3])

    wb = Workbook()
    wb.remove(wb.active)
    thin = Side(style="thin", color="999999")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    fills = ["DCE6F1", "E2EFDA", "FCE4D6", "EDE7F6"]
    for sheet, note in SHEET_NOTES.items():
        ws = wb.create_sheet(sheet[:31])
        ws.cell(1, 1, f"{sheet}: {note}").font = Font(bold=True)
        top = 2
        for c, title in enumerate(["Family", "Method", "Parameters"], start=1):
            ws.merge_cells(start_row=top, start_column=c, end_row=top + 2, end_column=c)
            ws.cell(top, c, title).font = Font(bold=True)
        col = 4
        colmap = {}
        for fi, f in enumerate(functions):
            fill = PatternFill("solid", fgColor=fills[fi % len(fills)])
            width = len(gammas) * len(starts)
            ws.merge_cells(start_row=top, start_column=col, end_row=top, end_column=col + width - 1)
            ws.cell(top, col, FUNCTIONS[f][0]).fill = fill
            for gi, g in enumerate(gammas):
                gc = col + gi * len(starts)
                ws.merge_cells(start_row=top + 1, start_column=gc, end_row=top + 1,
                               end_column=gc + len(starts) - 1)
                ws.cell(top + 1, gc, f"γ = {g:g}").fill = fill
                for si, s in enumerate(starts):
                    ws.cell(top + 2, gc + si, STARTS[s]).fill = fill
                    colmap[(f, g, s)] = gc + si
            col += width
        for r, (fam, meth, par, *_rest) in enumerate(METHODS):
            rr = top + 3 + r
            ws.cell(rr, 1, fam); ws.cell(rr, 2, meth); ws.cell(rr, 3, par)
            for (f, g, s), c in colmap.items():
                if (r, f, g, s) in cells:
                    ws.cell(rr, c, cells[(r, f, g, s)][sheet])
        for row in ws.iter_rows(min_row=top, max_row=top + 2 + len(METHODS), max_col=col - 1):
            for cell in row:
                cell.border = box
                cell.alignment = center if cell.column > 3 or cell.row < top + 3 else \
                    Alignment(vertical="center", wrap_text=True)
        ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = 22, 40
        ws.column_dimensions["C"].width = 28
        for c in range(4, col):
            ws.column_dimensions[ws.cell(1, c).column_letter].width = 14
        ws.freeze_panes = ws.cell(top + 3, 4)
    wb.save(path)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke test: 2 seeds, 50 updates, gamma 0.1, 10D instead of 100D")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="CPU cores to use")
    ap.add_argument("--functions", nargs="+", default=DEFAULT_FUNCTIONS, choices=FUNCTIONS)
    ap.add_argument("--gammas", nargs="+", type=float, default=GAMMAS)
    ap.add_argument("--starts", nargs="+", default=DEFAULT_STARTS, choices=STARTS)
    ap.add_argument("--rows", nargs="+", type=int, default=list(range(len(METHODS))),
                    help="which method rows to run (0-based, see METHODS)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "results"))
    ap.add_argument("--no-pictures", action="store_true", help="skip the 2D pictures")
    ap.add_argument("--seeds", type=int, default=1,
                    help="runs per random method (seeds 0..N-1); 1 = one run per cell")
    args = ap.parse_args()

    global N_BIG
    seeds, iters = SEEDS[:args.seeds], K
    if args.quick:
        N_BIG, seeds, iters = 10, SEEDS[:2], 50
        args.gammas = [0.1]
        args.out = os.path.join(args.out, "quick")
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "cases_runs.csv")
    xlsx_path = os.path.join(args.out, "cases_results.xlsx")

    # the pictures go into docs/runs (committed, so the README can show them);
    # the quick test keeps its throw-away pictures in results/quick
    pic_dir = None if args.no_pictures else (
        os.path.join(args.out, "figures", "runs") if args.quick
        else os.path.join(os.path.dirname(HERE), "docs", "runs"))
    jobs = make_jobs(args.rows, args.functions, args.gammas, args.starts, seeds, iters, pic_dir)
    done = load_done(csv_path)

    def picture_missing(j):
        return pic_dir and wants_picture(j[1], j[4]) and not os.path.exists(
            pictures.picture_path(pic_dir, j[1], j[2], j[3], METHODS[j[0]][3]))

    todo = [j for j in jobs if (j[0], j[1], j[2], j[3], j[4]) not in done]
    redraw = [j for j in jobs if (j[0], j[1], j[2], j[3], j[4]) in done and picture_missing(j)]
    print(f"{len(jobs)} runs in these cases, {len(jobs) - len(todo)} already saved, "
          f"{len(todo)} to go, on {args.workers} cores")

    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fh, Pool(args.workers, initializer=_set_size,
                                                    initargs=(N_BIG,)) as pool:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        t0 = time.time()
        for i, line in enumerate(pool.imap_unordered(one_run, todo), start=1):
            writer.writerow(line)
            fh.flush()                                 # saved at once: safe to stop any time
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done, {time.time() - t0:.0f} s")
        if redraw:                                     # saved runs whose picture is missing
            print(f"drawing {len(redraw)} missing pictures (running those 2D runs again)")
            for _ in pool.imap_unordered(one_run, redraw):
                pass

    write_excel(xlsx_path, load_done(csv_path), args.functions, args.gammas, args.starts)
    print("wrote", csv_path, "and", xlsx_path)
    if pic_dir:
        print("pictures in", pic_dir)


def _set_size(n):
    """Each worker process needs the same 100D size (the quick test makes it smaller)."""
    global N_BIG
    N_BIG = n


if __name__ == "__main__":
    main()
