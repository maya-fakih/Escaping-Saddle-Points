"""
Only what the assignment asks for, in its one-table format.

    python experiments/run_hw.py               # about 300 short runs, all 2D
    python experiments/run_hw.py --workers 4
    python experiments/run_hw.py --pgd T=50 --gdnc g_trig=1e-4    # after Question 4

--pgd / --gdnc: the configurations you CHOSE in Question 4. Section 4 of the assignment
compares the proposed method on S1-S5 against GD and these two choices, so they get their
own S1-S5 rows ("S3, chosen: T=50").

WHAT IT RUNS
  Questions 1-3 and the comparison (section 4):  S1-S5  x  GD, P-GD, GD-NC (defaults),
      Jin et al. PGD, and the proposed method: LDH and LDH with a guessed B.
  Question 4 (on S2 only, one parameter at a time, the others at their defaults):
      P-GD   g_trig = 1e-4, 1e-2    r = 1e-3, 5e-2    T = 10, 50
      GD-NC  g_trig = 1e-4, 1e-2
      (the default value of each is the "S2, default" row, reused as the assignment says)
  Deterministic methods run once, random ones 10 times (seeds 0-9), K = 3000 updates.

WHAT IT WRITES (into results/)
  hw_runs.csv       one line per run
  hw_results.xlsx   the assignment's table: Setting / parameters, Method, Success runs /
                    total, k_esc, k_succ, Time to success, Final gap, Work Nf / Ng / NH
                    (mean ± sample sd; "NR" when never reached)
"""
import argparse
import csv
import inspect
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # so "saddle" can be imported
sys.path.insert(0, HERE)

from saddle.algorithms import load                 # noqa: E402
from saddle.benchmark import RotatedSaddle         # noqa: E402
from saddle.metrics import evaluate                # noqa: E402
from run_cases import FIELDS, num, mean_sd          # noqa: E402

K = 3000
SEEDS = list(range(10))

# The assignment's five cases: gamma and the start (see RotatedSaddle.start)
CASES = {"S1": (1.0, "close"), "S2": (0.1, "close"), "S3": (0.01, "close"),
         "S4": (1.0, "on"), "S5": (1.0, "near_min")}

# The methods: (name in the table, file in saddle/algorithms, random?)
METHODS = {"GD": ("gd", False), "P-GD": ("pgd", True), "GD-NC": ("gdnc", False),
           "Jin et al. PGD": ("jin_pgd", True), "LDH (proposed)": ("ldh", True),
           "LDH, guessed B": ("ldh_guessed", True)}
CHOSEN = {"P-GD": {}, "GD-NC": {}}                  # filled from --pgd / --gdnc


def experiments():
    """Every table row: (case, parameters shown, method, settings)."""
    rows = []
    for case in CASES:                                     # Questions 1-3 + section 4
        for method in METHODS:
            rows.append((case, "default", method, {}))
    for key, values in (("g_trig", (1e-4, 1e-2)), ("r", (1e-3, 5e-2)), ("T", (10, 50))):
        for v in values:                                   # Question 4, P-GD
            rows.append(("S2", f"{key} = {v:g}", "P-GD", {key: v}))
    for v in (1e-4, 1e-2):                                 # Question 4, GD-NC
        rows.append(("S2", f"g_trig = {v:g}", "GD-NC", {"g_trig": v}))
    for method, settings in CHOSEN.items():                # section 4: the chosen configs
        if settings:
            shown = ", ".join(f"{k}={v:g}" for k, v in settings.items())
            for case in CASES:
                rows.append((case, f"chosen: {shown}", method, dict(settings)))
    return rows


def parse_settings(pairs):
    """['T=50', 'r=0.05'] -> {'T': 50, 'r': 0.05}"""
    out = {}
    for pair in pairs or []:
        key, value = pair.split("=")
        out[key] = int(value) if key == "T" else float(value)
    return out


def one_run(job):
    """Run one method once on one case; return its numbers (one CSV line)."""
    i, seed = job
    case, _, method, settings = experiments()[i]
    gamma, start = CASES[case]
    problem = RotatedSaddle.plain(gamma)
    cls = load(METHODS[method][0])
    kwargs = dict(settings, max_iters=K)
    if "seed" in inspect.signature(cls.__init__).parameters:
        kwargs["seed"] = seed
    t0 = time.perf_counter()
    algo = cls(problem, problem.start(start), **kwargs)
    algo.run()
    m = evaluate(algo)
    m.update(row=i, function="2d", gamma=gamma, start=start, seed=seed,
             seconds=round(time.perf_counter() - t0, 3))
    return {k: m.get(k) for k in FIELDS}


def table_row(case, params, method, runs):
    """One line of the assignment's table, as text."""
    get = lambda key: [num(r[key]) for r in runs]
    n = len(runs)
    succ = sum(v is not None for v in get("k_succ"))
    esc = [v for v in get("k_esc") if v is not None]
    escape = "–" if case == "S5" else mean_sd(esc)          # Q3: omit escape near a minimum
    if case != "S5" and 0 < len(esc) < n:
        escape += f" ({len(esc)}/{n} escape)"
    work = " / ".join(mean_sd(get(k)) for k in ("nf", "ng", "nh"))
    return [f"{case}, {params}", method, f"{succ}/{n}", escape, mean_sd(get("k_succ")),
            mean_sd(get("t_succ"), "{:.4f}"), mean_sd(get("q"), "{:.1e}"), work]


HEADER = ["Setting / parameters", "Method", "Success runs / total", "Escape k_esc",
          "Success k_succ", "Time to success (s)", "Final gap", "Work Nf / Ng / NH"]


def write_excel(path, lines):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    wb = Workbook()
    ws = wb.active
    ws.title = "Results"
    ws.append(HEADER)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDEBF7")
        c.alignment = Alignment(wrap_text=True, vertical="center")
    for line in lines:
        ws.append(line)
    for col, width in zip("ABCDEFGH", (20, 16, 12, 22, 16, 18, 18, 30)):
        ws.column_dimensions[col].width = width
    ws.append([])
    ws.append(["mean ± sample sd over the runs that reach it; NR = never reached; "
               "random methods: seeds 0-9; K = 3000 updates"])
    wb.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="CPU cores to use")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "results"))
    ap.add_argument("--pgd", nargs="+", metavar="KEY=VALUE",
                    help="your chosen P-GD configuration from Question 4, e.g. T=50")
    ap.add_argument("--gdnc", nargs="+", metavar="KEY=VALUE",
                    help="your chosen GD-NC configuration from Question 4, e.g. g_trig=1e-4")
    args = ap.parse_args()
    CHOSEN["P-GD"], CHOSEN["GD-NC"] = parse_settings(args.pgd), parse_settings(args.gdnc)
    os.makedirs(args.out, exist_ok=True)

    rows = experiments()
    jobs = [(i, s) for i, (_, _, m, _) in enumerate(rows)
            for s in (SEEDS if METHODS[m][1] else SEEDS[:1])]
    print(f"{len(rows)} table rows, {len(jobs)} runs")
    t0 = time.time()
    with Pool(args.workers, initializer=_set_chosen, initargs=(CHOSEN,)) as pool:
        results = pool.map(one_run, jobs)
    print(f"done in {time.time() - t0:.0f} s")

    csv_path = os.path.join(args.out, "hw_runs.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(results)

    lines = []
    for i, (case, params, method, _) in enumerate(rows):
        runs = [{k: ("" if v is None else v) for k, v in r.items()} for r in results if r["row"] == i]
        lines.append(table_row(case, params, method, runs))
    xlsx_path = os.path.join(args.out, "hw_results.xlsx")
    write_excel(xlsx_path, lines)
    print("wrote", csv_path, "and", xlsx_path)


def _set_chosen(chosen):
    """Worker processes need the same chosen configurations to rebuild the row list."""
    CHOSEN.update(chosen)


if __name__ == "__main__":
    main()
