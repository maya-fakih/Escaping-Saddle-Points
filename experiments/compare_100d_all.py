"""Same-machine four-method comparison, with LDH retry cap 20 by default."""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import statistics
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from saddle.algorithms.ldh import LDH
from saddle.algorithms.gdnc import GDNC
from saddle.algorithms.pgd import PGD
from saddle.algorithms.jin_pgd import JinPGD
from saddle.benchmark import RotatedSaddle
from saddle.metrics import evaluate
from experiments import compare_100d as export

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--k', type=int, default=3000)
    parser.add_argument('--out', type=Path, default=ROOT/'results/comparison_100d_all_R20')
    args = parser.parse_args()
    if args.rounds < 1 or args.k < 1:
        parser.error('rounds and k must be positive')
    if (args.out/'runs.csv').exists():
        parser.error('Output already exists; choose another --out directory.')
    args.out.mkdir(parents=True, exist_ok=True)
    history = args.out/'trajectories'
    history.mkdir(exist_ok=True)
    export.FIELDS = export.FIELDS + ['rounds_to_success', 'kicks_to_success',
                                     'max_failed_streak_to_success']
    config = dict(dimension=100, rotation_seed=0, gammas=[1., .1, .01],
                  starts=['close', 'on'], seeds=args.seeds, K=args.k, LDH_R=args.rounds,
                  timing='Same machine, single BLAS thread; optimizer updates only, diagnostics and initialization excluded.',
                  deterministic_note='GD-NC seeds are timing repeats; its trajectory is deterministic.',
                  retries_note='R caps consecutive failed rounds per search episode, not total lifetime rounds. Counts to success exclude later searches.',
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((ROOT/'saddle').rglob('*.py'))})
    (args.out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    rows=[]
    # Warm up the dense eigen-solver outside the recorded runs.
    np.linalg.eigh(np.eye(100))
    with (args.out/'runs.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=export.FIELDS)
        writer.writeheader()
        for gamma in config['gammas']:
            for start in config['starts']:
                for name, cls in [('LDH-GD', LDH), ('Assignment P-GD', PGD),
                                  ('Jin-inspired P-GD', JinPGD), ('GD-NC', GDNC)]:
                    for seed in args.seeds:
                        problem=RotatedSaddle.random(100,gamma,seed=0)
                        kwargs=dict(max_iters=args.k)
                        if cls is not GDNC: kwargs['seed']=seed
                        if cls is LDH: kwargs['R']=args.rounds
                        algorithm=cls(problem,problem.start(start),**kwargs)
                        streaks=[]
                        if cls is LDH:
                            original=algorithm._search
                            def traced_search():
                                prior=algorithm.fails
                                original()
                                outcome=algorithm.probe_log[-1][1]
                                streaks.append((algorithm.k, 0 if outcome=='escaped' else prior+1))
                            algorithm._search=traced_search
                        algorithm.run()
                        metrics=evaluate(algorithm)
                        success=metrics['k_succ']
                        row={key:metrics.get(key) for key in export.FIELDS}
                        row.update(gamma=gamma,start=start,method=name,seed=seed,
                                   search_rounds=len(algorithm.probe_log),
                                   kick_count=len(getattr(algorithm,'kicks',[])),
                                   gave_up=any(o=='gave up' for _,o in algorithm.probe_log),
                                   optimizer_seconds=algorithm.clock,
                                   rounds_to_success=None if success is None else sum(k<success for k,_ in algorithm.probe_log),
                                   kicks_to_success=None if success is None else sum(k<success for k in getattr(algorithm,'kicks',[])),
                                   max_failed_streak_to_success=None if success is None else max([v for k,v in streaks if k<success],default=0))
                        np.savez_compressed(history/f'g{gamma:g}_{start}_{cls.__name__}_seed{seed}.npz',
                                            z=algorithm.path,times=algorithm.times,work=algorithm.work,
                                            probe_log=np.asarray(algorithm.probe_log,dtype=str),
                                            failed_streaks=np.asarray(streaks,dtype=int))
                        rows.append(row);writer.writerow(row);stream.flush()
                        print(f'{gamma:g} {start} {name} seed={seed}: success={success}, time={metrics["t_succ"]}, rounds={row["rounds_to_success"]}, streak={row["max_failed_streak_to_success"]}',flush=True)
    export.write_excel(rows,config,args.out/'comparison_100d_all.xlsx')
    lines=['# Same-machine 100D comparison (LDH R='+str(args.rounds)+')','',
           config['timing'],config['deterministic_note'],config['retries_note'],'',
           'Success-conditioned means +/- sample SD; NR means no successful runs. Full details in Excel and CSV.','',
           '| gamma | start | method | success | updates | seconds | Nf | Ng | NH | rounds to success |',
           '|---|---|---|---|---|---|---|---|---|---|']
    groups={}
    for row in rows: groups.setdefault((row['gamma'],row['start'],row['method']),[]).append(row)
    for (g,s,m),group in groups.items():
        ok=[r for r in group if r['k_succ'] is not None]
        cells=[str(g),s,m,f'{len(ok)}/{len(group)}']
        for key in ('k_succ','t_succ','nf_succ','ng_succ','nh_succ','rounds_to_success'):
            cells.append(export.mean_sd([r[key] for r in ok],key=='t_succ'))
        lines.append('| '+' | '.join(cells)+' |')
    (args.out/'summary.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    main()
