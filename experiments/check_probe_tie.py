"""Test the no-kick, tied-probe variant; retain all earlier results unchanged."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import csv
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from saddle.algorithms.ldh_probe_tie import LDHProbeTie
from saddle.benchmark import RotatedSaddle
from saddle.metrics import evaluate
from experiments import compare_100d as export

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/probe_tie_check_local')
    args=parser.parse_args()
    out=args.out
    if (out/'runs.csv').exists():
        raise SystemExit('Results exist; retain them rather than overwrite.')
    out.mkdir(parents=True,exist_ok=True)
    fields=export.FIELDS+['tie_walks','tie_walks_to_success','dimension','angle']
    rows=[]
    for n,angle,starts,seeds in [(2,0,['on'],[0]),(2,45,['on'],[0]),
                                (100,None,['close','on'],[0,1])]:
        for gamma in (1.,.1,.01):
            for start in starts:
                for seed in seeds:
                    problem=RotatedSaddle.random(n,gamma,seed=0) if n==100 else RotatedSaddle.rotated_2d(gamma,angle)
                    a=LDHProbeTie(problem,problem.start(start),R=20 if n==100 else 3,
                                  seed=seed,max_iters=3000)
                    a.run();m=evaluate(a)
                    row={key:m.get(key) for key in fields}
                    row.update(gamma=gamma,start=start,method=a.name,seed=seed,
                               optimizer_seconds=a.clock,search_rounds=len(a.probe_log),
                               kick_count=len(a.kicks),gave_up=bool(a.giveups),
                               tie_walks=len(a.tie_walks),dimension=n,angle=angle,
                               tie_walks_to_success=None if m['k_succ'] is None else
                               sum(k<m['k_succ'] for k in a.tie_walks))
                    rows.append(row)
                    np.savez_compressed(out/f'n{n}_angle{angle}_g{gamma:g}_{start}_seed{seed}.npz',
                                        z=a.path,times=a.times,work=a.work,
                                        probe_log=np.asarray(a.probe_log,dtype=str))
                    print(n,angle,gamma,start,seed,'escape',m['k_esc'],'success',m['k_succ'],
                          'tie walks',len(a.tie_walks),'random kicks',len(a.kicks),flush=True)
    with (out/'runs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    export.FIELDS=fields
    # Split dimensional settings to avoid pooling different landscapes in Summary.
    for row in rows:
        row['method']=f"{row['method']}; n={row['dimension']}, angle={row['angle']}"
    config=dict(tie_tol=1e-12,rule='No descending selected probe: move only if ALL raw probe objective values tie within absolute tolerance; otherwise stay and count failure.',
                R_2D=3,R_100D=20,rotation_seed_100D=0,K=3000,
                note='No random kicks. Two 100D seeds repeat an identical deterministic trajectory; they are timing repeats. Earlier random-kick LDH remains unchanged.')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    export.write_excel(rows,config,out/'probe_tie_check.xlsx')

if __name__=='__main__':main()
