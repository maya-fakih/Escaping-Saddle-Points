"""Small deterministic validation: S1-S5, three rotations, six 100D cases + GD-NC."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from saddle.algorithms.ldh import LDH
from saddle.algorithms.gdnc import GDNC
from saddle.benchmark import Benchmark,RotatedSaddle
from saddle.cases import CASES
from saddle.metrics import evaluate
from experiments import compare_100d as export

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/deterministic_correction_local')
    args=parser.parse_args();out=args.out
    if (out/'runs.csv').exists():raise SystemExit('Results exist; choose a new --out.')
    out.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for case,(gamma,z0) in CASES.items():
        p=Benchmark(gamma);jobs.append((case,2,p,np.array(z0),LDH,3))
    for gamma in (1.,.1,.01):
        p=RotatedSaddle.rotated_2d(gamma,45)
        jobs.append(('45deg-on',2,p,p.start('on'),LDH,3))
    for gamma in (1.,.1,.01):
        for start in ('close','on'):
            for cls in (LDH,GDNC):
                # Fixed benchmark rotation, not a stochastic optimizer seed.
                p=RotatedSaddle.random(100,gamma,seed=0)
                jobs.append((start,100,p,p.start(start),cls,20))
    fields=export.FIELDS+['dimension','walks','walks_to_success','rounds_to_success']
    rows=[];logs={}
    np.linalg.eigh(np.eye(100))
    with (out/'runs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for case,n,p,z0,cls,rounds in jobs:
            kwargs=dict(R=rounds) if cls is LDH else {}
            a=cls(p,z0,max_iters=3000,**kwargs)
            a.run();m=evaluate(a);success=m['k_succ']
            row={k:m.get(k) for k in fields}
            row.update(gamma=p.gamma,start=case,method='LDH-GD deterministic' if cls is LDH else 'GD-NC',
                       seed=None,dimension=n,optimizer_seconds=a.clock,
                       search_rounds=len(a.probe_log),kick_count=len(getattr(a,'kicks',[])),
                       gave_up=any(o=='gave up' for _,o in a.probe_log),walks=len(getattr(a,'walks',[])),
                       walks_to_success=None if success is None else sum(k<success for k in getattr(a,'walks',[])),
                       rounds_to_success=None if success is None else sum(k<success for k,_ in a.probe_log))
            rows.append(row);writer.writerow(row);f.flush()
            label=f'{n}D_g{p.gamma:g}_{case}_{cls.__name__}'
            logs[label]=a.probe_log
            np.savez_compressed(out/(label+'.npz'),z=a.path,times=a.times,work=a.work)
            print(label,'escape',m['k_esc'],'success',success,'time',m['t_succ'],
                  'walks',row['walks'],'kicks',row['kick_count'],flush=True)
    config=dict(K=3000,LDH_R_2D=3,LDH_R_100D=20,optimizer_randomness='None; one trajectory per case',
                rotation='Fixed 100D Q constructed with seed 0; not an optimizer repetition',
                rule='Select minimum objective-change + fading direction penalty among evaluated probes, even uphill; deterministic ties; no separate kick or enclosing trust region.',
                learning='Coordinate scales learn from their own gradient-aligned objective trials each update. Escape probes do not directly update scales or delta.',
                sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (ROOT/'saddle').rglob('*.py')})
    (out/'config.json').write_text(json.dumps(config,indent=2))
    (out/'probe_logs.json').write_text(json.dumps(logs,indent=2))
    export.FIELDS=fields
    export.write_excel(rows,config,out/'deterministic_correction.xlsx')

if __name__=='__main__':main()
