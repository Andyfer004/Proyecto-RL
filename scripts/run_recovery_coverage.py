"""Reproduce the frozen coverage method from scratch, in a NEW budgeted directory.
This reproduces intervention2, not a new search or a multi-seed campaign.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-seconds',type=int,default=7200)
    p.add_argument('--max-steps',type=int,default=5000000)
    a=p.parse_args()
    if not 0<a.max_seconds<=7200 or not 0<a.max_steps<=5000000:
        p.error('Maximum7200seconds and5000000steps')
    a.output.mkdir(parents=True,exist_ok=False)
    now=time.time();budget=a.output/'budget.json'
    budget.write_text(json.dumps(dict(start_epoch=now,deadline_epoch=now+a.max_seconds,
        wall_limit_seconds=a.max_seconds,step_limit=a.max_steps,actual_steps=0,stages=[],
        interventions=[dict(id=2,method='assisted_dqn_coverage_v2',scope='Frozen reproduction, no tuning')]),indent=2)+'\n')
    paths=['configs/recovery_assisted_v2.json']+[f'configs/recovery_v2_{s}_scenarios.json' for s in ['train','development','check']]
    (a.output/'frozen_coverage_v2_protocol.json').write_text(json.dumps(dict(sha256={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in paths}),indent=2)+'\n')
    env=os.environ.copy();env['THRUSTER_RECOVERY_ROOT']=str(a.output)
    env['MPLCONFIGDIR']=str(a.output/'matplotlib_cache')
    env['PYTHONPATH']=str(Path('scripts').resolve())+os.pathsep+env.get('PYTHONPATH','')
    def run(name,*args):
        if time.time()>=now+a.max_seconds:raise RuntimeError('Wall budget exhausted')
        print(name,flush=True)
        with (a.output/(name+'.log')).open('w') as log:
            subprocess.run([sys.executable,*args],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    run('tests','-m','pytest','-p','recovery_pytest_budget','-q')
    run('checker','scripts/check_recovery_env.py','--count-budget')
    run('demonstrations','scripts/collect_recovery_demos.py','--config',paths[0])
    methods=[('dqn_coverage_v2_seed42','coverage42','dqn',42),
             ('double_coverage_v2_seed42','coverage_double42','double_dqn',42),
             ('dqn_coverage_v2_seed43','coverage43','dqn',43)]
    for directory,prefix,algorithm,seed in methods:
        run(directory,'scripts/train_recovery.py','--config',paths[0],'--algorithm',algorithm,'--seed',str(seed),'--output',str(a.output/directory))
        meta=json.loads((a.output/directory/'metadata.json').read_text())
        if prefix=='coverage42' and not meta['achieved']:
            raise RuntimeError('No sustained useful DQN: stop and inspect before transferring')
        for split,manifest in [('development',paths[2]),('check',paths[3]),('original','configs/rl_validation_scenarios.json')]:
            run(prefix+'_'+split,'scripts/evaluate_recovery.py','--checkpoint',str(a.output/directory/'best.policy.pt'),
                '--scenarios',manifest,'--output',str(a.output/f'loaded_{prefix}_{split}.csv'),'--count-budget')
    run('report','scripts/report_recovery_coverage.py')
    record=json.loads(budget.read_text());record.update(status='complete',finished_epoch=time.time())
    budget.write_text(json.dumps(record,indent=2)+'\n')
    print('Finished:',a.output,flush=True)

if __name__=='__main__':main()
