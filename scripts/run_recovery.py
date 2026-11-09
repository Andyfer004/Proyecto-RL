"""Reproduce the bounded assisted recovery protocol in a NEW output directory."""
import argparse,json,os,subprocess,sys,time,hashlib
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-seconds',type=int,default=7200);p.add_argument('--max-steps',type=int,default=5000000);a=p.parse_args()
    if not 0<a.max_seconds<=7200 or not 0<a.max_steps<=5000000:p.error('Limits must be positive and no greater than7200s/5000000steps')
    a.output.mkdir(parents=True,exist_ok=False);now=time.time();budget=a.output/'budget.json'
    budget.write_text(json.dumps(dict(start_epoch=now,deadline_epoch=now+a.max_seconds,wall_limit_seconds=a.max_seconds,step_limit=a.max_steps,actual_steps=0,stages=[],interventions=[dict(id=1,method='assisted_dqn_v1',teacher='PID demonstrations from train only')]),indent=2)+'\n')
    frozen=['configs/recovery_assisted_v1.json',*['configs/recovery_'+s+'_scenarios.json' for s in ['train','development','check']]]
    (a.output/'frozen_development_protocol.json').write_text(json.dumps(dict(sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in frozen},roles='Train and demo-holdout separate from development and check; fixed before training.'),indent=2)+'\n')
    env=os.environ.copy();env['THRUSTER_RECOVERY_ROOT']=str(a.output);env['MPLCONFIGDIR']=str(a.output/'matplotlib_cache')
    def run(name,*args):
        if time.time()>=now+a.max_seconds:raise RuntimeError('Wall budget exhausted')
        print('START',name,flush=True)
        with (a.output/(name+'.log')).open('w') as log:subprocess.run([sys.executable,*args],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    env['PYTHONPATH']=str(Path('scripts').resolve())+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    run('tests','-m','pytest','-p','recovery_pytest_budget','-q');run('checker','scripts/check_recovery_env.py','--count-budget')
    run('audit','scripts/audit_recovery.py');run('cartpole','scripts/benchmark_recovery_cartpole.py');run('demos','scripts/collect_recovery_demos.py')
    run('dqn42','scripts/train_recovery.py','--output',str(a.output/'dqn_assisted_seed42'))
    metadata=json.loads((a.output/'dqn_assisted_seed42/metadata.json').read_text())
    if not metadata['achieved']:raise RuntimeError('No sustained useful DQN: stop before Double DQN; inspect this attempt')
    def evaluate(directory,prefix,checkpoint):
        for split,manifest in [('development','configs/recovery_development_scenarios.json'),('check','configs/recovery_check_scenarios.json'),('original','configs/rl_validation_scenarios.json')]:
            run('eval_'+prefix+'_'+split,'scripts/evaluate_recovery.py','--checkpoint',str(a.output/directory/checkpoint),'--scenarios',manifest,'--output',str(a.output/f'loaded_{prefix}_{split}.csv'),'--count-budget')
    evaluate('dqn_assisted_seed42','dqn','best.policy.pt')
    run('matched','scripts/evaluate_recovery.py','--checkpoint',str(a.output/'dqn_assisted_seed42/best.policy.pt'),'--scenarios','configs/level1_validation_scenarios.json','--episodes','1','--output',str(a.output/'loaded_primary_matched.csv'),'--count-budget')
    run('double42','scripts/train_recovery.py','--algorithm','double_dqn','--output',str(a.output/'double_dqn_assisted_seed42'));evaluate('double_dqn_assisted_seed42','double42','last.policy.pt')
    run('dqn43','scripts/train_recovery.py','--seed','43','--output',str(a.output/'dqn_assisted_seed43'));evaluate('dqn_assisted_seed43','dqn43','last.policy.pt')
    run('report','scripts/report_recovery.py')
    record=json.loads(budget.read_text());record.update(status='complete',finished_epoch=time.time());budget.write_text(json.dumps(record,indent=2)+'\n')
    print('Recovery complete:',a.output,flush=True)
if __name__=='__main__':main()
