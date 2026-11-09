"""Evaluation of versioned candidates without teacher actions or weight updates."""
import json
from pathlib import Path
from .envs import EnvConfig,ThrusterLandingEnv
from .rl_evaluation import episode_metrics
from .evaluation import write_csv


def evaluate_candidate(agent,manifest,budget=None,trace=False,render=False):
    env=ThrusterLandingEnv(EnvConfig(),render_mode='human' if render else None);rows=[];traces=[]
    try:
        for s in manifest['scenarios']:
            obs,info=env.reset(seed=s['seed'],options=s['options']);initial=info['state'];ret=disc=0.;parts={};steps=0
            if trace:traces.append(dict(scenario_id=s['id'],time=0.,action=-1,outcome='flying',**initial))
            while True:
                action=agent.act(obs,explore=False)
                obs,r,t,tr,info=budget.step(env,action) if budget is not None else env.step(action);ret+=r;disc+=agent.config['gamma']**steps*r;steps+=1
                for key,value in info['reward_components'].items():parts[key]=parts.get(key,0.)+value
                if trace:traces.append(dict(scenario_id=s['id'],time=info['time'],action=action,outcome=info['outcome'],**info['state']))
                if t or tr:
                    row=episode_metrics(info,initial,ret,steps,s,agent.config['algorithm']+'_assisted',manifest['split']);row.update(discounted_return=disc,terminated=t,truncated=tr,contact=info['contact'],gas_exhausted=info['state']['gas']<=1e-8,**{'reward_'+k:v for k,v in parts.items()});rows.append(row);break
    finally:env.close()
    return rows,traces


def export_candidate(path,rows,traces=None):
    path=Path(path);write_csv(path,rows)
    if traces is not None:write_csv(path.with_suffix('.traces.csv'),traces)
