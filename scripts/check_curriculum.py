"""Preflight: independent PID/random/off rollouts; no learning or RL replay."""
from pathlib import Path
import argparse
import json
import numpy as np
from thruster_landing.agents.pid import PIDConfig,PIDController
from thruster_landing.envs import EnvConfig,ThrusterLandingEnv
from thruster_landing.envs.physics import State,geometry
from thruster_landing.experiments import prepare_experiment,assert_disjoint,load_manifest
from thruster_landing.rl_evaluation import episode_metrics
from thruster_landing.evaluation import write_csv

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('artifacts/curriculum_preflight'));args=parser.parse_args()
 out=args.output;out.mkdir(parents=True,exist_ok=False)
 cfg=EnvConfig();exp=prepare_experiment(cfg,'configs/curriculum_train_scenarios.json','configs/curriculum_validation_scenarios.json')
 # All previous training states are also distinct; original validation intentionally retained.
 assert_disjoint(exp['train_manifest'],load_manifest('configs/rl_train_scenarios.json','train'))
 rows=[];clearances=[]
 for manifest in [exp['train_manifest'],exp['validation_manifest']]:
  for s in manifest['scenarios']:
   if s['source_level']==0:continue
   state=State(**s['options']['initial_state']);height=float(geometry(state,cfg)[:,1].min())
   assert height>0
   clearances.append(dict(split=manifest['split'],scenario_id=s['id'],source_level=s['source_level'],minimum_clearance_m=height))
 for mode in ['pid','random','off']:
  env=ThrusterLandingEnv(cfg);pid=PIDController(cfg,PIDConfig.from_json('configs/pid.json'))
  for s in exp['validation_manifest']['scenarios']:
   if s['source_level']==0:continue
   rng=np.random.default_rng(s['seed']+500000);pid.reset();obs,info=env.reset(seed=s['seed'],options=s['options']);initial=info['state'];total=0.;steps=0
   while True:
    action=pid.act(State(**info['state']),info['wind_force']) if mode=='pid' else int(rng.integers(8)) if mode=='random' else 0
    obs,reward,t,tr,info=env.step(action);total+=reward;steps+=1
    if t or tr:break
   row=episode_metrics(info,initial,total,steps,s,mode,'preflight_validation');row['source_level']=s['source_level'];rows.append(row)
  env.close()
 write_csv(out/'episodes.csv',rows);write_csv(out/'clearances.csv',clearances)
 for mode in ['pid','random','off']:
  for l in [1,2,3]:
   group=[r for r in rows if r['controller']==mode and r['source_level']==l]
   print(mode,l,sum(r['success'] for r in group),'/',len(group),{k:sum(r['outcome']==k for r in group) for k in set(r['outcome'] for r in group)},flush=True)
 (out/'experiment.json').write_text(json.dumps(exp,indent=2)+'\n')
if __name__=='__main__':main()
