"""PID demonstrations only on new training cases; grouped holdout stays train-only."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from thruster_landing.envs import EnvConfig,ThrusterLandingEnv
from thruster_landing.agents.pid import PIDConfig,PIDController
from thruster_landing.evaluation import write_csv
from thruster_landing.recovery_agent import landing_features
from thruster_landing.recovery_budget import Budget,ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=Path('configs/recovery_assisted_v1.json'));args=parser.parse_args()
    cfg=json.loads(args.config.read_text());out=ROOT/cfg.get('demo_directory','demonstrations_v1');out.mkdir(exist_ok=False);cases=json.loads(Path(cfg.get('train_manifest','configs/recovery_train_scenarios.json')).read_text())['scenarios'][:cfg['demo_episodes']]
    b=Budget(cfg['version']+'_PID_training_demonstrations');env=ThrusterLandingEnv(EnvConfig());pid=PIDController(env.config,PIDConfig.from_json('configs/pid.json'));data=[];episodes=[];start=time.perf_counter()
    try:
        for episode,s in enumerate(cases):
            obs,info=env.reset(seed=s['seed'],options=s['options']);pid.reset();trajectory=[];ret=0.
            while True:
                action=pid.act(env.state,info['wind_force']);next_obs,r,t,tr,info=b.step(env,action);trajectory.append((obs.copy(),action,float(r),next_obs.copy(),t,tr));ret+=r;obs=next_obs
                if t or tr:break
            episodes.append(dict(episode=episode,scenario_id=s['id'],seed=s['seed'],success=info['is_success'],outcome=info['outcome'],steps=len(trajectory),return_=ret,duration_s=info['time'],gas_consumed_Ns=info['gas_consumed']))
            if info['is_success']:
                for i,(obs,a,r,next_obs,t,tr) in enumerate(trajectory):
                    stop=min(i+cfg['n_step'],len(trajectory));future=trajectory[stop-1];nr=sum(cfg['gamma']**k*trajectory[i+k][2] for k in range(stop-i))
                    data.append(dict(observations=landing_features(obs,cfg),actions=a,rewards=r,next_observations=landing_features(next_obs,cfg),terminated=t,n_next_observations=landing_features(future[3],cfg),n_rewards=nr,n_gamma=cfg['gamma']**(stop-i),n_terminated=future[4],train_mask=episode%8!=0,episode_ids=episode))
            if (episode+1)%128==0:print('demos',episode+1,'successes',sum(e['success'] for e in episodes),'transitions',len(data),flush=True)
        result={}
        for key in data[0]:
            values=np.asarray([d[key] for d in data]);dtype=torch.bool if key in ['terminated','n_terminated','train_mask'] else torch.long if key in ['actions','episode_ids'] else torch.float32
            result[key]=torch.as_tensor(values,dtype=dtype)
        torch.save(result,out/'demonstrations.pt');write_csv(out/'episodes.csv',episodes)
        (out/'metadata.json').write_text(json.dumps(dict(teacher='Frozen PID',split='training only',episodes=len(episodes),successful_episodes=sum(e['success'] for e in episodes),transitions=len(data),heldout_episode_rule='episode_index%8==0; never sampled for learning',discarded_failed_episodes=True,observation_version=cfg['observation_version'],wall_seconds=time.perf_counter()-start),indent=2)+'\n')
    finally:env.close();b.close()
if __name__=='__main__':main()
