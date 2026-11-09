"""Assisted DQN first, version1. No PID queries during online interaction/evaluation."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from thruster_landing.recovery_agent import AssistedDQN
from thruster_landing.recovery_evaluation import evaluate_candidate,export_candidate
from thruster_landing.recovery_budget import Budget,ROOT
from thruster_landing.envs import EnvConfig,ThrusterLandingEnv
from thruster_landing.rl_evaluation import episode_metrics
from thruster_landing.evaluation import write_csv


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--algorithm',choices=['dqn','double_dqn'],default='dqn');p.add_argument('--seed',type=int,default=42);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=Path('configs/recovery_assisted_v1.json'));args=p.parse_args()
    cfg=json.loads(args.config.read_text());cfg.update(algorithm=args.algorithm,seed=args.seed);args.output.mkdir(parents=True,exist_ok=False)
    b=Budget(f"{cfg['version']}_{args.algorithm}_seed{args.seed}");start=time.perf_counter();torch.use_deterministic_algorithms(True)
    demos=torch.load(ROOT/cfg.get('demo_directory','demonstrations_v1')/'demonstrations.pt',weights_only=True);agent=AssistedDQN(cfg,demos);dev=json.loads(Path(cfg.get('development_manifest','configs/recovery_development_scenarios.json')).read_text());train=json.loads(Path(cfg.get('train_manifest','configs/recovery_train_scenarios.json')).read_text())['scenarios'];case_rng=np.random.default_rng(args.seed+100)
    evaluations=[];training=[];metrics=[];best=None;streak=0;achieved=False
    env=ThrusterLandingEnv(EnvConfig())
    def assess(phase):
        nonlocal best,streak,achieved
        rows,traces=evaluate_candidate(agent,dev,budget=b,trace=True)
        for row in rows:row.update(phase=phase,online_steps=agent.env_steps,gradient_updates=agent.updates)
        evaluations.extend(rows);score=sum(r['success'] for r in rows)/len(rows)
        export_candidate(args.output/f'{phase}_{agent.env_steps:08d}.csv',rows,traces)
        group_rates={name:sum(r['success'] for r in rows if r['condition']==name)/sum(r['condition']==name for r in rows) for name in cfg.get('goal_groups',[])}
        goal_score=min(group_rates.values()) if group_rates else score
        successes=[r for r in rows if r['success']];rank=(goal_score,-np.mean([r['contact_error_x_m'] for r in successes]) if successes else -float('inf'))
        if best is None or rank>best:
            best=rank;agent.save(args.output/'best.policy.pt',dict(phase=phase,online_steps=agent.env_steps,development_success_rate=score,assisted=True,teacher_actions_at_eval=False))
        agent.save(args.output/'last.policy.pt',dict(phase=phase,development_success_rate=score,assisted=True,teacher_actions_at_eval=False))
        if phase=='online':streak=streak+1 if goal_score>=cfg['success_threshold'] else 0
        achieved=streak>=cfg['success_streak'] and agent.env_steps>=cfg['minimum_online_steps']
        print(args.algorithm,args.seed,phase,'steps',agent.env_steps,'updates',agent.updates,'dev',sum(r['success'] for r in rows),'/',len(rows),'groups',group_rates,'streak',streak,flush=True)
        write_csv(args.output/'evaluations.csv',evaluations);write_csv(args.output/'metrics.csv',metrics);write_csv(args.output/'episodes.csv',training)
    try:
        for update in range(cfg['pretrain_classification_updates']):
            if update%100==0:b.guard()
            loss=agent.classification_update()
            if (update+1)%1000==0:
                with torch.no_grad():
                    mask=~demos['train_mask'];pred=agent.online(demos['observations'][mask]).argmax(1);accuracy=float((pred==demos['actions'][mask]).float().mean())
                metrics.append(dict(phase='classification',updates=agent.updates,loss=loss,heldout_action_accuracy=accuracy));print('BC',update+1,'loss',loss,'heldout',accuracy,flush=True)
        agent.sync();assess('classification')
        for update in range(cfg['pretrain_td_updates']):
            if update%100==0:b.guard()
            m=agent.learn(offline=True)
            if (update+1)%cfg['target_update_frequency']==0:agent.sync()
            if (update+1)%1000==0:metrics.append(dict(phase='offline_td',updates=agent.updates,**m))
        assess('offline_td')
        obs=None;episode_return=0.;episode_steps=0;parts={};initial=None;scenario=None
        while agent.env_steps<cfg['total_online_steps']:
            if obs is None:
                scenario=train[int(case_rng.integers(len(train)))];obs,info=env.reset(seed=scenario['seed'],options=scenario['options']);initial=info['state'];episode_return=0.;episode_steps=0;parts={}
            a=agent.act(obs,explore=True);nobs,r,t,tr,info=b.step(env,a);agent.add(obs,a,r,nobs,t,tr);agent.env_steps+=1;episode_steps+=1;episode_return+=r;obs=nobs
            for key,value in info['reward_components'].items():parts[key]=parts.get(key,0.)+value
            if agent.env_steps%4==0 and len(agent.replay)>=cfg['batch_size']:
                m=agent.learn()
                if agent.env_steps%1000==0:metrics.append(dict(phase='online',online_steps=agent.env_steps,updates=agent.updates,**m))
            if agent.env_steps%cfg['target_update_frequency']==0:agent.sync()
            if t or tr:
                row=episode_metrics(info,initial,episode_return,episode_steps,scenario,args.algorithm+'_assisted','train');row.update(online_steps=agent.env_steps,**{'reward_'+k:v for k,v in parts.items()});training.append(row);obs=None
            if agent.env_steps%cfg['validation_frequency']==0:
                assess('online')
                if achieved:break
        agent.save(args.output/'last.policy.pt',dict(phase='online',aspirational_goal=achieved,streak=streak,assisted=True,teacher_actions_at_eval=False))
        write_csv(args.output/'evaluations.csv',evaluations);write_csv(args.output/'metrics.csv',metrics);write_csv(args.output/'episodes.csv',training)
        (args.output/'metadata.json').write_text(json.dumps(dict(config=cfg,achieved=achieved,streak=streak,online_steps=agent.env_steps,gradient_updates=agent.updates,wall_seconds=time.perf_counter()-start,assisted=True,demonstrations_source=str(ROOT/cfg.get('demo_directory','demonstrations_v1')/'demonstrations.pt'),reward_changed=False,observation_version=cfg['observation_version'],partial_episode_steps=episode_steps if obs is not None else 0),indent=2)+'\n')
    finally:env.close();b.close()
if __name__=='__main__':main()
