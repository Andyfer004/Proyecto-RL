"""Load a candidate and evaluate or visualize without teacher actions."""
import argparse,json
from pathlib import Path
from thruster_landing.recovery_agent import AssistedDQN
from thruster_landing.recovery_evaluation import evaluate_candidate,export_candidate
from thruster_landing.recovery_budget import Budget
from thruster_landing.experiments import assert_disjoint,load_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--scenarios',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--episodes',type=int);p.add_argument('--render',action='store_true');p.add_argument('--count-budget',action='store_true');a=p.parse_args()
    manifest=load_manifest(a.scenarios,'validation')
    agent,meta=AssistedDQN.load(a.checkpoint)
    assert_disjoint(load_manifest(agent.config.get('train_manifest','configs/recovery_train_scenarios.json'),'train'),manifest)
    if a.episodes is not None:manifest={**manifest,'scenarios':manifest['scenarios'][:a.episodes]}
    if a.output.exists():raise ValueError('Usar salida nueva')
    budget=Budget('loaded_check_'+a.output.stem) if a.count_budget else None
    before={k:v.clone() for k,v in agent.online.state_dict().items()}
    try:
        rows,traces=evaluate_candidate(agent,manifest,budget,trace=True,render=a.render)
        for k,v in agent.online.state_dict().items():assert (v==before[k]).all()
        export_candidate(a.output,rows,traces)
        a.output.with_suffix('.metadata.json').write_text(json.dumps(dict(checkpoint=str(a.checkpoint),checkpoint_metadata=meta,manifest=str(a.scenarios),loaded_online_steps=agent.env_steps,assisted=True,teacher_calls=0,exploration=False,updates_during_evaluation=0,successes=sum(r['success'] for r in rows),episodes=len(rows)),indent=2)+'\n')
        print('Loaded policy',a.output,sum(r['success'] for r in rows),'/',len(rows),flush=True)
    finally:
        if budget is not None:budget.close()
if __name__=='__main__':main()
