"""Read-only fixed and new-replay Q/TD diagnostics for equal-budget exploration runs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from thruster_landing.beta_diagnostics import measure_groups
from thruster_landing.q_diagnostics import load_model,transition_id
from thruster_landing.offline_diagnostics import subset
from thruster_landing.checkpoints import read_checkpoint
from thruster_landing.experiments import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path('artifacts/exploration_comparison_level1_seed42_100k_v1'));args=parser.parse_args()
    out=args.root/'diagnostics';out.mkdir(exist_ok=False)
    probe_path=Path('artifacts/lr_audit_reference/fixed_probe.pt');probe=torch.load(probe_path,weights_only=True)
    old_meta_path=Path('artifacts/offline_interference_v1/metadata.json');old_meta=json.loads(old_meta_path.read_text())
    samples=[];gradients=[];batches=[];heldout=[];sources=dict(probe_sha256=digest(probe_path),holdout_metadata_sha256=digest(old_meta_path),replays={},checkpoints={})
    def measure(agent,step,batch,ids,source,model,k):
        rows,full,detail=measure_groups(agent,batch,10.)
        common=dict(model=model,env_steps=step,source=source,batch=k)
        gradients.extend({**common,**r} for r in rows);batches.append({**common,**full})
        for i,identifier in enumerate(ids):
            terminal=bool(batch['terminated'][i]);row=dict(**common,transition_id=identifier,group='terminal' if terminal else 'nonterminal',action=int(batch['actions'][i]),successful_terminal=terminal and float(batch['rewards'][i])>0,truncated=bool(batch['truncated'][i]))
            row.update({name:float(value[i]) for name,value in detail.items()})
            row.update(bootstrap=0. if terminal else row['target']-row['reward'],terminal_error_q_minus_r=row['q']-row['reward'] if terminal else None)
            assert not terminal or row['target']==row['reward']
            samples.append(row)
    for alg in ['dqn','double_dqn']:
        oldname=alg+'_lr1e4_100k';old_path=Path('artifacts')/oldname/'last.resume.pt';old=read_checkpoint(old_path,'resume')['agent']['replay']['arrays']
        held=old_meta['manifests'][oldname]['supervised_split']['holdout_indices'];held_batch=subset(old,held)
        runs=[args.root/f'{alg}_{method}' for method in ['uniform','weighted_variant']]
        successful_probe=None
        for source_checkpoint in sorted(runs[1].glob('step_*.resume.pt')):
            source_arrays=read_checkpoint(source_checkpoint,'resume')['agent']['replay']['arrays']
            positive=torch.where(source_arrays['terminated'] & (source_arrays['rewards']>0))[0].tolist()[:32]
            if positive:
                successful_probe=(subset(source_arrays,positive),[transition_id(source_arrays,i) for i in positive])
                sources['replays']['E_first_successes_'+alg]=dict(checkpoint=str(source_checkpoint),sha256=digest(source_checkpoint),indices=positive,role='Posthoc fixed training transitions, never additional training or heldout evaluation')
                break
        for run in runs:
            for path in [run/'initial.policy.pt',*sorted(run.glob('step_*.resume.pt'))]:
                agent,step=load_model(path);sources['checkpoints'][str(path)]=digest(path)
                measure(agent,step,probe['arrays'],probe['metadata']['transition_ids'],'A_prior_fixed128',run.name,0)
                if successful_probe is not None:
                    measure(agent,step,successful_probe[0],successful_probe[1],'E_first_successes_'+alg,run.name,0)
                with torch.no_grad():q=agent.online(held_batch['observations']).gather(1,held_batch['actions'][:,None]).squeeze(1)
                for i,index in enumerate(held):heldout.append(dict(model=run.name,source='D_prior_terminal_holdout',env_steps=step,transition_id=transition_id(old,index),q=float(q[i]),reward=float(held_batch['rewards'][i]),terminal_error_q_minus_r=float(q[i]-held_batch['rewards'][i])))
        for source_run in runs:
            checkpoint=source_run/'last.resume.pt';arrays=read_checkpoint(checkpoint,'resume')['agent']['replay']['arrays'];rng=np.random.default_rng(20261106)
            indices=[rng.choice(len(arrays['rewards']),64,replace=False).tolist() for _ in range(128)]
            source='C_new_'+source_run.name;sources['replays'][source]=dict(checkpoint_sha256=digest(checkpoint),minibatch_indices=indices)
            for run in runs:
                agent,step=load_model(run/'last.resume.pt')
                for k,index in enumerate(indices):measure(agent,step,subset(arrays,index),[transition_id(arrays,i) for i in index],source,run.name,k)
        print(alg,'Q/TD and gradient diagnostics complete',flush=True)
    df=pd.DataFrame(samples);df.to_csv(out/'samples.csv',index=False);pd.DataFrame(gradients).to_csv(out/'gradients.csv',index=False);pd.DataFrame(batches).to_csv(out/'batches.csv',index=False);pd.DataFrame(heldout).to_csv(out/'fixed_terminal_holdout.csv',index=False)
    summary=[]
    for key,g in df.groupby(['model','source','env_steps','group']):
        for field in ['q','reward','target','td','loss','bootstrap','terminal_error_q_minus_r']:
            values=g[field].dropna()
            if len(values):summary.append(dict(zip(['model','source','env_steps','group'],key))|dict(quantity=field,n=len(values),mean=values.mean(),p05=values.quantile(.05),p50=values.quantile(.5),p95=values.quantile(.95),mean_abs=values.abs().mean()))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False);(out/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
if __name__=='__main__':main()
