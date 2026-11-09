"""Replay retention, exact exposure, and paired checkpoint diagnostics; no training."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from thruster_landing.checkpoints import read_checkpoint
from thruster_landing.replay_exposure import SAMPLE_DTYPE
from thruster_landing.q_diagnostics import load_model
from thruster_landing.beta_diagnostics import measure_groups
from thruster_landing.offline_diagnostics import subset

BASELINE=Path('artifacts/exploration_comparison_level1_seed42_100k_v1')


def summarize_exposure(root, algorithm, capacity):
    if capacity==50000:
        run=BASELINE/f'{algorithm}_weighted_variant';segments=[run];history=run
        samples=np.load(root/'baseline_sampling'/f'{algorithm}.npz');ids=samples['transition_steps'].ravel();when=np.repeat(samples['env_steps'],64);counts=samples['sample_counts'];q=target=None
    else:
        segments=[root/f'{algorithm}_prefix50k',root/f'{algorithm}_continued100k'];history=segments[-1]
        samples=np.concatenate([np.fromfile(p/'learning_samples.bin',dtype=SAMPLE_DTYPE) for p in segments]);ids=samples['transition_step'];when=samples['env_steps'];q=samples['q'];target=samples['target'];counts=np.load(history/'exposure_counts.npz')['sample_counts']
    assert len(ids)==24751*64
    np.testing.assert_array_equal(np.bincount(ids,minlength=100001),counts)
    ep=pd.read_csv(history/'episodes.csv');val=pd.read_csv(history/'validation.csv');up=pd.read_csv(history/'updates.csv')
    positive=ep[ep.success];label=np.zeros(100001,dtype=bool);terminal=np.zeros(100001,dtype=bool);epid=np.zeros(100001,dtype=np.int32);end=np.zeros(100001,dtype=np.int32)
    for row in positive.itertuples():
        first=int(row.env_steps-row.steps+1);last=int(row.env_steps);label[first:last+1]=True;epid[first:last+1]=row.episode;end[first:last+1]=last;terminal[last]=True
    hit=label[ids];hit_ids=ids[hit];hit_when=when[hit];hit_terminal=terminal[hit_ids];known=hit_when>=end[hit_ids]
    sampled=pd.DataFrame(dict(algorithm=algorithm,capacity=capacity,transition_step=hit_ids,update_env_steps=hit_when,episode=epid[hit_ids],terminal_success=hit_terminal,episode_outcome_known=known,q=q[hit] if q is not None else np.nan,target=target[hit] if target is not None else np.nan))
    sampled['td']=sampled.target-sampled.q
    per_transition=[]
    for tid in np.where(label)[0]:
        h=hit_ids==tid;per_transition.append(dict(algorithm=algorithm,capacity=capacity,transition_step=tid,episode=int(epid[tid]),terminal_success=bool(terminal[tid]),episode_end=int(end[tid]),retained_at100k=tid>100000-capacity,total_draws=int(h.sum()),draws_before_outcome=int((h&~known).sum()),draws_after_outcome=int((h&known).sum())))
    progress=[]
    for t in range(0,100001,10000):
        discovered=label&(end<=t);active=discovered&(np.arange(100001)>max(0,t-capacity))&(np.arange(100001)<=t)
        selected=(hit_when<=t)&(end[hit_ids]<=t)
        progress.append(dict(algorithm=algorithm,capacity=capacity,env_steps=t,training_successes=int((positive.env_steps<=t).sum()),retained_success_terminal=int((active&terminal).sum()),retained_success_episode_transitions=int(active.sum()),cumulative_success_terminal_draws=int((selected&hit_terminal).sum()),cumulative_success_episode_draws=int(selected.sum()),updates_with_success_terminal=int(np.unique(hit_when[selected&hit_terminal]).size),updates_with_success_episode=int(np.unique(hit_when[selected]).size)))
    replay=read_checkpoint(history/'last.resume.pt','resume')['agent']['replay'];actual_retained=int((replay['arrays']['terminated']&(replay['arrays']['rewards']>0)).sum());assert actual_retained==progress[-1]['retained_success_terminal']
    result=dict(algorithm=algorithm,capacity=capacity,training_episodes=len(ep),training_successes=len(positive),first_success_step=int(positive.env_steps.min()) if len(positive) else None,retained_success_terminal=progress[-1]['retained_success_terminal'],retained_success_episode_transitions=progress[-1]['retained_success_episode_transitions'],terminal_success_draws=int(hit_terminal.sum()),success_episode_draws=len(hit_ids),terminal_updates=int(np.unique(hit_when[hit_terminal]).size),success_episode_updates=int(np.unique(hit_when).size),draws_before_outcome=int((~known).sum()),draws_after_outcome=int(known.sum()))
    for level,name in [(1,'level1'),(0,'original'),(2,'level2'),(3,'level3')]:
        g=val[(val.env_steps==100000)&(val.source_level==level)];result[name+'_successes']=int(g.success.sum());result[name+'_n']=len(g)
    return result,progress,sampled,per_transition,ep,val,up,segments


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path('artifacts/replay150k_weighted_seed42_100k_v1'));args=parser.parse_args();out=args.root/'analysis';out.mkdir(exist_ok=False)
    results=[];progress=[];sampled=[];transitions=[];episodes=[];validations=[];updates=[];outcomes=[];timings=[]
    fixed=torch.load('artifacts/replay_capacity_reference/early_success_probe.pt',weights_only=True)
    prior=torch.load('artifacts/lr_audit_reference/fixed_probe.pt',weights_only=True)
    probes=[];groups=[];batches=[]
    for alg in ['dqn','double_dqn']:
        for capacity in [50000,150000]:
            result,p,s,t,ep,val,up,segments=summarize_exposure(args.root,alg,capacity);results.append(result);progress+=p;sampled.append(s);transitions+=t
            for df in [ep,val,up]:df['algorithm']=alg;df['capacity']=capacity
            episodes.append(ep);validations.append(val);updates.append(up)
            for scope,data in [('training',ep),('greedy_final',val[val.env_steps==100000])]:
                for (level,cause),g in data.groupby(['source_level','termination_cause']):
                    c=g[g.contact];outcomes.append(dict(algorithm=alg,capacity=capacity,scope=scope,source_level=level,cause=cause,n=len(g),contacts=len(c),gas_exhausted=int(g.gas_exhausted.sum()),duration_mean_s=g.duration_s.mean(),gas_mean_Ns=g.gas_consumed_Ns.mean(),return_mean=g['return'].mean(),discounted_return_mean=g.discounted_return.mean(),**{field+'_mean':c[field].mean() for field in ['contact_error_x_m','contact_vx_m_s','contact_vy_m_s','contact_theta_rad','contact_omega_rad_s']},contact_theta_abs_mean=c.contact_theta_rad.abs().mean(),contact_omega_abs_mean=c.contact_omega_rad_s.abs().mean()))
            for segment in segments:
                meta=json.loads((segment/'metadata.json').read_text());timings.append(dict(algorithm=alg,capacity=capacity,segment=segment.name,start_steps=meta['start_steps'],end_steps=meta['env_steps'],wall_seconds=meta['wall_seconds'],validation_seconds=meta['validation_seconds'],checkpoint_seconds=meta['checkpoint_seconds']))
            checkpoints=[segments[0]/'initial.policy.pt']
            for segment in segments:checkpoints+=sorted(segment.glob('step_*.resume.pt'))
            for path in checkpoints:
                agent,step=load_model(path)
                for source,probe in [('early_success_episodes',fixed),('prior_fixed128',prior)]:
                    batch=probe['arrays'];grow,full,detail=measure_groups(agent,batch,10.)
                    common=dict(algorithm=alg,capacity=capacity,env_steps=step,source=source)
                    groups += [{**common,**row} for row in grow];batches.append({**common,**full})
                    for i in range(len(batch['rewards'])):
                        row={**common,'transition_step':fixed['transition_steps'][i] if source=='early_success_episodes' else None,'probe_index':i,'terminal':bool(batch['terminated'][i]),'action':int(batch['actions'][i])};row.update({key:float(v[i]) for key,v in detail.items()});row['bootstrap']=0. if row['terminal'] else row['target']-row['reward'];probes.append(row)
            print(alg,capacity,'retention/exposure/fixed-Q complete',flush=True)
    for name,data in [('results',results),('progress',progress),('successful_transition_exposure',transitions),('by_outcome',outcomes),('timings',timings),('fixed_probe_values',probes),('fixed_probe_gradients',groups),('fixed_probe_batches',batches)]:pd.DataFrame(data).to_csv(out/(name+'.csv'),index=False)
    pd.concat(sampled).to_csv(out/'successful_learning_samples.csv',index=False);pd.concat(episodes).to_csv(out/'episodes.csv',index=False);pd.concat(validations).to_csv(out/'validation.csv',index=False);pd.concat(updates).to_csv(out/'updates.csv',index=False)
    pd.DataFrame(results).to_csv(out/'results.csv',index=False)
    print(pd.DataFrame(results).to_string(index=False),flush=True)
if __name__=='__main__':main()
