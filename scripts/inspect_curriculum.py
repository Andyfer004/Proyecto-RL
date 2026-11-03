"""Supplementary read-only control diagnosis and matched-budget comparison."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from thruster_landing.checkpoints import read_checkpoint
from thruster_landing.q_diagnostics import load_model,transition_id
from thruster_landing.offline_diagnostics import subset
ROOT=Path('artifacts/curriculum_beta10_lr1e4_seed42_v1')

def main():
 out=ROOT/'report';control=[];comparison=[];fixed=[];holdout=[]
 metadata=json.loads(Path('artifacts/offline_interference_v1/metadata.json').read_text())
 fig,axes=plt.subplots(2,4,figsize=(16,8))
 for ai,alg in enumerate(['dqn','double_dqn']):
  first=ROOT/f'{alg}_level1';end=ROOT/f'{alg}_complete';end=end if end.exists() else first
  traces=pd.read_csv(end/'final_validation.traces.csv');initial=pd.read_csv(first/'initial_validation.traces.csv')
  results=pd.read_csv(end/'final_validation.csv').set_index('scenario_id')
  for level,condition in [(1,'level1'),(2,'level2'),(3,'level3'),(0,'original')]:
   for identifier,g in traces[traces.condition==condition].groupby('scenario_id'):
    state=g.iloc[-1];noninitial=g[g.action.notna()];gasempty=g[g.gas<1e-6];actions=noninitial.action.astype(int)
    control.append(dict(algorithm=alg,source_level=level,scenario_id=identifier,outcome=results.loc[identifier,'outcome'],max_height_m=g.y.max(),min_height_m=g.y.min(),empty_gas_time_s=gasempty.time.iloc[0] if len(gasempty) else None,empty_gas_height_m=gasempty.y.iloc[0] if len(gasempty) else None,ceiling_exit=results.loc[identifier,'outcome']=='out_of_bounds' and state.y>23.,**{f'action_{i}_fraction':(actions==i).mean() for i in range(8)}))
  identifier=sorted(traces.loc[traces.condition=='level1','scenario_id'].unique())[0]
  for label,frame in [('Inicial',initial),('Final',traces)]:
   g=frame[frame.scenario_id==identifier]
   for col,(field,unit) in enumerate([('y','y (m)'),('vy','vy (m/s)'),('theta','θ (rad)'),('gas','Impulso restante (N s)')]):
    axes[ai,col].plot(g.time,g[field],label=label);axes[ai,col].set_ylabel(f'{alg} {unit}');axes[ai,col].set_xlabel('Tiempo (s)');axes[ai,col].legend()
  baseline=Path(f'artifacts/{alg}_beta10_100k')
  for name,run,step in [('beta10_50k',baseline,50000),('beta10_100k',baseline,100000),('curriculum',end,json.loads((end/'metadata.json').read_text())['env_steps'])]:
   v=pd.read_csv(run/'validation.csv');v=v[v.env_steps==step]
   if name=='curriculum':v=v[v.source_level==0]
   train=pd.read_csv(run/'episodes.csv');train=train[train.env_steps<=step];c=v[v.contact_vy_m_s.notna()]
   comparison.append(dict(algorithm=alg,run=name,env_steps=step,training_episodes=len(train),training_successes=int(train.success.sum()),validation_successes=int(v.success.sum()),contacts=len(c),ceiling_exits=sum(r.outcome=='out_of_bounds' and json.loads(r.final_state)['y']>23 for r in v.itertuples()),return_mean=v['return'].mean(),duration_s=v.duration_s.mean(),gas_Ns=v.gas_consumed_Ns.mean(),contact_abs_vy=c.contact_vy_m_s.abs().mean(),contact_abs_theta=c.contact_theta_rad.abs().mean(),contact_error_x=c.contact_error_x_m.mean()))
  # Preserve the preceding terminal holdout exactly, including its baseline-specific indices.
  oldname=alg+'_lr1e4_100k';old=read_checkpoint(Path('artifacts')/oldname/'last.resume.pt','resume')['agent']['replay']['arrays'];indices=metadata['manifests'][oldname]['supervised_split']['holdout_indices'];batch=subset(old,indices)
  paths=[*baseline.glob('step_*.resume.pt'),*first.glob('step_*.resume.pt')]+([*end.glob('step_*.resume.pt')] if end!=first else [])
  for path in paths:
   model,step=load_model(path)
   with torch.no_grad():q=model.online(batch['observations']).gather(1,batch['actions'][:,None]).squeeze(1)
   for i,index in enumerate(indices):holdout.append(dict(model=('baseline_' if path.parent==baseline else 'curriculum_')+alg,env_steps=step,source='D_prior_terminal_holdout',transition_id=transition_id(old,index),q=float(q[i]),reward=float(batch['rewards'][i]),terminal_error_q_minus_r=float(q[i]-batch['rewards'][i])))
 fig.tight_layout();fig.savefig(out/'fixed_level1_states.png',dpi=150);plt.close(fig)
 pd.DataFrame(control).to_csv(out/'control_diagnosis.csv',index=False);pd.DataFrame(comparison).to_csv(out/'baseline_comparison.csv',index=False);pd.DataFrame(holdout).to_csv(ROOT/'diagnostics/fixed_terminal_holdout.csv',index=False)
 # Fixed-probe signed TD trajectories, excluding new-replay distribution changes.
 d=pd.read_csv(ROOT/'diagnostics/summary.csv');fig,axes=plt.subplots(2,2,figsize=(12,7))
 for ai,alg in enumerate(['dqn','double_dqn']):
  for model in ['baseline_'+alg,'curriculum_'+alg]:
   for gi,group in enumerate(['terminal','nonterminal']):
    g=d[(d.model==model)&(d.source=='A_prior_fixed128')&(d.group==group)&(d.quantity=='td')].sort_values('env_steps')
    axes[ai,gi].plot(g.env_steps,g['mean'],label=model);axes[ai,gi].fill_between(g.env_steps,g.p05,g.p95,alpha=.12);axes[ai,gi].axhline(0,color='k',lw=.5);axes[ai,gi].set_title(f'{alg} {group}: TD firmado, p05–p95');axes[ai,gi].legend()
 fig.tight_layout();fig.savefig(out/'fixed_td_comparison.png',dpi=150);plt.close(fig)
 print(pd.DataFrame(comparison).to_string(index=False))
if __name__=='__main__':main()
