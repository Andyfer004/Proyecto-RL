"""Generate actual metrics/curves without environment rollouts or training."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from thruster_landing.checkpoints import load_inference,read_checkpoint
from thruster_landing.envs import EnvConfig
from thruster_landing.envs.physics import State
from thruster_landing.envs.rewards import components

ROOT=Path('artifacts/curriculum_beta10_lr1e4_seed42_v1')

def main():
 out=ROOT/'report';out.mkdir(exist_ok=True)
 cfg=EnvConfig();algorithms=['dqn','double_dqn'];allval=[];alltrain=[];allupdates=[];contact_rows=[];outcome_rows=[];timings=[];gaps=[];sources=[]
 fig,axes=plt.subplots(2,4,figsize=(16,7),sharex=True)
 trajectories,tx=plt.subplots(2,4,figsize=(16,8))
 for ai,alg in enumerate(algorithms):
  first=ROOT/f'{alg}_level1';end=ROOT/f'{alg}_complete';end=end if end.exists() else first
  val=pd.read_csv(end/'validation.csv');train=pd.read_csv(end/'episodes.csv');updates=pd.read_csv(end/'updates.csv');meta=json.loads((end/'metadata.json').read_text());step=meta['env_steps']
  val['algorithm']=alg;train['algorithm']=alg;updates['algorithm']=alg
  train['calendar_phase_at_end']=np.select([train.env_steps<=50000,train.env_steps<=150000],[1,2],default=3)
  allval.append(val);alltrain.append(train);allupdates.append(updates)
  for segment in [first]+([end] if end!=first else []):
   m=json.loads((segment/'metadata.json').read_text());timings.append(dict(algorithm=alg,segment=segment.name,start_steps=m['start_steps'],env_steps=m['env_steps'],wall_seconds=m['wall_seconds'],interactions_per_second=m['segment_steps_per_second'],validation_seconds=m['validation_seconds'],checkpoint_seconds=m['checkpoint_seconds']))
  for level in [1,2,3,0]:
   group=val[val.source_level==level];curve=group.groupby('env_steps').success.mean()
   axes[ai,0].plot(curve.index,curve*100,label=f'L{level}' if level else 'Original')
  axes[ai,0].set_ylabel(alg+' — éxito (%)');axes[ai,0].legend()
  for origin,g in train.groupby('source_level'):
   axes[ai,1].plot(g.env_steps,g.success.astype(float).rolling(100,min_periods=1).mean()*100,label=f'Origen L{origin}')
  axes[ai,0].set_ylim(0,100);axes[ai,1].set_ylim(0,100)
  for axis in axes[ai]:axis.set_xlabel('Pasos de entorno')
  axes[ai,1].legend();axes[ai,2].plot(updates.env_steps,updates.mean_abs_td_error,label='|TD|');axes[ai,2].plot(updates.env_steps,updates.mean_abs_q,label='|Q|');axes[ai,2].legend()
  axes[ai,3].plot(updates.env_steps,updates.gradient_norm_before_clip,label='Antes');axes[ai,3].plot(updates.env_steps,updates.gradient_norm_after_clip,label='Después');axes[ai,3].legend()
  # Use the fixed first validation ID per level, at initial and final. Include all outcomes in metrics.
  initial=pd.read_csv(first/'initial_validation.traces.csv');final=pd.read_csv(end/'final_validation.traces.csv')
  for col,level in enumerate([1,2,3,0]):
   condition=f'level{level}' if level else 'original';ids=sorted(final.loc[final.condition==condition,'scenario_id'].unique());identifier=ids[0]
   for phase,frame in [('Inicial',initial),('Final',final)]:
    t=frame[frame.scenario_id==identifier];tx[ai,col].plot(t.x,t.y,label=phase)
   tx[ai,col].plot([-2.5,2.5],[0,0],'k-',lw=3);tx[ai,col].set_title(f'{alg} {condition}\n{identifier}');tx[ai,col].legend();tx[ai,col].set_xlabel('x (m)');tx[ai,col].set_ylabel('y (m)')
  # Also export all final trajectories and states, so selection cannot conceal failures.
  final.to_csv(out/f'{alg}_all_final_trajectories.csv',index=False)
  for phase,frame in [('initial',initial),('final',final)]:
   model,_=load_inference(first/'initial.policy.pt' if phase=='initial' else end/'last.policy.pt')
   actual=val[val.env_steps==(0 if phase=='initial' else step)].set_index('scenario_id')
   for identifier,t in frame.groupby('scenario_id',sort=False):
    records=t.to_dict('records');raw=0.;disc=0.
    for k,(a,b) in enumerate(zip(records,records[1:])):
     sa=State(**{key:a[key] for key in ['x','y','vx','vy','theta','omega','gas']});sb=State(**{key:b[key] for key in ['x','y','vx','vy','theta','omega','gas']})
     r=sum(components(sa,sb,b['time']-a['time'],b['outcome'],cfg).values());raw+=r;disc+=model.config.gamma**k*r
    assert abs(raw-actual.loc[identifier,'return'])<1e-6
    a=records[0];obs=np.array([a['x']/12,(a['y']-1)/25,a['vx']/10,a['vy']/10,np.sin(a['theta']),np.cos(a['theta']),a['omega']/3,a['gas']/18000,0.],np.float32)
    with torch.inference_mode():q=float(model.online(torch.from_numpy(obs)).max())
    gaps.append(dict(algorithm=alg,phase=phase,env_steps=0 if phase=='initial' else step,source_level=int(actual.loc[identifier,'source_level']),scenario_id=identifier,outcome=actual.loc[identifier,'outcome'],q_initial_greedy=q,discounted_greedy_return=disc,undiscounted_return=raw,greedy_policy_return_gap=q-disc))
  # Reconstruct origin of every transition from successive ring snapshots, without resampling.
  previous=0
  for path in sorted([*first.glob('step_*.resume.pt'),*end.glob('step_*.resume.pt')] if end!=first else first.glob('step_*.resume.pt'),key=lambda p:int(p.stem.split('.')[0].split('_')[1])):
   payload=read_checkpoint(path,'resume');n=payload['agent']['env_steps'];replay=payload['agent']['replay'];positions=(np.arange(previous,n)%replay['capacity']).astype(int);levels=replay['arrays']['source_levels'][positions].numpy()
   for l in [1,2,3]:sources.append(dict(algorithm=alg,from_step=previous+1,to_step=n,source_level=l,transitions=int((levels==l).sum())))
   previous=n
  for split,data in [('train',train),('validation_final',val[val.env_steps==step])]:
   for (level,outcome),g in data.groupby(['source_level','outcome']):
    row=dict(algorithm=alg,split=split,source_level=level,outcome=outcome,n=len(g),successes=int(g.success.sum()),duration_mean_s=g.duration_s.mean(),gas_mean_Ns=g.gas_consumed_Ns.mean(),return_mean=g['return'].mean(),contacts=int(g.contact_vy_m_s.notna().sum()))
    for field in [c for c in g if c.startswith('reward_')]+['discounted_return']:
     if field in g:row[field+'_mean']=g[field].mean()
    outcome_rows.append(row)
   for level,g in data.groupby('source_level'):
    c=g[g.contact_vy_m_s.notna()];row=dict(algorithm=alg,split=split,source_level=level,n=len(g),successes=int(g.success.sum()),contacts=len(c),near_platform_contacts=int((c.contact_error_x_m<=1.5).sum()),no_contact=len(g)-len(c))
    for field in ['contact_error_x_m','contact_vx_m_s','contact_vy_m_s','contact_theta_rad','contact_omega_rad_s']:
     row[field+'_mean']=c[field].mean();row[field+'_abs_mean']=c[field].abs().mean()
    for name,mask in dict(vx=c.contact_vx_m_s.abs()>.7,vy=(c.contact_vy_m_s<-1)|(c.contact_vy_m_s>0),theta=c.contact_theta_rad.abs()>.12,omega=c.contact_omega_rad_s.abs()>.25,x=c.contact_error_x_m>1.5).items():row['violations_'+name]=int(mask.sum())
    contact_rows.append(row)
 fig.suptitle('Currículo: validación, entrenamiento y estabilidad (una semilla)');fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=150);plt.close(fig)
 trajectories.tight_layout();trajectories.savefig(out/'fixed_trajectories.png',dpi=150);plt.close(trajectories)
 validation=pd.concat(allval);training=pd.concat(alltrain);updates=pd.concat(allupdates)
 validation.to_csv(out/'validation.csv',index=False);training.to_csv(out/'episodes.csv',index=False);updates.to_csv(out/'updates.csv',index=False)
 pd.DataFrame(contact_rows).to_csv(out/'contact_metrics.csv',index=False);pd.DataFrame(outcome_rows).to_csv(out/'by_outcome.csv',index=False);pd.DataFrame(timings).to_csv(out/'timings.csv',index=False);pd.DataFrame(gaps).to_csv(out/'greedy_return_gap.csv',index=False);pd.DataFrame(sources).to_csv(out/'transition_origins.csv',index=False)
 summaries=[]
 for (alg,step,level),g in validation.groupby(['algorithm','env_steps','source_level']):
  summaries.append(dict(algorithm=alg,env_steps=step,source_level=level,n=len(g),successes=int(g.success.sum()),contacts=int(g.contact_vy_m_s.notna().sum()),return_mean=g['return'].mean(),duration_s=g.duration_s.mean(),gas_Ns=g.gas_consumed_Ns.mean(),**{outcome:int((g.outcome==outcome).sum()) for outcome in ['success','out_of_bounds','body_contact','invalid_contact','missed_platform','time_limit']}))
 pd.DataFrame(summaries).to_csv(out/'validation_summary.csv',index=False)
 # Freeze result integrity; source archives explain authorized infrastructure modifications.
 reference=json.loads(Path('configs/reference_before_curriculum.json').read_text())['sha256'];changed=[]
 for path,expected in reference.items():
  if path in ['src/thruster_landing/train.py','src/thruster_landing/agents/replay.py']:continue
  assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
 execution=json.loads(Path('configs/curriculum_execution_manifest.json').read_text())['sha256']
 for path,expected in execution.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
 (out/'integrity.json').write_text(json.dumps(dict(preserved_reference_files=len(reference)-2,execution_files_verified=len(execution),physics_PID_reward_success_unchanged=True),indent=2)+'\n')
 print(pd.DataFrame(summaries).groupby('algorithm').tail(4).to_string(index=False));print('report:',out,flush=True)
if __name__=='__main__':main()
