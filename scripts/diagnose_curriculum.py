"""Read-only fixed-probe and uniform replay diagnostics for the curriculum pilot."""
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

ROOT=Path('artifacts/curriculum_beta10_lr1e4_seed42_v1')

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'diagnostics');args=parser.parse_args()
 out=args.output;out.mkdir(parents=True,exist_ok=False)
 probe_path=Path('artifacts/lr_audit_reference/fixed_probe.pt');probe=torch.load(probe_path,weights_only=True)
 samples=[];gradients=[];batches=[];sources={'fixed_probe_sha256':digest(probe_path),'new_uniform_replay_indices':{},'checkpoints':{}}
 def measure(agent,step,batch,ids,source,model,batch_id):
  groups,full,details=measure_groups(agent,batch,10.)
  common=dict(model=model,env_steps=step,source=source,batch=batch_id)
  gradients.extend({**common,**r} for r in groups);batches.append({**common,**full})
  for i,identifier in enumerate(ids):
   terminal=bool(batch['terminated'][i]);row={**common,'transition_id':identifier,'group':'terminal' if terminal else 'nonterminal','source_level':int(batch['source_levels'][i]) if 'source_levels' in batch else 0,'action':int(batch['actions'][i]),'truncated':bool(batch['truncated'][i])}
   row.update({k:float(v[i]) for k,v in details.items()})
   row.update(bootstrap=0. if terminal else row['target']-row['reward'],terminal_error_q_minus_r=row['q']-row['reward'] if terminal else None)
   assert not terminal or row['target']==row['reward']
   samples.append(row)
 for alg in ['dqn','double_dqn']:
  first=ROOT/f'{alg}_level1';end=ROOT/f'{alg}_complete';end=end if end.exists() else first
  baseline=Path(f'artifacts/{alg}_beta10_100k')
  checkpoints=[baseline/'initial.policy.pt',*sorted(baseline.glob('step_*.resume.pt')),first/'initial.policy.pt',*sorted(first.glob('step_*.resume.pt'))]
  if end!=first:checkpoints+=sorted(end.glob('step_*.resume.pt'))
  for path in checkpoints:
   agent,step=load_model(path);model='baseline_'+alg if path.parent==baseline else 'curriculum_'+alg
   sources['checkpoints'][str(path)]=digest(path)
   measure(agent,step,probe['arrays'],probe['metadata']['transition_ids'],'A_prior_fixed128',model,0)
  # Each replay source is shared between baseline and new models; never conflate sources.
  for source_run in [baseline,end]:
   arrays=read_checkpoint(source_run/'last.resume.pt','resume')['agent']['replay']['arrays'];rng=np.random.default_rng(20261105)
   indices=[rng.choice(len(arrays['rewards']),64,replace=False).tolist() for _ in range(128)]
   source='B_beta10_baseline_replay' if source_run==baseline else 'C_curriculum_replay'
   sources['new_uniform_replay_indices'][str(source_run)]={'indices':indices,'checkpoint_sha256':digest(source_run/'last.resume.pt')}
   for model_run in [baseline,end]:
    agent,step=load_model(model_run/'last.resume.pt');model=('baseline_' if model_run==baseline else 'curriculum_')+alg
    for k,index in enumerate(indices):measure(agent,step,subset(arrays,index),[transition_id(arrays,i) for i in index],source,model,k)
  print(alg,'diagnostics complete',flush=True)
 df=pd.DataFrame(samples);df.to_csv(out/'samples.csv',index=False)
 pd.DataFrame(gradients).to_csv(out/'gradients.csv',index=False);pd.DataFrame(batches).to_csv(out/'batches.csv',index=False)
 summary=[]
 for key,group in df.groupby(['model','source','env_steps','group']):
  for field in ['q','reward','target','td','loss','bootstrap','terminal_error_q_minus_r']:
   values=group[field].dropna()
   if len(values):summary.append(dict(zip(['model','source','env_steps','group'],key))|dict(quantity=field,n=len(values),mean=values.mean(),p05=values.quantile(.05),p50=values.quantile(.5),p95=values.quantile(.95)))
 pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
 (out/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
if __name__=='__main__':main()
