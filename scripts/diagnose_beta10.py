"""Fuentes A/B/C/D explícitas: sondas fijas previas y replays nuevos por separado."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from thruster_landing.beta_diagnostics import measure_groups
from thruster_landing.q_diagnostics import load_model,transition_id
from thruster_landing.offline_diagnostics import subset
from thruster_landing.checkpoints import read_checkpoint
from thruster_landing.evaluation import write_csv
from thruster_landing.experiments import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('artifacts/beta10_diagnostics'))
    parser.add_argument('--new-runs',type=Path,nargs=2,default=[Path('artifacts/dqn_beta10_100k'),Path('artifacts/double_dqn_beta10_100k')])
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Usar un directorio de diagnóstico nuevo')
    args.output.mkdir(parents=True)
    probe_path=Path('artifacts/lr_audit_reference/fixed_probe.pt')
    probe=torch.load(probe_path,weights_only=True)
    old_metadata=json.load(open('artifacts/offline_interference_v1/metadata.json'))
    sample_rows,group_rows,batch_rows,holdout_rows=[],[],[],[]
    manifests={'A_probe_sha256':digest(probe_path),'B_and_D_metadata_sha256':digest('artifacts/offline_interference_v1/metadata.json'),
               'definitions':{'A_fixed128':'same prior 64 terminal + 64 nonterminal transitions, all checkpoints',
                   'B_fixed_baseline':'same previous 128 uniform B=64 minibatches on baseline replay, final checkpoints',
                   'C_new_replay':'128 uniform B=64 batches on new replay, final checkpoints; measured by both models',
                   'D_fixed_holdout':'same prior 256 terminal holdout transitions per baseline, all checkpoints'},'runs':{}}

    def measure(agent,batch,source,source_run,model_run,step,batch_id,identifiers):
        details={}
        for beta in (1.,10.):
            rows,full,detail=measure_groups(agent,batch,beta)
            common={'source':source,'source_run':source_run,'model_run':model_run,'model_beta':agent.config.smooth_l1_beta,
                    'env_steps':step,'batch':batch_id,'evaluated_beta':beta}
            group_rows.extend({**common,**row} for row in rows)
            batch_rows.append({**common,**full});details[beta]=detail
        for index,identifier in enumerate(identifiers):
            first,second=details[1.],details[10.]
            assert torch.equal(first['q'],second['q']) and torch.equal(first['target'],second['target'])
            terminal=bool(batch['terminated'][index])
            sample_rows.append({'source':source,'source_run':source_run,'model_run':model_run,'model_beta':agent.config.smooth_l1_beta,
                'env_steps':step,'batch':batch_id,'transition_id':identifier,'group':'terminal' if terminal else 'nonterminal',
                'q':float(first['q'][index]),'reward':float(first['reward'][index]),'target':float(first['target'][index]),
                'td_signed':float(first['td'][index]),'loss_beta1':float(first['loss'][index]),'loss_beta10':float(second['loss'][index]),
                'derivative_beta1':float(first['derivative'][index]),'derivative_beta10':float(second['derivative'][index])})

    for algorithm_index,algorithm in enumerate(('dqn','double_dqn')):
        baseline_name=algorithm+'_lr1e4_100k';baseline=Path('artifacts')/baseline_name
        new=args.new_runs[algorithm_index]
        baseline_payload=read_checkpoint(baseline/'last.resume.pt','resume')
        baseline_arrays=baseline_payload['agent']['replay']['arrays']
        new_payload=read_checkpoint(new/'last.resume.pt','resume');new_arrays=new_payload['agent']['replay']['arrays']
        old_manifest=old_metadata['manifests'][baseline_name]
        baseline_indices=old_manifest['uniform_minibatch_indices']
        held=old_manifest['supervised_split']['holdout_indices']
        rng=np.random.default_rng(20261104+algorithm_index)
        new_indices=[rng.choice(len(new_arrays['rewards']),64,replace=False).tolist() for _ in range(128)]
        manifests['runs'][algorithm]={'baseline_replay_sha256':digest(baseline/'last.resume.pt'),'new_replay_sha256':digest(new/'last.resume.pt'),
            'baseline_run':str(baseline),'new_run':str(new),'new_replay_indices':new_indices,
            'fixed_baseline_indices':baseline_indices,'fixed_holdout_indices':held}
        for root in (baseline,new):
            checkpoints=[root/'initial.policy.pt',*sorted(root.glob('step_*.resume.pt'))]
            for checkpoint in checkpoints:
                agent,step=load_model(checkpoint)
                assert agent.config.algorithm==algorithm
                measure(agent,probe['arrays'],'A_fixed128','prior_common_probe',root.name,step,0,probe['metadata']['transition_ids'])
                data=subset(baseline_arrays,held)
                with torch.no_grad():
                    q=agent.online(data['observations']).gather(1,data['actions'][:,None]).squeeze(1)
                for local,index in enumerate(held):
                    holdout_rows.append({'source':'D_fixed_holdout','source_run':baseline_name,'model_run':root.name,
                        'model_beta':agent.config.smooth_l1_beta,'env_steps':step,'transition_id':transition_id(baseline_arrays,index),
                        'q':float(q[local]),'reward':float(data['rewards'][local]),'terminal_error_q_minus_r':float(q[local]-data['rewards'][local])})
            agent,step=load_model(root/'last.resume.pt')
            for source,source_run,arrays,indices_list in (('B_fixed_baseline',baseline_name,baseline_arrays,baseline_indices),
                                                          ('C_new_replay',new.name,new_arrays,new_indices)):
                for number,indices in enumerate(indices_list):
                    measure(agent,subset(arrays,indices),source,source_run,root.name,step,number,
                            [transition_id(arrays,index) for index in indices])
            print(root.name,'fixed/new replay diagnostics complete',flush=True)
    write_csv(args.output/'samples.csv',sample_rows);write_csv(args.output/'gradients.csv',group_rows)
    write_csv(args.output/'batches.csv',batch_rows);write_csv(args.output/'fixed_holdout.csv',holdout_rows)
    sample=pd.DataFrame(sample_rows);summaries=[]
    keys=['source','source_run','model_run','model_beta','env_steps','group']
    for key,group in sample.groupby(keys,sort=False):
        for field in ('q','target','td_signed','loss_beta1','loss_beta10','derivative_beta1','derivative_beta10'):
            values=group[field]
            summaries.append({**dict(zip(keys,key)),'quantity':field,'n':len(values),'mean':values.mean(),
                'p05':values.quantile(.05),'p50':values.quantile(.5),'p95':values.quantile(.95)})
    write_csv(args.output/'sample_summary.csv',summaries)
    (args.output/'sources.json').write_text(json.dumps(manifests,indent=2)+'\n')
    print(args.output,flush=True)


if __name__=='__main__':main()
