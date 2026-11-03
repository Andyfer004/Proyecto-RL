"""Diagnóstico acotado en copias: no llama al simulador ni al entrenador RL."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from thruster_landing.agents.dqn import DQNAgent,DQNConfig
from thruster_landing.checkpoints import read_checkpoint,atomic_save
from thruster_landing.experiments import digest,verify_reference
from thruster_landing.envs import EnvConfig
from thruster_landing.envs.physics import State,geometry
from thruster_landing.evaluation import write_csv
from thruster_landing.offline_diagnostics import (decompose,subset,split_terminal_groups,
    state_action_key,terminal_errors,targets_for,selected_values,one_update_copy)


def decode(observation,c):
    return State(x=float(observation[0])*c.position_scale,y=float(observation[1])*c.height_scale+c.foot_drop,
        vx=float(observation[2])*c.velocity_scale,vy=float(observation[3])*c.velocity_scale,
        theta=float(torch.atan2(observation[4],observation[5])),omega=float(observation[6])*c.omega_scale,
        gas=float(observation[7])*c.gas_capacity)


def terminal_kind(next_observation,c):
    points=geometry(decode(next_observation,c),c)
    tol=1e-4 # error de reconstrucción desde observaciones float32
    if points[:,1].min()<=tol:
        if points[:4,1].min()<=tol:return 'body_impact'
        if (np.abs(points[4:,0])>c.platform_half_width+tol).any():return 'missed_platform'
        return 'platform_contact'
    if points[:,1].max()>=c.ceiling-tol:return 'ceiling_exit'
    if np.abs(points[:,0]).max()>=c.x_limit-tol:return 'horizontal_exit'
    return 'unresolved'


def summarize_statistics(frame,keys,fields):
    rows=[]
    for key,group in frame.groupby(keys,dropna=False):
        key=key if isinstance(key,tuple) else (key,)
        for field in fields:
            values=group[field].dropna()
            if not len(values):continue
            rows.append({**dict(zip(keys,key)), 'quantity':field,'n':len(values),'mean':values.mean(),
                'std':values.std(ddof=1) if len(values)>1 else 0.,'p05':values.quantile(.05),
                'p50':values.quantile(.5),'p95':values.quantile(.95),'min':values.min(),'max':values.max()})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=Path('configs/offline_interference_protocol.json'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/offline_interference_v1'))
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Usar directorio nuevo; no sobrescribir diagnósticos')
    protocol=json.loads(args.protocol.read_text())
    if protocol['huber_beta']!=1.0:raise ValueError('Este diagnóstico solo usa beta=1')
    verify_reference()
    for reference_file in ('configs/reference_before_lr_ablation.json','configs/reference_lr1e4.json'):
        for path,expected in json.loads(Path(reference_file).read_text())['sha256'].items():
            assert digest(path)==expected,path
    args.output.mkdir(parents=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    groups,summaries,samples,supervised,interference,problems,manifests=[],[],[],[],[],[],{}
    checks={'weighted_gradient_identity':True,'target_terminal_is_reward':True,'no_source_mutation':True,'split_disjoint':True}
    checkpoint_hashes={}
    for run_number,run in enumerate(protocol['runs']):
        root=Path('artifacts')/run
        path=root/protocol['checkpoint'];checkpoint_hashes[str(path)]=digest(path)
        payload=read_checkpoint(path,'resume');saved=payload['agent']
        agent=DQNAgent(DQNConfig(**saved['config']))
        agent.online.load_state_dict(saved['online']);agent.target.load_state_dict(saved['target'])
        arrays=saved['replay']['arrays'];count=len(arrays['rewards'])
        source_weights={key:value.clone() for key,value in agent.online.state_dict().items()}
        rng=np.random.default_rng(protocol['seed']+run_number)
        batch_size=protocol['batch_size'];index_batches=[]
        for number in range(protocol['representative_minibatches']):
            indices=rng.choice(count,batch_size,replace=False).tolist();index_batches.append(indices)
            batch=subset(arrays,indices)
            group_rows,full,detail=decompose(agent,batch)
            summaries.append({'run':run,'batch':number,**full})
            for row in group_rows:groups.append({'run':run,'batch':number,**row})
            for i,index in enumerate(indices):
                samples.append({'run':run,'batch':number,'replay_index':index,
                    'group':'terminal' if bool(batch['terminated'][i]) else 'nonterminal',
                    'action':int(batch['actions'][i]),'reward':float(batch['rewards'][i]),'q':float(detail['q'][i]),'target':float(detail['target'][i]),
                    'td_signed':float(detail['td'][i]),'loss_per_sample':float(detail['loss'][i])})
        spec=protocol['supervised']
        training_indices,held_indices,excluded,split=split_terminal_groups(arrays,rng,spec['max_unique_state_action_groups'],spec['training_fraction'])
        training,held=subset(arrays,training_indices),subset(arrays,held_indices)
        held_keys=set(split['holdout_state_action_hashes'])
        # Excluir también cualquier no terminal idéntica en (s,a), si existiese.
        excluded={i for i in range(count) if state_action_key(arrays,i) in held_keys}
        split['excluded_replay_indices_for_interference']=sorted(excluded)
        model=copy.deepcopy(agent.online)
        optimizer=torch.optim.Adam(model.parameters(),lr=spec['learning_rate'])
        for step in range(spec['updates']+1):
            if step%spec['evaluation_frequency']==0 or step==spec['updates']:
                for label,data in (('train',training),('heldout',held)):
                    supervised.append({'run':run,'update':step,'split':label,'n':len(data['rewards']),**terminal_errors(model,data)})
            if step==spec['updates']:break
            indices=rng.choice(len(training_indices),spec['batch_size'],replace=False)
            batch=subset(training,indices)
            loss=F.smooth_l1_loss(selected_values(model,batch),batch['rewards'],beta=1.0)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),spec['gradient_norm_limit'],error_if_nonfinite=True)
            optimizer.step()
        atomic_save({'role':'supervised diagnostic only; not a landing policy','protocol':spec,
                     'run':run,'state_dict':model.state_dict()},args.output/f'{run}.supervised_diagnostic_only.pt')
        before=terminal_errors(agent.online,held)
        available=np.array([i for i in range(count) if i not in excluded],dtype=np.int64)
        paired_batches=[];rejected=0
        for number in range(protocol['interference']['minibatches']):
            while True:
                indices=rng.choice(available,batch_size,replace=False).tolist()
                mask=arrays['terminated'][indices]
                if mask.any() and (~mask).any():break
                rejected+=1
            paired_batches.append(indices)
            batch=subset(arrays,indices);target=targets_for(agent,batch)
            assert torch.equal(target[batch['terminated']],batch['rewards'][batch['terminated']])
            for mode in protocol['interference']['optimizer_modes']:
                for branch,mask in (('terminal_only',batch['terminated']),('nonterminal_only',~batch['terminated']),
                                     ('mixed',torch.ones(batch_size,dtype=torch.bool))):
                    clone,metrics=one_update_copy(agent.online,saved['optimizer'],agent.config.learning_rate,mode,
                                                  batch,target,mask,agent.config.gradient_norm_limit)
                    after=terminal_errors(clone,held)
                    interference.append({'run':run,'batch':number,'optimizer_mode':mode,'branch':branch,
                        'terminal_count':int(batch['terminated'].sum()),'active_count':int(mask.sum()),
                        'before_mae':before['mae'],'after_mae':after['mae'],'delta_mae':after['mae']-before['mae'],
                        'before_signed_error':before['mean_signed_error'],'after_signed_error':after['mean_signed_error'],
                        'delta_signed_error':after['mean_signed_error']-before['mean_signed_error'],**metrics})
        # Medir la región problemática sobre TODAS las terminales, en el s que recibe r.
        c=EnvConfig();terminal_indices=torch.where(arrays['terminated'])[0].tolist()
        terminal_batch=subset(arrays,terminal_indices)
        with torch.no_grad():
            q_values=selected_values(agent.online,terminal_batch)
        size=saved['replay']['size'];capacity=saved['replay']['capacity'];position=saved['replay']['position']
        first_index=position if size==capacity else 0
        for local,index in enumerate(terminal_indices):
            state=decode(arrays['observations'][index],c);next_state=decode(arrays['next_observations'][index],c)
            error=float(q_values[local]-arrays['rewards'][index])
            record={'run':run,'replay_index':index,'action':int(arrays['actions'][index]),
                'event_from_last_observation':terminal_kind(arrays['next_observations'][index],c),
                'q_state_action_before_event':float(q_values[local]),'terminal_reward':float(arrays['rewards'][index]),
                'terminal_error':error,'terminal_abs_error':abs(error),
                'norm_obs_s':float(arrays['observations'][index].norm()),
                'norm_last_obs':float(arrays['next_observations'][index].norm()),
                'speed_s':float(np.hypot(state.vx,state.vy)),'speed_last_obs':float(np.hypot(next_state.vx,next_state.vy)),
                'x_s':state.x,'y_s':state.y,'vy_s':state.vy,'omega_s':state.omega,
                'x_last_obs':next_state.x,'y_last_obs':next_state.y,'vy_last_obs':next_state.vy,'omega_last_obs':next_state.omega}
            previous=(index-1)%capacity
            if index!=first_index and not bool(arrays['terminated'][previous] or arrays['truncated'][previous]):
                assert torch.equal(arrays['next_observations'][previous],arrays['observations'][index])
                with torch.no_grad():previous_q=selected_values(agent.online,subset(arrays,[previous]))
                record.update(previous_replay_index=previous,previous_action=int(arrays['actions'][previous]),
                    previous_reward=float(arrays['rewards'][previous]),q_previous_state_action=float(previous_q[0]),
                    distance_previous_obs_to_s=float((arrays['observations'][index]-arrays['observations'][previous]).norm()),
                    same_action_as_previous=bool(arrays['actions'][index]==arrays['actions'][previous]))
            problems.append(record)
        assert all(torch.equal(value,agent.online.state_dict()[key]) for key,value in source_weights.items())
        assert digest(path)==checkpoint_hashes[str(path)]
        manifests[run]={'source_checkpoint':str(path),'source_sha256':checkpoint_hashes[str(path)],
            'replay_size':count,'terminal_replay_count':len(terminal_indices),'terminal_replay_fraction':len(terminal_indices)/count,
            'truncated_replay_count':int(arrays['truncated'].sum()),'uniform_minibatch_indices':index_batches,
            'supervised_split':split,'interference_minibatch_indices':paired_batches,'rejected_empty_group_interference_batches':rejected}
        print(run,'offline diagnostics complete',flush=True)
    write_csv(args.output/'group_gradients.csv',groups);write_csv(args.output/'minibatches.csv',summaries)
    write_csv(args.output/'sample_values.csv',samples);write_csv(args.output/'supervised_fit.csv',supervised)
    write_csv(args.output/'interference.csv',interference);write_csv(args.output/'problem_samples.csv',problems)
    group_frame=pd.DataFrame(groups);sample_frame=pd.DataFrame(samples)
    stats=summarize_statistics(group_frame,['run','group'],['count','proportion','shared_gradient_mean_norm',
        'shared_weighted_contribution_norm','projection_share_full_gradient','loss_per_sample_mean'])
    write_csv(args.output/'group_gradient_summary.csv',stats)
    write_csv(args.output/'sample_summary.csv',summarize_statistics(sample_frame,['run','group'],['q','target','td_signed','loss_per_sample']))
    write_csv(args.output/'interference_summary.csv',summarize_statistics(pd.DataFrame(interference),['run','optimizer_mode','branch'],['delta_mae','delta_signed_error']))
    problem_frame=pd.DataFrame(problems);problem_stats=[]
    for grouping in ('event_from_last_observation','action'):
        rows=summarize_statistics(problem_frame,['run',grouping],['terminal_error','terminal_abs_error','norm_obs_s','speed_s'])
        for row in rows:row['grouping']=grouping
        problem_stats.extend(rows)
    for run,values in problem_frame.groupby('run'):
        for field in ('norm_obs_s','speed_s'):
            threshold=values[field].quantile(.75)
            high=values[field]>=threshold
            for label,subset_frame in (('upper_quartile',values[high]),('lower_three_quartiles',values[~high])):
                for quantity in ('terminal_error','terminal_abs_error'):
                    data=subset_frame[quantity]
                    problem_stats.append({'run':run,'grouping':field,'group':label,'threshold':threshold,'quantity':quantity,
                        'n':len(data),'mean':data.mean(),'p50':data.median(),'p95':data.quantile(.95)})
    write_csv(args.output/'problem_summary.csv',problem_stats)
    for path,expected in checkpoint_hashes.items():assert digest(path)==expected
    for reference in ('configs/reference_before_lr_ablation.json','configs/reference_lr1e4.json'):
        for path,expected in json.loads(Path(reference).read_text())['sha256'].items():assert digest(path)==expected,path
    payload={'diagnostic_id':protocol['diagnostic_id'],'protocol':protocol,'manifests':manifests,'checks':checks,
             'wall_seconds':time.perf_counter()-start,'environment_interactions':0,
             'supervised_copy_updates':len(protocol['runs'])*protocol['supervised']['updates'],
             'independent_one_step_copy_updates':len(protocol['runs'])*protocol['interference']['minibatches']*3*2}
    (args.output/'metadata.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(args.output,flush=True)


if __name__=='__main__':main()
