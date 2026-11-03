"""Diagnóstico reproducible desde registros; no entrena ni ejecuta prueba final."""
import argparse
import json
import os
from pathlib import Path
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'thruster-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from thruster_landing.checkpoints import load_inference, read_checkpoint
from thruster_landing.agents.dqn import compute_targets
from thruster_landing.envs import EnvConfig
from thruster_landing.envs.physics import State, geometry
from thruster_landing.envs.rewards import components


def aggregate(group, config):
    contact = group[group.contact_vy_m_s.notna()]
    success = group[group.success]
    result = {'n': len(group), 'successes': len(success), 'n_contact': len(contact),
              'success_rate': len(success)/len(group)}
    for outcome in ('success','invalid_contact','body_contact','missed_platform','out_of_bounds','time_limit'):
        result['n_' + outcome] = int((group.outcome == outcome).sum())
    for field in ('return','duration_s','gas_consumed_Ns'):
        result['mean_' + field + '_all'] = group[field].mean()
    for field in ('contact_error_x_m','contact_vx_m_s','contact_vy_m_s','contact_theta_rad','contact_omega_rad_s'):
        result['mean_abs_' + field] = contact[field].abs().mean()
        result['mean_abs_' + field + '_success'] = success[field].abs().mean()
    for field in (column for column in group.columns if column.startswith('reward_')):
        result['mean_' + field] = group[field].mean()
    for label, field, bound in (('x','contact_error_x_m',config.success_x),
                                ('vx','contact_vx_m_s',config.success_vx),
                                ('theta','contact_theta_rad',config.success_angle),
                                ('omega','contact_omega_rad_s',config.success_omega)):
        result['fraction_contacts_within_' + label] = (contact[field].abs() <= bound).mean() if len(contact) else np.nan
    result['fraction_contacts_within_vy'] = contact.contact_vy_m_s.between(-config.success_vy,0).mean() if len(contact) else np.nan
    states = [json.loads(value) for value in group.final_state]
    result['mean_abs_x_final_all'] = np.mean([abs(state['x']) for state in states])
    result['mean_abs_theta_final_all'] = np.mean([abs(state['theta']) for state in states])
    result['mean_abs_vy_final_all'] = np.mean([abs(state['vy']) for state in states])
    result['mean_vy_signed_final_all'] = np.mean([state['vy'] for state in states])
    result['mean_center_height_final_all'] = np.mean([state['y'] for state in states])
    result['mean_abs_vx_final_all'] = np.mean([abs(state['vx']) for state in states])
    result['mean_abs_omega_final_all'] = np.mean([abs(state['omega']) for state in states])
    exits = [geometry(State(**json.loads(row.final_state)),config)
             for row in group[group.outcome=='out_of_bounds'].itertuples()]
    result['n_exits_ceiling'] = sum(points[:,1].max() >= config.ceiling-1e-6 for points in exits)
    result['n_exits_horizontal'] = sum(np.abs(points[:,0]).max() >= config.x_limit-1e-6 for points in exits)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/pilot_100k_analysis'))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    config = EnvConfig()
    summaries, timings, examples, diagnostics, q_progress, calibration = [], [], [], {}, [], []
    fig, axes = plt.subplots(len(args.runs),2,figsize=(12,5*len(args.runs)),squeeze=False)
    example_fig, example_axes = plt.subplots(len(args.runs),3,figsize=(16,5*len(args.runs)),squeeze=False)
    for index, run in enumerate(args.runs):
        meta = json.loads((run/'metadata.json').read_text())
        algorithm = meta['config']['algorithm']
        validation = pd.read_csv(run/'validation.csv')
        episodes = pd.read_csv(run/'episodes.csv')
        updates = pd.read_csv(run/'updates.csv')
        for step, group in validation.groupby('env_steps'):
            summaries.append({'algorithm':algorithm,'env_steps':step,**aggregate(group,config)})
        timings.append({'algorithm':algorithm,**{key:meta[key] for key in
                       ('wall_seconds','segment_steps_per_second','environment_seconds','learning_seconds',
                        'validation_seconds','checkpoint_seconds','env_steps','updates','completed_training_episodes')},
                        'training_successes':int(episodes.success.sum()),
                        'best_step':meta['best_validation']['env_steps'],
                        'best_success_rate':meta['best_validation']['success_rate'],
                        'best_landing_error_m':meta['best_validation']['landing_error_m']})
        phase_rows, phase_traces = {}, {}
        for phase in ('initial','best','final'):
            phase_rows[phase] = pd.read_csv(run/f'{phase}_validation.csv')
            phase_traces[phase] = pd.read_csv(run/f'{phase}_validation.traces.csv')
        initial_states = phase_traces['initial'].groupby('scenario_id',sort=False).head(1)
        def normalize(trace):
            return np.column_stack([trace.x/config.position_scale,(trace.y-config.foot_drop)/config.height_scale,
                trace.vx/config.velocity_scale,trace.vy/config.velocity_scale,np.sin(trace.theta),np.cos(trace.theta),
                trace.omega/config.omega_scale,trace.gas/config.gas_capacity,np.zeros(len(trace))]).astype(np.float32)
        fixed = torch.from_numpy(normalize(initial_states))
        for step in sorted(validation.env_steps.unique()):
            checkpoint = run/'initial.policy.pt' if step == 0 else run/f'step_{int(step):08d}.policy.pt'
            model,_ = load_inference(checkpoint)
            with torch.inference_mode():
                q = model.online(fixed)
            q_progress.append({'algorithm':algorithm,'env_steps':step,'min_q':float(q.min()),
                              'max_q':float(q.max()),'mean_q':float(q.mean()),
                              'mean_greedy_q':float(q.max(dim=1).values.mean())})
        for phase in ('initial','best','final'):
            checkpoint=run/('initial.policy.pt' if phase=='initial' else ('best.policy.pt' if phase=='best' else 'last.policy.pt'))
            model,_=load_inference(checkpoint)
            for identifier,trajectory in phase_traces[phase].groupby('scenario_id',sort=False):
                raw_return=discounted_return=0.0
                records=trajectory.to_dict('records')
                for transition,(before,after) in enumerate(zip(records,records[1:])):
                    a=State(**{key:before[key] for key in ('x','y','vx','vy','theta','omega','gas')})
                    b=State(**{key:after[key] for key in ('x','y','vx','vy','theta','omega','gas')})
                    reward=sum(components(a,b,after['time']-before['time'],after['outcome'],config).values())
                    raw_return+=reward
                    discounted_return+=model.config.gamma**transition*reward
                actual=phase_rows[phase].set_index('scenario_id').loc[identifier]
                assert abs(raw_return-actual['return'])<1e-6, (phase,identifier,raw_return,actual['return'])
                with torch.inference_mode():
                    predicted=float(model.online(torch.from_numpy(normalize(trajectory.head(1)))).max())
                calibration.append({'algorithm':algorithm,'phase':phase,'scenario_id':identifier,
                    'outcome':actual.outcome,'predicted_greedy_q':predicted,'observed_discounted_return':discounted_return,
                    'prediction_minus_observed':predicted-discounted_return})
        for column, phase in enumerate(('initial','final')):
            ax = axes[index,column]
            results = phase_rows[phase].set_index('scenario_id')
            for identifier, trajectory in phase_traces[phase].groupby('scenario_id',sort=False):
                status = results.loc[identifier,'outcome']
                color = 'tab:green' if status=='success' else ('tab:gray' if status in ('time_limit','out_of_bounds') else 'tab:red')
                ax.plot(trajectory.x,trajectory.y,color=color,alpha=.4,lw=.8)
                ax.scatter(trajectory.x.iloc[-1],trajectory.y.iloc[-1],color=color,s=10)
            ax.plot([-config.platform_half_width,config.platform_half_width],[0,0],color='black',lw=3)
            ax.axhline(config.foot_drop,color='gray',ls=':',lw=.8)
            ax.set(title=f'{algorithm} {phase}: {int(phase_rows[phase].success.sum())}/32 éxitos',
                   xlabel='x del centro (m)',ylabel='y del centro (m)')
            ax.set_xlim(-config.x_limit,config.x_limit)
            ax.set_ylim(-.2,config.ceiling+1)
            ax.grid(alpha=.2)
        # Ejemplos declarados: primero de cada condición, primer éxito/fallo.
        pool = pd.concat(phase_rows.values(),ignore_index=True)
        chosen = phase_rows['final'].groupby('condition',sort=False).head(1).scenario_id.tolist()
        for status in (True,False):
            available = pool[pool.success == status].sort_values('scenario_id')
            if len(available) and available.scenario_id.iloc[0] not in chosen:
                chosen.append(available.scenario_id.iloc[0])
        for column, identifier in enumerate(chosen):
            ax=example_axes[index,column]
            for phase in ('initial','best','final'):
                row=phase_rows[phase][phase_rows[phase].scenario_id==identifier].iloc[0]
                trajectory=phase_traces[phase][phase_traces[phase].scenario_id==identifier]
                examples.append({'algorithm':algorithm,'phase':phase,'selection':'first per condition / first success / first failure',**row.to_dict()})
                ax.plot(trajectory.x,trajectory.y,label=f'{phase}: {row.outcome}',lw=1)
            ax.plot([-config.platform_half_width,config.platform_half_width],[0,0],color='black',lw=3)
            ax.set(title=f'{algorithm}: {identifier}',xlabel='x centro (m)',ylabel='y centro (m)')
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        for ax in example_axes[index,len(chosen):]:
            ax.set_visible(False)
        saved=read_checkpoint(run/'last.resume.pt','resume')
        replay=saved['agent']['replay']['arrays']
        observations=replay['observations']
        all_finite=all(bool(torch.isfinite(value).all()) for value in replay.values())
        checkpoint_diagnostics={'replay_size':saved['agent']['replay']['size'],'finite_replay':all_finite,
            'observation_min':observations.min(dim=0).values.tolist(),
            'observation_max':observations.max(dim=0).values.tolist(),
            'max_sincos_norm_error':float((observations[:,4]**2+observations[:,5]**2-1).abs().max()),
            'gas_fraction_in_range':bool(((observations[:,7]>=0)&(observations[:,7]<=1)).all()),
            'terminal_transitions_in_replay':int(replay['terminated'].sum()),
            'truncated_transitions_in_replay':int(replay['truncated'].sum()),
            'positive_terminal_transitions_in_replay':int(((replay['rewards']>0)&replay['terminated']).sum()),
            'finite_update_metrics':bool(np.isfinite(updates[['loss','mean_abs_td_error','mean_abs_q','gradient_norm_before_clip']].dropna()).all().all()),
            'max_interval_mean_abs_q':float(updates.mean_abs_q.max()),
            'last_interval_mean_abs_q':float(updates.mean_abs_q.iloc[-1]),
            'max_interval_loss':float(updates.loss.max()),'last_interval_loss':float(updates.loss.iloc[-1]),
            'last_interval_td_error':float(updates.mean_abs_td_error.iloc[-1]),
            'training_return_duration_correlation':float(episodes['return'].corr(episodes.duration_s)),
            'training_outcomes':{str(k):int(v) for k,v in episodes.outcome.value_counts().items()}}
        agent,_=load_inference(run/'last.policy.pt')
        agent.target.load_state_dict(saved['agent']['target'])
        batch={key:value[:512] for key,value in replay.items()}
        with torch.no_grad():
            next_online=agent.online(batch['next_observations'])
            next_target=agent.target(batch['next_observations'])
            targets=compute_targets(batch['rewards'],batch['terminated'],next_online,next_target,
                                    agent.config.gamma,algorithm=='double_dqn')
        mask=batch['terminated']
        checkpoint_diagnostics['terminal_target_error_max']=float((targets[mask]-batch['rewards'][mask]).abs().max()) if mask.any() else None
        checkpoint_diagnostics['sampled_targets_finite']=bool(torch.isfinite(targets).all())
        checkpoint_diagnostics['replay_reward_range']=[float(replay['rewards'].min()),float(replay['rewards'].max())]
        trace=phase_traces['final']
        normalized=normalize(trace)
        with torch.inference_mode():
            values=agent.online(torch.from_numpy(normalized))
        checkpoint_diagnostics['max_abs_q_on_final_validation_states']=float(values.abs().max())
        diagnostics[algorithm]=checkpoint_diagnostics
    summary=pd.DataFrame(summaries)
    summary.to_csv(out/'validation_progress.csv',index=False)
    pd.DataFrame(timings).to_csv(out/'timing.csv',index=False)
    pd.DataFrame(examples).to_csv(out/'examples.csv',index=False)
    (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    pd.DataFrame(q_progress).to_csv(out/'q_progress.csv',index=False)
    pd.DataFrame(calibration).to_csv(out/'q_calibration.csv',index=False)
    fig.suptitle('Todas las trayectorias de validación — verde: éxito; rojo: contacto fallido; gris: sin contacto')
    fig.tight_layout()
    fig.savefig(out/'initial_final_trajectories.png',dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for algorithm,group in pd.DataFrame(q_progress).groupby('algorithm'):
        axes[0].plot(group.env_steps,group.mean_greedy_q,label=algorithm)
    for algorithm,group in pd.DataFrame(calibration).groupby('algorithm'):
        for label,field in (('Q predicho','predicted_greedy_q'),('Retorno descontado observado','observed_discounted_return')):
            values=group.groupby('phase',sort=False)[field].mean()
            axes[1].plot(values.index,values.values,marker='o',label=algorithm+' '+label)
    axes[0].set(title='Q greedy en los mismos 32 estados iniciales',xlabel='Pasos',ylabel='Q medio')
    axes[1].set(title='Calibración respecto a la trayectoria greedy',ylabel='Valor / retorno descontado')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out/'q_calibration.png',dpi=150)
    plt.close(fig)
    example_fig.suptitle('Ejemplos: primero de cada condición y primer éxito/fallo disponible; sin elegir por retorno')
    example_fig.tight_layout()
    example_fig.savefig(out/'examples.png',dpi=150)
    plt.close(example_fig)
    fig,axes=plt.subplots(3,2,figsize=(12,10))
    fields=[('mean_abs_contact_error_x_m','Error horizontal al contacto (m)'),
            ('mean_abs_contact_vy_m_s','|vy| al contacto (m/s)'),
            ('mean_abs_contact_vx_m_s','|vx| al contacto (m/s)'),
            ('mean_abs_contact_theta_rad','|θ| al contacto (rad)'),
            ('mean_abs_contact_omega_rad_s','|ω| al contacto (rad/s)'),
            ('mean_gas_consumed_Ns_all','Impulso consumido, todos (N s)')]
    for algorithm,group in summary.groupby('algorithm'):
        for ax,(field,title) in zip(axes.flat,fields):
            ax.plot(group.env_steps,group[field],marker='o',label=algorithm)
            ax.set(title=title,xlabel='Pasos de entrenamiento')
            ax.grid(alpha=.2)
            ax.legend()
    fig.suptitle('Diagnóstico: medias condicionadas a contacto; denominadores n_contact en CSV')
    fig.tight_layout()
    fig.savefig(out/'contact_progress.png',dpi=150)
    plt.close(fig)
    print(out)


if __name__=='__main__':
    main()
