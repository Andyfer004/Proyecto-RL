"""Comparación LR: registros preservados y sonda fija; sin entrenar ni simular."""
import argparse
import json
import os
from pathlib import Path
import tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'thruster-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from thruster_landing.envs import EnvConfig
from thruster_landing.envs.physics import State,geometry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-analysis',type=Path,default=Path('artifacts/lr1e4_analysis'))
    parser.add_argument('--old-analysis',type=Path,default=Path('artifacts/pilot_100k_analysis'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/lr_ablation_comparison'))
    parser.add_argument('--old-runs',type=Path,nargs=2,default=[Path('artifacts/dqn_pilot_100k'),Path('artifacts/double_dqn_pilot_100k')],metavar=('DQN','DOUBLE_DQN'))
    parser.add_argument('--new-runs',type=Path,nargs=2,default=[Path('artifacts/dqn_lr1e4_100k'),Path('artifacts/double_dqn_lr1e4_100k')],metavar=('DQN','DOUBLE_DQN'))
    args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    groups=[]; summary=[]; thresholds=[]; approaches=[];gradients=[];gap_frames=[]
    config=EnvConfig()
    runs=[('pilot_100k_seed42_v1',alg,run,args.old_analysis) for alg,run in zip(('dqn','double_dqn'),args.old_runs)]
    runs += [('lr_0p0001_seed42_100k_v1',alg,run,args.new_analysis) for alg,run in zip(('dqn','double_dqn'),args.new_runs)]
    fig,axes=plt.subplots(2,2,figsize=(12,10))
    examples,exaxes=plt.subplots(2,2,figsize=(12,10))
    for experiment,algorithm,run,analysis in runs:
        m=json.loads((run/'metadata.json').read_text())
        tag=f'{algorithm} LR={m["config"]["learning_rate"]}'
        validation=pd.read_csv(run/'validation.csv')
        episodes=pd.read_csv(run/'episodes.csv')
        update=pd.read_csv(run/'updates.csv')
        progress=pd.read_csv(analysis/'validation_progress.csv')
        progress=progress[progress.algorithm==algorithm].copy()
        progress['experiment']=experiment
        groups.append(progress)
        calibration=pd.read_csv(analysis/'q_calibration.csv')
        final_calibration=calibration[(calibration.algorithm==algorithm)&(calibration.phase=='final')]
        calibration=calibration[calibration.algorithm==algorithm].copy()
        calibration['experiment']=experiment
        calibration=calibration.rename(columns={'prediction_minus_observed':'greedy_policy_return_gap'})
        calibration['run']=run.name
        calibration['env_steps']=calibration.phase.map({'initial':0,'best':m['best_validation']['env_steps'],'final':100000})
        gap_frames.append(calibration)
        calibration.to_csv(out/f'{run.name}_greedy_policy_gap.csv',index=False)
        final=progress[progress.env_steps==100000].iloc[0].to_dict()
        summary.append({'experiment':experiment,'algorithm':algorithm,'learning_rate':m['config']['learning_rate'],
            'training_episodes':len(episodes),'training_successes':int(episodes.success.sum()),
            'validation_successes_final':int(final['successes']),'contacts_final':int(final['n_contact']),
            'return_undiscounted_final':final['mean_return_all'],
            'return_discounted_final':final_calibration.observed_discounted_return.mean(),
            'greedy_q_initial_states_final':final_calibration.predicted_greedy_q.mean(),
            'greedy_policy_return_gap':final_calibration.prediction_minus_observed.mean(),
            'gas_all_final':final['mean_gas_consumed_Ns_all'],'duration_all_final':final['mean_duration_s_all'],
            'out_of_bounds_final':final['n_out_of_bounds'],'body_contact_final':final['n_body_contact'],
            'invalid_contact_final':final['n_invalid_contact'],'missed_platform_final':final['n_missed_platform'],
            'time_limit_final':final['n_time_limit'],'best_step':m['best_validation']['env_steps'],
            'gradient_before_actual_last_interval':update.gradient_norm_before_clip.iloc[-1],
            'gradient_after_actual_last_interval':update.gradient_norm_after_clip.iloc[-1] if 'gradient_norm_after_clip' in update else None,
            'wall_seconds':m['wall_seconds'],'steps_per_second':m['segment_steps_per_second']})
        for (step,group) in validation.groupby('env_steps'):
            for row in group.itertuples():
                contact=not pd.isna(row.contact_vy_m_s)
                state=State(**json.loads(row.final_state));points=geometry(state,config)
                criteria={'body_above':points[:4,1].min()>1e-6,
                    'feet_inside_platform':bool((np.abs(points[4:,0])<=config.platform_half_width).all()),
                    'x':abs(state.x)<=config.success_x,'vx':abs(state.vx)<=config.success_vx,
                    'vy':-config.success_vy<=state.vy<=0,'theta':abs(state.theta)<=config.success_angle,
                    'omega':abs(state.omega)<=config.success_omega,
                    'support_height':points[4:,1].max()<=config.support_height_tolerance}
                thresholds.append({'experiment':experiment,'algorithm':algorithm,'env_steps':step,'scenario_id':row.scenario_id,
                    'outcome':row.outcome,'has_contact':contact,
                    'failed_contact_criteria':','.join(name for name,ok in criteria.items() if not ok) if contact else None,
                    **{name:bool(ok) if contact else None for name,ok in criteria.items()}})
        trace=pd.read_csv(run/'final_validation.traces.csv')
        result=pd.read_csv(run/'final_validation.csv').set_index('scenario_id')
        ax=axes[0 if algorithm=='dqn' else 1,0 if m['config']['learning_rate']==.0003 else 1]
        for identifier,trajectory in trace.groupby('scenario_id',sort=False):
            color='tab:green' if result.loc[identifier,'success'] else ('tab:red' if not pd.isna(result.loc[identifier,'contact_vy_m_s']) else 'tab:gray')
            ax.plot(trajectory.x,trajectory.y,color=color,alpha=.4,lw=.8)
            positions=np.hypot(trajectory.x,trajectory.y-config.foot_drop)
            entered=False; clearances=[]
            for item in trajectory.itertuples():
                state=State(**{key:getattr(item,key) for key in ('x','y','vx','vy','theta','omega','gas')})
                points=geometry(state,config)
                if points[:4,1].min()>0 and abs(state.x)<=config.success_x and (np.abs(points[4:,0])<=config.platform_half_width).all():
                    clearance=max(0.,float(points[4:,1].min()))
                    clearances.append(clearance)
                    entered |= clearance<=1.0
            approaches.append({'experiment':experiment,'algorithm':algorithm,'scenario_id':identifier,
                'outcome':result.loc[identifier,'outcome'],'min_center_distance_to_landing_pose_m':positions.min(),
                'min_foot_clearance_over_platform_m':min(clearances) if clearances else None,
                'entered_last_meter_over_platform':bool(entered)})
        ax.plot([-config.platform_half_width,config.platform_half_width],[0,0],color='black',lw=3)
        ax.set(title=tag+' — todos los 32 casos finales',xlabel='x centro (m)',ylabel='y centro (m)',
               xlim=(-config.x_limit,config.x_limit),ylim=(-.2,config.ceiling+1))
        ax.grid(alpha=.2)
        for column,identifier in enumerate(('rl-validation-000','rl-validation-016')):
            selected=trace[trace.scenario_id==identifier]
            panel=exaxes[0 if algorithm=='dqn' else 1,column]
            panel.plot(selected.x,selected.y,label=f'LR={m["config"]["learning_rate"]}: {result.loc[identifier,"outcome"]}')
            panel.set(title=f'{algorithm}: {identifier}',xlabel='x centro (m)',ylabel='y centro (m)')
            panel.grid(alpha=.2);panel.legend(fontsize=8)
    pd.DataFrame(summary).to_csv(out/'comparison.csv',index=False)
    pd.concat(groups).to_csv(out/'validation_progress.csv',index=False)
    pd.DataFrame(thresholds).to_csv(out/'contact_criteria.csv',index=False)
    pd.DataFrame(approaches).to_csv(out/'approach.csv',index=False)
    gaps=pd.concat(gap_frames)
    gaps.to_csv(out/'greedy_policy_return_gap.csv',index=False)
    gap_fig,gap_axes=plt.subplots(1,3,figsize=(16,4))
    for run,values in gaps.groupby('run',sort=False):
        grouped=values.groupby('env_steps').mean(numeric_only=True)
        for ax,column,title in zip(gap_axes,('predicted_greedy_q','observed_discounted_return','greedy_policy_return_gap'),
                    ('Q greedy firmado','Retorno descontado de la política greedy','Brecha frente al retorno\nde la política greedy')):
            ax.plot(grouped.index,grouped[column],marker='o',label=run)
            ax.set(title=title,xlabel='Pasos')
            ax.grid(alpha=.2);ax.legend(fontsize=6)
    gap_fig.suptitle('Q − retorno greedy observado: no es error exacto respecto a Q óptimo')
    gap_fig.tight_layout();gap_fig.savefig(out/'greedy_policy_return_gap.png',dpi=150);plt.close(gap_fig)
    fig.suptitle('Resultados finales: verde éxito, rojo contacto fallido, gris sin contacto; sin selección de episodios')
    fig.tight_layout();fig.savefig(out/'all_final_trajectories.png',dpi=150);plt.close(fig)
    examples.suptitle('Casos fijos: primer vertical y primer desplazado; ambas tasas LR')
    examples.tight_layout();examples.savefig(out/'fixed_examples.png',dpi=150);plt.close(examples)
    data=pd.read_csv(out/'fixed_transition_summary.csv')
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    for row,group in enumerate(('terminal','nonterminal')):
        for col,quantity in enumerate(('q','td_target_minus_q','terminal_error_q_minus_reward' if group=='terminal' else 'bootstrap')):
            subset=data[(data.group==group)&(data.quantity==quantity)]
            for run,values in subset.groupby('run',sort=False):
                ax=axes[row,col]
                ax.plot(values.env_steps,values['mean'],label=run)
                ax.fill_between(values.env_steps,values.p05,values.p95,alpha=.1)
                short={'q':'Q(s,a)','td_target_minus_q':'TD = target − Q',
                       'terminal_error_q_minus_reward':'Error terminal = Q − r','bootstrap':'Bootstrap γ Q_obj'}[quantity]
                ax.set(title=f'{group}: {short}',xlabel='Pasos',ylabel='Valor con signo')
                ax.grid(alpha=.2);ax.legend(fontsize=6)
    fig.suptitle('Mismas 64 terminales y 64 no terminales — líneas: media; bandas: percentiles 05–95')
    fig.tight_layout();fig.savefig(out/'fixed_td_signed.png',dpi=150);plt.close(fig)
    gradients=pd.read_csv(out/'fixed_batch_gradients.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for run,values in gradients.groupby('run',sort=False):
        for ax,column in zip(axes,('gradient_norm_before_clip','gradient_norm_after_clip')):
            ax.plot(values.env_steps,values[column],label=run)
            ax.set(title=column+' — sonda fija balanceada',xlabel='Pasos',ylabel='Norma')
            ax.grid(alpha=.2);ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(out/'fixed_gradients.png',dpi=150);plt.close(fig)
    print(out)


if __name__=='__main__':
    main()
