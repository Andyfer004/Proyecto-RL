"""Paired returns and step-level interpretation of frozen v1 behaviour rollouts."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TERMS=['terminal','progress','speed','orientation','gas','time']
BEHAVIOURS=['pid','off','up','dqn','double_dqn']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('artifacts/reward_audit_level1_v1'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/reward_audit_level1_v1/report'))
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    episodes=pd.read_csv(args.input/'episodes.csv');steps=pd.read_csv(args.input/'steps.csv');pairs=pd.read_csv(args.input/'paired_differences.csv')
    assert len(episodes)==800 and len(pairs)==640
    assert episodes.groupby(['split','scenario_id']).behaviour.nunique().eq(5).all()
    assert episodes.groupby(['split','scenario_id']).initial_state.nunique().eq(1).all()
    episodes['contact']=episodes.contact_vy_m_s.notna()
    summaries=[];by_outcome=[]
    for (split,behaviour),g in episodes.groupby(['split','behaviour'],sort=False):
        row=dict(split=split,behaviour=behaviour,n=len(g),successes=int(g.success.sum()),contacts=int(g.contact.sum()),discounted_return_mean=g.discounted_return.mean(),discounted_return_min=g.discounted_return.min(),discounted_return_max=g.discounted_return.max(),return_mean=g['return'].mean(),duration_mean_s=g.duration_s.mean(),max_height_mean_m=g.max_center_height_m.mean(),gas_mean_Ns=g.gas_consumed_Ns.mean())
        row.update({prefix+term+'_mean':g[prefix+term].mean() for prefix in ['reward_','discounted_'] for term in TERMS})
        summaries.append(row)
        for cause,rows in g.groupby('termination_cause'):
            by_outcome.append(dict(split=split,behaviour=behaviour,termination_cause=cause,n=len(rows),contact_n=int(rows.contact.sum()),contact_vy_mean=rows.contact_vy_m_s.mean(),contact_theta_abs_mean=rows.contact_theta_rad.abs().mean(),contact_error_x_mean=rows.contact_error_x_m.mean()))
    pd.DataFrame(summaries).to_csv(out/'behaviour_summary.csv',index=False);pd.DataFrame(by_outcome).to_csv(out/'termination_summary.csv',index=False)
    paired=[]
    for (split,other),g in pairs.groupby(['split','other'],sort=False):
        row=dict(split=split,other=other,n=len(g),pid_successes=int(g.pid_success.sum()),wins_discounted=int((g.delta_discounted_return>0).sum()),wins_undiscounted=int((g.delta_return>0).sum()))
        for field in ['delta_discounted_return','delta_return']:
            row.update({field+'_min':g[field].min(),field+'_p05':g[field].quantile(.05),field+'_median':g[field].median(),field+'_mean':g[field].mean(),field+'_max':g[field].max()})
        paired.append(row)
    pd.DataFrame(paired).to_csv(out/'paired_summary.csv',index=False)
    # This CSV keeps each matched state and all five behaviours side by side, not just aggregate means.
    wide=episodes.pivot(index=['split','scenario_id','seed'],columns='behaviour',values=['discounted_return','return','outcome','duration_s','gas_consumed_Ns'])
    wide.columns=[f'{metric}__{behaviour}' for metric,behaviour in wide.columns];wide.reset_index().to_csv(out/'all_states_comparison.csv',index=False)
    signal=[];scales=[]
    pid_success=set(episodes.loc[(episodes.behaviour=='pid')&episodes.success,'scenario_id'])
    pid=steps[(steps.behaviour=='pid')&steps.scenario_id.isin(pid_success)].copy()
    pid['progress_x']=2*(pid.before_x.abs()-pid.after_x.abs())/12
    pid['progress_height']=2*.3*((pid.before_y-1).clip(lower=0)-(pid.after_y-1).clip(lower=0))/25
    np.testing.assert_allclose(pid.progress_x+pid.progress_height,pid.reward_progress,atol=1e-12)
    pid['speed_penalty_change_vs_before']=-.08*pid.elapsed_s*(np.hypot(pid.after_vx,pid.after_vy)-np.hypot(pid.before_vx,pid.before_vy))/10
    pid['orientation_penalty_change_vs_before']=-pid.elapsed_s*(.3*(pid.after_theta.abs()-pid.before_theta.abs())/np.pi+.05*(pid.after_omega.abs()-pid.before_omega.abs())/3)
    for tag in ['tilt_correction','precontact_braking','horizontal_approach','descent','ascent','landing']:
        g=pid[pid[tag]]
        row=dict(signal=tag,n_steps=len(g),n_episodes=g.scenario_id.nunique(),reward_mean=g.reward.mean(),reward_sum=g.reward.sum(),reward_min=g.reward.min(),reward_max=g.reward.max(),positive_reward_steps=int((g.reward>0).sum()))
        for field in ['reward_'+term for term in TERMS]+['progress_x','progress_height','speed_penalty_change_vs_before','orientation_penalty_change_vs_before']:
            row[field+'_mean']=g[field].mean();row[field+'_sum']=g[field].sum()
        signal.append(row)
    pd.DataFrame(signal).to_csv(out/'pid_signal.csv',index=False)
    for (behaviour,terminal),g in steps.groupby(['behaviour','terminated']):
        for term in TERMS:
            v=g['reward_'+term];scales.append(dict(behaviour=behaviour,terminated=terminal,term=term,n_steps=len(g),mean_per_step=v.mean(),min_per_step=v.min(),p05=v.quantile(.05),p50=v.quantile(.5),p95=v.quantile(.95),max_per_step=v.max(),sum_all_episodes=v.sum(),mean_per_second=(v/g.elapsed_s).mean()))
    pd.DataFrame(scales).to_csv(out/'per_step_scales.csv',index=False)
    # Signed full-episode accumulation: each column/term is independent of group overlap above.
    components=episodes.groupby(['split','behaviour'])[['reward_'+t for t in TERMS]+['discounted_'+t for t in TERMS]].mean()
    components.to_csv(out/'episode_components.csv')
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for col,other in enumerate(['off','up','dqn','double_dqn']):
        for split,label in [('train','Entrenamiento'),('validation','Validación')]:
            g=pairs[(pairs.other==other)&(pairs.split==split)]
            for row,prefix in [(0,'discounted_'),(1,'')]:
                axes[row,col].scatter(g['other_'+prefix+'return'],g['pid_'+prefix+'return'],s=12,alpha=.6,label=label)
        for row in range(2):
            axes[row,col].plot([-110,110],[-110,110],'k--',lw=.7);axes[row,col].set_xlabel(other+' — retorno');axes[row,col].set_ylabel('PID — retorno');axes[row,col].set_title('Descontado' if row==0 else 'Sin descuento');axes[row,col].legend()
    fig.suptitle('Mismos estados de nivel 1: cada punto es un caso emparejado');fig.tight_layout();fig.savefig(out/'paired_returns.png',dpi=150);plt.close(fig)
    identifier=sorted(episodes.loc[episodes.split=='validation','scenario_id'].unique())[0]
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    for behaviour in BEHAVIOURS:
        g=steps[(steps.scenario_id==identifier)&(steps.behaviour==behaviour)].sort_values('step')
        for ax,field,label in zip(axes.flat,['after_y','after_vy','reward','discounted_reward','after_gas','after_theta'],['Altura centro (m)','vy (m/s)','Recompensa por paso','Recompensa descontada por paso','Gas restante (N s)','θ (rad)']):
            ax.plot(g.time_after_s,g[field],label=behaviour);ax.set_ylabel(label);ax.set_xlabel('Tiempo (s)')
    axes[0,0].legend();fig.suptitle('Primer ID de validación: '+identifier);fig.tight_layout();fig.savefig(out/'fixed_case_trajectories.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,7))
    for ax,behaviour in zip(axes,['pid','double_dqn']):
        g=steps[(steps.scenario_id==identifier)&(steps.behaviour==behaviour)].sort_values('step')
        for term in TERMS:ax.plot(g.time_after_s,g['discounted_'+term].cumsum(),label=term)
        ax.set_title(behaviour+' — componentes descontados acumulados');ax.set_xlabel('Tiempo (s)');ax.legend(ncol=3)
    fig.tight_layout();fig.savefig(out/'discounted_components.png',dpi=150);plt.close(fig)
    print(pd.DataFrame(summaries)[['split','behaviour','n','successes','discounted_return_mean','return_mean','duration_mean_s','max_height_mean_m','gas_mean_Ns']].to_string(index=False),flush=True)
    print(pd.DataFrame(paired).to_string(index=False),flush=True)
    print(pd.DataFrame(signal)[['signal','n_steps','reward_mean','positive_reward_steps','reward_progress_mean','reward_gas_mean','reward_time_mean']].to_string(index=False),flush=True)
    (out/'scope.json').write_text(json.dumps(dict(matched_states=160,train_states=128,validation_states=32,behaviours=BEHAVIOURS,episodes=800,pairwise_comparisons=640,signal_masks_overlap=True,representative_case_rule='First validation scenario ID; fixed, not selected by outcome'),indent=2)+'\n')
if __name__=='__main__':main()
