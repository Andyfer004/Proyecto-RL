"""Comparación beta=1/10: control, valores y fuentes diagnósticas separadas."""
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
from analyze_pilot import aggregate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-runs',type=Path,nargs=2,default=[Path('artifacts/dqn_beta10_100k'),Path('artifacts/double_dqn_beta10_100k')])
    parser.add_argument('--new-analysis',type=Path,default=Path('artifacts/beta10_analysis'))
    parser.add_argument('--diagnostics',type=Path,default=Path('artifacts/beta10_diagnostics'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/beta10_comparison'))
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    runs=[]
    for algorithm,new in zip(('dqn','double_dqn'),args.new_runs):
        runs.extend([(algorithm,Path(f'artifacts/{algorithm}_lr1e4_100k'),Path('artifacts/lr1e4_analysis')),
                     (algorithm,new,args.new_analysis)])
    c=EnvConfig();summaries=[];progress=[];by_outcome=[];train_outcomes=[];gaps=[]
    fig,axes=plt.subplots(2,2,figsize=(12,10))
    curves,caxes=plt.subplots(2,3,figsize=(15,8))
    for algorithm,run,analysis in runs:
        meta=json.loads((run/'metadata.json').read_text());beta=meta['config'].get('smooth_l1_beta',1.)
        tag=f'{algorithm} beta={beta:g}'
        val=pd.read_csv(run/'validation.csv');episodes=pd.read_csv(run/'episodes.csv');updates=pd.read_csv(run/'updates.csv')
        calibration=pd.read_csv(analysis/'q_calibration.csv');calibration=calibration[calibration.algorithm==algorithm].copy()
        calibration['run']=run.name;calibration['beta']=beta
        calibration['greedy_policy_return_gap']=calibration.predicted_greedy_q-calibration.observed_discounted_return
        calibration['env_steps']=calibration.phase.map({'initial':0,'best':meta['best_validation']['env_steps'],'final':100000})
        gaps.append(calibration)
        final=val[val.env_steps==100000];final_stats=aggregate(final,c)
        final_gap=calibration[calibration.phase=='final']
        summaries.append({'run':run.name,'algorithm':algorithm,'beta':beta,'train_n':len(episodes),'train_successes':int(episodes.success.sum()),
            'validation_n':len(final),'validation_successes':int(final.success.sum()),'n_contact':final_stats['n_contact'],
            **{k:final_stats[k] for k in ('n_out_of_bounds','n_exits_ceiling','n_exits_horizontal','n_body_contact','n_invalid_contact','n_missed_platform','n_time_limit',
                'mean_return_all','mean_duration_s_all','mean_gas_consumed_Ns_all','mean_abs_contact_error_x_m','mean_abs_contact_vx_m_s','mean_abs_contact_vy_m_s','mean_abs_contact_theta_rad','mean_abs_contact_omega_rad_s')},
            'greedy_q_initial_mean':final_gap.predicted_greedy_q.mean(),'discounted_return_greedy_mean':final_gap.observed_discounted_return.mean(),
            'greedy_policy_return_gap':final_gap.greedy_policy_return_gap.mean(),'best_step':meta['best_validation']['env_steps'],
            'wall_seconds':meta['wall_seconds'],'steps_per_second':meta['segment_steps_per_second']})
        for step,group in val.groupby('env_steps'):
            progress.append({'run':run.name,'algorithm':algorithm,'beta':beta,'env_steps':step,**aggregate(group,c)})
            for outcome,data in group.groupby('outcome'):
                by_outcome.append({'run':run.name,'algorithm':algorithm,'beta':beta,'env_steps':step,'outcome':outcome,**aggregate(data,c)})
        for outcome,data in episodes.groupby('outcome'):
            train_outcomes.append({'run':run.name,'algorithm':algorithm,'beta':beta,'outcome':outcome,**aggregate(data,c)})
        aggregated=val.groupby('env_steps').agg(success_rate=('success','mean'),mean_return=('return','mean'),duration=('duration_s','mean'),gas=('gas_consumed_Ns','mean'))
        for ax,field in zip(caxes.flat[:4],('success_rate','mean_return','duration','gas')):
            ax.plot(aggregated.index,aggregated[field],marker='o',label=tag)
        caxes[1,1].plot(updates.env_steps,updates.mean_abs_td_error,label=tag)
        caxes[1,2].plot(updates.env_steps,updates.epsilon,label=tag)
        trace=pd.read_csv(run/'final_validation.traces.csv');results=final.set_index('scenario_id')
        ax=axes[0 if algorithm=='dqn' else 1,0 if beta==1 else 1]
        for identifier,trajectory in trace.groupby('scenario_id',sort=False):
            row=results.loc[identifier];color='tab:green' if row.success else ('tab:red' if not pd.isna(row.contact_vy_m_s) else 'tab:gray')
            ax.plot(trajectory.x,trajectory.y,color=color,alpha=.4,lw=.8)
        ax.plot([-c.platform_half_width,c.platform_half_width],[0,0],color='black',lw=3)
        ax.set(title=f'{tag}: {int(final.success.sum())}/32 éxitos finales',xlabel='x centro (m)',ylabel='y centro (m)',xlim=(-c.x_limit,c.x_limit),ylim=(-.2,c.ceiling+1))
        ax.grid(alpha=.2)
    pd.DataFrame(summaries).to_csv(out/'comparison.csv',index=False)
    pd.DataFrame(progress).to_csv(out/'validation_progress.csv',index=False)
    pd.DataFrame(by_outcome).to_csv(out/'validation_by_outcome.csv',index=False)
    pd.DataFrame(train_outcomes).to_csv(out/'training_by_outcome.csv',index=False)
    gap=pd.concat(gaps);gap.to_csv(out/'greedy_policy_return_gap.csv',index=False)
    fig.suptitle('Las 32 trayectorias finales: verde éxito; rojo contacto fallido; gris sin contacto')
    fig.tight_layout();fig.savefig(out/'all_final_trajectories.png',dpi=150);plt.close(fig)
    titles=['Éxitos en validación','Retorno SIN descuento','Duración, todos (s)','Impulso, todos (N s)','Error TD absoluto de minibatches','Epsilon']
    for ax,title in zip(caxes.flat,titles):
        ax.set(title=title,xlabel='Pasos de entrenamiento');ax.legend(fontsize=7);ax.grid(alpha=.2)
    caxes[0,0].set_ylim(-.05,1.05)
    curves.suptitle('Una semilla: mejora numérica y mejora de control son criterios distintos')
    curves.tight_layout();curves.savefig(out/'control_curves.png',dpi=150);plt.close(curves)
    gradients=pd.read_csv(args.diagnostics/'gradients.csv');batches=pd.read_csv(args.diagnostics/'batches.csv')
    compact=[]
    keys=['source','source_run','model_run','model_beta','evaluated_beta']
    natural=gradients[gradients.source.isin(['B_fixed_baseline','C_new_replay'])]
    for key,data in natural.groupby(keys,sort=False):
        q=dict(zip(keys,key));full=batches[(batches.source==q['source'])&(batches.source_run==q['source_run'])&(batches.model_run==q['model_run'])&(batches.evaluated_beta==q['evaluated_beta'])]
        q.update(cosine_mean=full.cosine_shared.mean(),cosine_std=full.cosine_shared.std(),cosine_defined=int(full.cosine_shared.count()))
        for group,subset in data.groupby('group'):
            count=subset.n.sum()
            q.update({group+'_n':int(count),group+'_fraction':count/(128*64),
                group+'_weighted_gradient_norm':subset.shared_weighted_gradient_norm.mean(),
                group+'_conditional_gradient_norm':subset[subset.n>0].shared_group_gradient_norm.mean(),
                group+'_mean_abs_derivative':np.average(subset[subset.n>0].mean_abs_loss_derivative,weights=subset[subset.n>0].n),
                group+'_linear_fraction':subset.linear_region_count.sum()/count,
                group+'_lt1_fraction':subset.abs_td_lt1_count.sum()/count,
                group+'_between1_and10_fraction':subset.abs_td_between1_and10_count.sum()/count,
                group+'_ge10_fraction':subset.abs_td_ge10_count.sum()/count})
        compact.append(q)
    compact=pd.DataFrame(compact);compact.to_csv(out/'gradient_distribution_sources.csv',index=False)
    hold=pd.read_csv(args.diagnostics/'fixed_holdout.csv');hold['abs_error']=hold.terminal_error_q_minus_r.abs()
    hold.groupby(['model_run','model_beta','env_steps']).agg(n=('q','size'),q=('q','mean'),reward=('reward','mean'),signed_error=('terminal_error_q_minus_r','mean'),mae=('abs_error','mean')).reset_index().to_csv(out/'fixed_holdout_summary.csv',index=False)
    samples=pd.read_csv(args.diagnostics/'samples.csv');signed=[]
    terminal=samples[samples.group=='terminal'].copy()
    terminal['terminal_error']=terminal.q-terminal.reward
    terminal['terminal_abs_error']=terminal.terminal_error.abs()
    terminal.groupby(['source','source_run','model_run','model_beta','env_steps'],sort=False).agg(
        n=('q','size'),q=('q','mean'),reward=('reward','mean'),signed_error=('terminal_error','mean'),mae=('terminal_abs_error','mean'),
        p05_error=('terminal_error',lambda v:v.quantile(.05)),p95_error=('terminal_error',lambda v:v.quantile(.95))
    ).reset_index().to_csv(out/'terminal_error_sources.csv',index=False)
    for key,data in samples.groupby(['source','source_run','model_run','model_beta','env_steps','group'],sort=False):
        for quantity in ('q','target','td_signed'):
            signed.append({**dict(zip(['source','source_run','model_run','model_beta','env_steps','group'],key)),'quantity':quantity,
                           'n':len(data),'mean':data[quantity].mean(),'p05':data[quantity].quantile(.05),'p50':data[quantity].median(),'p95':data[quantity].quantile(.95)})
    pd.DataFrame(signed).to_csv(out/'signed_q_target_td.csv',index=False)
    qfig,qaxes=plt.subplots(2,2,figsize=(12,8))
    a=samples[samples.source=='A_fixed128']
    for group,axis in zip(('terminal','nonterminal'),qaxes[0]):
        for run,data in a[a.group==group].groupby('model_run',sort=False):
            values=data.groupby('env_steps').q.mean();axis.plot(values.index,values.values,label=run)
        axis.set(title='Q(s,a) firmado — '+group,xlabel='Pasos');axis.grid(alpha=.2);axis.legend(fontsize=6)
    for run,data in gap.groupby('run',sort=False):
        values=data.groupby('env_steps').greedy_policy_return_gap.mean();qaxes[1,0].plot(values.index,values.values,marker='o',label=run)
    qaxes[1,0].set(title='Brecha frente al retorno greedy observado',xlabel='Pasos');qaxes[1,0].legend(fontsize=6);qaxes[1,0].grid(alpha=.2)
    for run,data in hold.groupby('model_run',sort=False):
        values=data.groupby('env_steps').abs_error.mean();qaxes[1,1].plot(values.index,values.values,label=run)
    qaxes[1,1].set(title='MAE en las MISMAS 256 terminales previas',xlabel='Pasos');qaxes[1,1].legend(fontsize=6);qaxes[1,1].grid(alpha=.2)
    qfig.suptitle('Fuentes fijas A/D y retorno greedy: la brecha no es error exacto respecto a Q óptimo')
    qfig.tight_layout();qfig.savefig(out/'fixed_q_and_gap.png',dpi=150);plt.close(qfig)
    gfig,gaxes=plt.subplots(2,2,figsize=(12,8))
    for row,source in enumerate(('B_fixed_baseline','C_new_replay')):
        data=compact[(compact.source==source)&(compact.model_beta==compact.evaluated_beta)]
        for col,field in enumerate(('terminal_weighted_gradient_norm','nonterminal_weighted_gradient_norm')):
            ax=gaxes[row,col];ax.bar(np.arange(len(data)),data[field]);ax.set(title=source+' — '+('terminales' if col==0 else 'no terminales'),
                xticks=np.arange(len(data)),xticklabels=[s.replace('_100k','') for s in data.model_run],ylabel='Media de norma de contribución p_g ∇loss_g')
            ax.tick_params(axis='x',rotation=20);ax.grid(axis='y',alpha=.2)
    gfig.suptitle('Objetivo propio de cada modelo; las fuentes B y C se comparan por separado')
    gfig.tight_layout();gfig.savefig(out/'gradient_sources.png',dpi=150);plt.close(gfig)
    print(out)


if __name__=='__main__':main()
