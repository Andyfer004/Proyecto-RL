"""Tablas y figuras del diagnóstico acotado; no ejecuta actualizaciones ni entornos."""
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
from thruster_landing.checkpoints import read_checkpoint
from thruster_landing.envs import EnvConfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path,nargs='?',default=Path('artifacts/offline_interference_v1'))
    args=parser.parse_args();root=args.directory
    metadata=json.loads((root/'metadata.json').read_text())
    g=pd.read_csv(root/'group_gradients.csv');m=pd.read_csv(root/'minibatches.csv')
    s=pd.read_csv(root/'sample_values.csv');fit=pd.read_csv(root/'supervised_fit.csv')
    i=pd.read_csv(root/'interference.csv');p=pd.read_csv(root/'problem_samples.csv')
    rows=[]
    for run,values in g.groupby('run',sort=False):
        c=m[m.run==run].cosine_shared_groups.dropna()
        row={'run':run,'replay_terminal_fraction':metadata['manifests'][run]['terminal_replay_fraction'],
             'cosine_mean':c.mean(),'cosine_std':c.std(),'cosine_n_defined':len(c),
             'cosine_negative_count':int((c<0).sum()),'empty_terminal_batches':int((m[m.run==run].terminal_count==0).sum())}
        for group,values in values.groupby('group',sort=False):
            prefix=group+'_';nonempty=values[values['count']>0]
            sampled=s[(s.run==run)&(s.group==group)]
            row.update({prefix+'draws':int(values['count'].sum()),prefix+'sample_fraction':values['count'].sum()/(128*64),
                        prefix+'q_mean':sampled.q.mean(),prefix+'target_mean':sampled.target.mean(),
                        prefix+'td_signed_mean':sampled.td_signed.mean(),prefix+'loss_per_sample_mean':sampled.loss_per_sample.mean(),
                        prefix+'huber_linear_fraction':values.huber_linear_count.sum()/values['count'].sum(),
                        prefix+'abs_td_ge_10_fraction':values.huber_abs_td_ge_10_count.sum()/values['count'].sum(),
                        prefix+'gradient_group_conditional_mean_norm':nonempty.shared_gradient_mean_norm.mean(),
                        prefix+'weighted_gradient_norm_mean':values.shared_weighted_contribution_norm.mean(),
                        prefix+'projection_share_mean':values.projection_share_full_gradient.mean()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(root/'contributions_compact.csv',index=False)
    result=[]
    for (run,mode,branch),values in i.groupby(['run','optimizer_mode','branch'],sort=False):
        result.append({'run':run,'optimizer_mode':mode,'branch':branch,'n':len(values),
            'before_mae':values.before_mae.iloc[0],'delta_mae_mean':values.delta_mae.mean(),
            'delta_mae_std':values.delta_mae.std(),'p05':values.delta_mae.quantile(.05),'p95':values.delta_mae.quantile(.95),
            'fraction_mae_increased':(values.delta_mae>0).mean()})
    pd.DataFrame(result).to_csv(root/'interference_compact.csv',index=False)
    # Verificar el evento y la ÚLTIMA observación contra logs originales de episodio.
    verified=[];checks=[];c=EnvConfig()
    for run,values in p.groupby('run',sort=False):
        saved=read_checkpoint(metadata['manifests'][run]['source_checkpoint'],'resume')['agent']
        replay=saved['replay'];capacity=replay['capacity'];size=replay['size']
        first=replay['position'] if size==capacity else 0
        starts=saved['env_steps']-size+1
        episodes=pd.read_csv(Path('artifacts')/run/'episodes.csv').set_index('env_steps')
        for row in values.to_dict('records'):
            index=int(row['replay_index']);step=starts+(index-first)%capacity
            episode=episodes.loc[step];state=json.loads(episode.final_state)
            encoded=np.array([state['x']/c.position_scale,(state['y']-c.foot_drop)/c.height_scale,
                state['vx']/c.velocity_scale,state['vy']/c.velocity_scale,np.sin(state['theta']),np.cos(state['theta']),
                state['omega']/c.omega_scale,state['gas']/c.gas_capacity,episode.wind_force_N/c.wind_scale],dtype=np.float32)
            error=float(np.max(np.abs(encoded-replay['arrays']['next_observations'][index].numpy())))
            assert error<1e-6
            expected={'body_contact':'body_impact','missed_platform':'missed_platform','invalid_contact':'platform_contact'}
            event=row['event_from_last_observation']
            if episode.outcome=='out_of_bounds':assert event in ('ceiling_exit','horizontal_exit')
            else:assert event==expected[episode.outcome]
            row.update(global_step=step,logged_outcome=episode.outcome,last_obs_encoding_max_error=error)
            verified.append(row)
        checks.append({'run':run,'n_verified':len(values),'max_last_obs_encoding_error':max(r['last_obs_encoding_max_error'] for r in verified if r['run']==run)})
    pd.DataFrame(verified).to_csv(root/'problem_samples_verified.csv',index=False)
    (root/'event_link_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    regions=[]
    for run,values in p.groupby('run',sort=False):
        for field in ('event_from_last_observation','action'):
            for label,subset in values.groupby(field,sort=False):
                regions.append({'run':run,'grouping':field,'group':label,'n':len(subset),
                    'fraction_samples':len(subset)/len(values),'mean_signed_error':subset.terminal_error.mean(),
                    'mae':subset.terminal_abs_error.mean(),'fraction_total_absolute_error':subset.terminal_abs_error.sum()/values.terminal_abs_error.sum(),
                    'mean_obs_norm_s':subset.norm_obs_s.mean(),'mean_speed_s':subset.speed_s.mean()})
    pd.DataFrame(regions).to_csv(root/'problem_regions.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for (run,split),values in fit.groupby(['run','split'],sort=False):
        axes[0 if split=='train' else 1].plot(values['update'],values.mae,label=run)
    for ax,label in zip(axes,('train','heldout (s,a) separados')):
        ax.set(title='Ajuste solo terminales — '+label,xlabel='Actualizaciones de regresión',ylabel='MAE Q(s,a) − recompensa',yscale='log')
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(root/'supervised_capacity.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,run in zip(axes.flat,metadata['protocol']['runs']):
        for mode,style in (('saved_Adam_state','o-'),('fresh_Adam_state','s--')):
            frame=pd.DataFrame(result);frame=frame[(frame.run==run)&(frame.optimizer_mode==mode)].set_index('branch').loc[['terminal_only','nonterminal_only','mixed']]
            ax.errorbar(np.arange(3),frame.delta_mae_mean,yerr=frame.delta_mae_std,fmt=style,label=mode,capsize=4)
        ax.axhline(0,color='black',lw=.8);ax.set(title=run,xticks=np.arange(3),xticklabels=['solo T','solo N','mixto'],ylabel='Δ MAE en terminales reservadas; media ± SD')
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('32 minibatches pareados: cada rama comienza con los mismos pesos y estado Adam')
    fig.tight_layout();fig.savefig(root/'one_step_interference.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,run in zip(axes.flat,metadata['protocol']['runs']):
        subset=p[p.run==run]
        for event,group in subset.groupby('event_from_last_observation'):
            ax.scatter(group.y_s,group.terminal_error,s=4,alpha=.3,label=event)
        ax.set(title=run,xlabel='Altura de s ANTES de aplicar a (m)',ylabel='Q(s,a) − recompensa terminal')
        ax.grid(alpha=.2);ax.legend(fontsize=6)
    fig.suptitle('La etiqueta terminal se ajusta en (s,a), no en la última observación s′')
    fig.tight_layout();fig.savefig(root/'terminal_regions.png',dpi=160);plt.close(fig)
    print(root)


if __name__=='__main__':main()
