"""Actual equal-budget comparison; no training, environment rollouts or test evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from thruster_landing.checkpoints import read_checkpoint,load_inference


def initial_observation(encoded):
    s=json.loads(encoded)
    return np.array([s['x']/12,(s['y']-1)/25,s['vx']/10,s['vy']/10,np.sin(s['theta']),np.cos(s['theta']),s['omega']/3,s['gas']/18000,0.],dtype=np.float32)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path('artifacts/exploration_comparison_level1_seed42_100k_v1'));args=parser.parse_args()
    out=args.root/'report';out.mkdir(exist_ok=True)
    results=[];progress=[];action_rows=[];contact_rows=[];outcome_rows=[];gaps=[];success_rows=[];timing=[];all_val=[];all_eps=[];all_updates=[]
    fig,axes=plt.subplots(2,4,figsize=(16,8),sharex=True)
    traj,axtraj=plt.subplots(2,2,figsize=(12,8));states,axstates=plt.subplots(2,3,figsize=(14,8))
    examples,axexamples=plt.subplots(2,3,figsize=(14,8))
    for ai,alg in enumerate(['dqn','double_dqn']):
        for method in ['uniform','weighted_variant']:
            run=args.root/f'{alg}_{method}';meta=json.loads((run/'metadata.json').read_text());ep=pd.read_csv(run/'episodes.csv');val=pd.read_csv(run/'validation.csv');updates=pd.read_csv(run/'updates.csv');decisions=pd.read_csv(run/'decisions.csv')
            assert meta['env_steps']==100000 and len(decisions)==100000
            assert decisions.env_steps.tolist()==list(range(1,100001))
            assert sum(sum(c) for c in meta['decision_telemetry']['histograms'].values())==100000
            assert ep.source_level.eq(1).all()
            payload=read_checkpoint(run/'last.resume.pt','resume');replay=payload['agent']['replay']['arrays'];retained=int((replay['terminated']&(replay['rewards']>0)).sum())
            assert set(replay['source_levels'].tolist())=={1}
            success=ep[ep.success];first_training=int(success.env_steps.min()) if len(success) else None
            successful_eval=val[(val.source_level==1)&val.success];first_greedy=int(successful_eval.env_steps.min()) if len(successful_eval) else None
            assert retained==int(ep[(ep.env_steps>50000)&ep.success].success.sum())
            final=val[val.env_steps==100000];l1=final[final.source_level==1]
            for frame in [ep,val,updates]:frame['algorithm']=alg;frame['method']=method
            all_eps.append(ep);all_val.append(val);all_updates.append(updates)
            row=dict(algorithm=alg,method=method,env_steps=100000,training_episodes=len(ep),training_successes=int(ep.success.sum()),training_contacts=int(ep.contact.sum()),training_gas_exhausted=int(ep.gas_exhausted.sum()),first_training_success_step=first_training,first_greedy_l1_success_step=first_greedy,greedy_l1_successes=int(l1.success.sum()),greedy_l1_n=len(l1),greedy_all_successes=int(final.success.sum()),greedy_all_n=len(final),replay_successful_terminal_transitions=retained,training_mean_return=ep['return'].mean(),training_mean_discounted_return=ep.discounted_return.mean(),greedy_l1_mean_return=l1['return'].mean(),greedy_l1_mean_discounted_return=l1.discounted_return.mean(),training_duration_s=ep.duration_s.mean(),training_gas_Ns=ep.gas_consumed_Ns.mean(),partial_episode_steps=meta['partial_episode_steps'])
            results.append(row)
            timing.append(dict(algorithm=alg,method=method,wall_seconds=meta['wall_seconds'],steps_per_second=meta['segment_steps_per_second'],validation_seconds=meta['validation_seconds'],checkpoint_seconds=meta['checkpoint_seconds']))
            for s in success.itertuples():
                success_rows.append(dict(algorithm=alg,method=method,env_steps=s.env_steps,episode=s.episode,scenario_id=s.scenario_id,exploration_steps=s.exploration_steps,exploitation_steps=s.exploitation_steps,terminal_selection_mode=s.terminal_selection_mode,gas_consumed_Ns=s.gas_consumed_Ns,duration_s=s.duration_s,contact_vx_m_s=s.contact_vx_m_s,contact_vy_m_s=s.contact_vy_m_s,contact_theta_rad=s.contact_theta_rad,contact_omega_rad_s=s.contact_omega_rad_s,contact_error_x_m=s.contact_error_x_m,retained_at_100k=s.env_steps>50000))
            if method=='weighted_variant':
                for label,subset in [('Primer éxito',success),('Primer fallo',ep[~ep.success])]:
                    if subset.empty:continue
                    example=subset.sort_values('env_steps').iloc[0];d=decisions[decisions.episode==example.episode];initial=json.loads(example.initial_state)
                    axexamples[ai,0].plot([initial['x'],*d.x],[initial['y'],*d.y],label=label,color='tab:blue' if label=='Primer éxito' else 'tab:orange')
                    axexamples[ai,1].plot([0.,*d.duration_s],[initial['vy'],*d.vy],label=label,color='tab:blue' if label=='Primer éxito' else 'tab:orange')
                    axexamples[ai,2].plot([0.,*d.duration_s],[initial['theta'],*d.theta],label=label,color='tab:blue' if label=='Primer éxito' else 'tab:orange')
                for ci,title in enumerate(['Trayectoria','Velocidad vertical','Inclinación']):
                    axexamples[ai,ci].set_title(alg.replace('double_dqn','Double DQN').replace('dqn','DQN')+' variante — episodios con epsilon\n'+title,fontsize=10)
                    axexamples[ai,ci].set_xlabel('x (m)' if ci==0 else 'Tiempo (s)')
                    axexamples[ai,ci].set_ylabel(['y (m)','vy (m/s)','θ (rad)'][ci]);axexamples[ai,ci].legend()
                if success.empty:axexamples[ai,0].text(.03,.95,'Sin éxitos de entrenamiento',transform=axexamples[ai,0].transAxes,va='top')
            for mode,g in decisions.groupby('mode'):
                counts=g.action.value_counts()
                for action in range(8):
                    n=int(counts.get(action,0));assert n==meta['decision_telemetry']['histograms'][mode][action]
                    action_rows.append(dict(algorithm=alg,method=method,scope='training',mode=mode,source_level=1,env_steps=100000,action=action,n=n,total=len(g),frequency=n/len(g)))
            for step,v in val.groupby('env_steps'):
                training=ep[ep.env_steps<=step]
                for level,g in v.groupby('source_level'):
                    progress.append(dict(algorithm=alg,method=method,env_steps=step,source_level=level,n=len(g),successes=int(g.success.sum()),contacts=int(g.contact.sum()),gas_exhausted=int(g.gas_exhausted.sum()),return_mean=g['return'].mean(),discounted_return_mean=g.discounted_return.mean(),training_successes_so_far=int(training.success.sum()),training_episodes_so_far=len(training)))
                checkpoint=run/'initial.policy.pt' if step==0 else run/f'step_{int(step):08d}.policy.pt';model,_=load_inference(checkpoint)
                observations=torch.from_numpy(np.stack([initial_observation(s) for s in v.initial_state]))
                with torch.inference_mode():q=model.online(observations).max(dim=1).values.numpy()
                for (_,result),prediction in zip(v.iterrows(),q):gaps.append(dict(algorithm=alg,method=method,env_steps=step,source_level=result.source_level,scenario_id=result.scenario_id,q_initial_greedy=float(prediction),discounted_greedy_return=result.discounted_return,greedy_policy_return_gap=float(prediction)-result.discounted_return))
                trace_path=run/'initial_validation.traces.csv' if step==0 else run/f'validation_{int(step):08d}.traces.csv'
                trace=pd.read_csv(trace_path)
                for level in [0,1,2,3]:
                    ids=set(v.loc[v.source_level==level,'scenario_id']);t=trace[trace.scenario_id.isin(ids)&trace.action.notna()]
                    counts=t.action.astype(int).value_counts()
                    for action in range(8):action_rows.append(dict(algorithm=alg,method=method,scope='greedy_validation',mode='evaluation',source_level=level,env_steps=step,action=action,n=int(counts.get(action,0)),total=len(t),frequency=int(counts.get(action,0))/len(t)))
            for scope,data in [('training',ep),('greedy_final',final)]:
                for (level,cause),g in data.groupby(['source_level','termination_cause']):
                    outcome_rows.append(dict(algorithm=alg,method=method,scope=scope,source_level=level,cause=cause,n=len(g),contacts=int(g.contact.sum()),gas_exhausted=int(g.gas_exhausted.sum()),duration_s=g.duration_s.mean(),gas_Ns=g.gas_consumed_Ns.mean(),return_mean=g['return'].mean(),discounted_return_mean=g.discounted_return.mean(),**{col+'_mean':g[col].mean() for col in g if col.startswith('reward_')}))
                for (level,outcome),g in data.groupby(['source_level','outcome']):
                    c=g[g.contact];contact_rows.append(dict(algorithm=alg,method=method,scope=scope,source_level=level,outcome=outcome,n=len(g),contacts=len(c),no_contact=len(g)-len(c),**{col+'_mean':c[col].mean() for col in ['contact_error_x_m','contact_vx_m_s','contact_vy_m_s','contact_theta_rad','contact_omega_rad_s']},**{col+'_abs_mean':c[col].abs().mean() for col in ['contact_vx_m_s','contact_vy_m_s','contact_theta_rad','contact_omega_rad_s']}))
            curve=val[val.source_level==1].groupby('env_steps').success.mean()*100;axes[ai,0].plot(curve.index,curve,label=('Uniforme' if method=='uniform' else 'Variante ponderada'))
            axes[ai,1].plot(ep.env_steps,ep.success.astype(int).cumsum(),label=('Uniforme' if method=='uniform' else 'Variante ponderada'))
            axes[ai,2].plot(updates.env_steps,updates.replay_successful_terminal_transitions,label=('Uniforme' if method=='uniform' else 'Variante ponderada'))
            axes[ai,3].plot(updates.env_steps,updates.mean_abs_td_error,label=('Uniforme' if method=='uniform' else 'Variante ponderada'))
            final_traces=pd.read_csv(run/'final_validation.traces.csv')
            l1_ids=sorted(l1.scenario_id);identifier=l1_ids[0]
            for ci,level in enumerate([1,0]):
                ids=sorted(final.loc[final.source_level==level,'scenario_id']);t=final_traces[final_traces.scenario_id==ids[0]];axtraj[ai,ci].plot(t.x,t.y,label=('Uniforme' if method=='uniform' else 'Variante ponderada'));axtraj[ai,ci].set_title(f'{alg}, '+('L1' if level==1 else 'Original')+' — primer ID')
            t=final_traces[final_traces.scenario_id==identifier]
            for ci,field in enumerate(['y','vy','gas']):axstates[ai,ci].plot(t.time,t[field],label=('Uniforme' if method=='uniform' else 'Variante ponderada'));axstates[ai,ci].set_ylabel(alg+' '+field);axstates[ai,ci].set_xlabel('Tiempo (s)')
        for ci,title in enumerate(['Validación greedy L1 (%)','Éxitos de entrenamiento','Terminales de éxito en replay','Media |TD| de entrenamiento']):
            axes[ai,ci].set_title(alg.replace('double_dqn','Double DQN').replace('dqn','DQN')+'\n'+title,fontsize=11);axes[ai,ci].set_xlabel('Pasos de entorno');axes[ai,ci].legend()
        axes[ai,0].set_ylim(-2,100);axes[ai,0].set_yticks([0,20,40,60,80,100])
        for ax in axtraj[ai]:ax.plot([-2.5,2.5],[0,0],'k-',lw=3);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.legend()
        for ax in axstates[ai]:ax.legend()
    for ai in range(2):
        axes[ai,1].set_ylim(-.05,max(1,max(r['training_successes'] for r in results)*1.1))
        axes[ai,2].set_ylim(-.05,max(1,max(float(u.replay_successful_terminal_transitions.max()) for u in all_updates)*1.1))
    for plot,name in [(fig,'learning_curves.png'),(traj,'fixed_final_trajectories.png'),(states,'fixed_final_states.png'),(examples,'first_training_success_and_failure.png')]:plot.tight_layout();plot.savefig(out/name,dpi=150);plt.close(plot)
    for name,rows in [('results',results),('validation_progress',progress),('action_frequencies',action_rows),('contact_metrics',contact_rows),('by_outcome',outcome_rows),('greedy_return_gap',gaps),('training_successes',success_rows),('timings',timing)]:
        columns=None if rows else ['algorithm','method','env_steps','episode','scenario_id','exploration_steps','exploitation_steps','terminal_selection_mode','retained_at_100k']
        pd.DataFrame(rows,columns=columns).to_csv(out/(name+'.csv'),index=False)
    pd.concat(all_val).to_csv(out/'validation.csv',index=False);pd.concat(all_eps).to_csv(out/'episodes.csv',index=False);pd.concat(all_updates).to_csv(out/'updates.csv',index=False)
    matched=pd.DataFrame(progress).query('source_level==1').pivot(index=['algorithm','env_steps'],columns='method',values=['training_successes_so_far','successes'])
    matched.columns=[a+'__'+b for a,b in matched.columns];matched.reset_index().to_csv(out/'matched_budget.csv',index=False)
    print(pd.DataFrame(results).to_string(index=False),flush=True)
if __name__=='__main__':main()
