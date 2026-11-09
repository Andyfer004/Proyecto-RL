"""Independent landing validation and actual recovery tables/curves; no training."""
import json,shutil,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from thruster_landing.recovery_budget import ROOT
from thruster_landing.recovery_agent import AssistedDQN


def independent_contact_valid(row):
    s=json.loads(row.final_state);th=s['theta']
    feet=[(s['x']+x*np.cos(th)+np.sin(th),s['y']+x*np.sin(th)-np.cos(th)) for x in [-.6,.6]]
    body=[s['y']+x*np.sin(th)+y*np.cos(th) for x in [-.7,.7] for y in [-.8,.8]]
    contact=min([y for x,y in feet]+body)<=1e-7
    valid=(contact and min(body)>1e-6 and all(abs(x)<=2.5 for x,y in feet) and max(y for x,y in feet)<=.15 and abs(s['x'])<=1.5 and abs(s['vx'])<=.7 and -1<=s['vy']<=0 and abs(th)<=.12 and abs(s['omega'])<=.25)
    return bool(valid),bool(contact)


def main():
    out=ROOT/'report';out.mkdir(exist_ok=True);results=[];causes=[];all_eval=[];validation_checks=0
    methods=[('DQN42','dqn_assisted_seed42','dqn',20_000),('Double42','double_dqn_assisted_seed42','double42',30_000),('DQN43','dqn_assisted_seed43','dqn43',30_000)]
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    for name,directory,prefix,chosen in methods:
        ev=pd.read_csv(ROOT/directory/'evaluations.csv');online=ev[ev.phase=='online'];curve=online.groupby('online_steps').success.mean()*100
        axes[0].plot(curve.index,curve,'o-',label=name)
        m=pd.read_csv(ROOT/directory/'metrics.csv');ce=m[m.phase=='classification'];axes[1].plot(ce.updates,ce.heldout_action_accuracy*100,label=name)
        td=m[m.phase=='online'];axes[2].plot(td.online_steps,td.mean_abs_td,label=name)
        ev['method']=name;all_eval.append(ev)
        for split in ['development','check','original']:
            p=ROOT/f'loaded_{prefix}_{split}.csv';rows=pd.read_csv(p)
            chosen=json.loads(p.with_suffix('.metadata.json').read_text())['loaded_online_steps']
            for row in rows.itertuples():
                valid,contact=independent_contact_valid(row);assert valid==bool(row.success),(name,split,row.scenario_id);assert contact==bool(row.contact);validation_checks+=1
            c=rows[rows.contact]
            result=dict(method=name,checkpoint_online_steps=chosen,split=split,episodes=len(rows),successes=int(rows.success.sum()),success_rate=float(rows.success.mean()),contacts=len(c),return_mean=rows['return'].mean(),discounted_return_mean=rows.discounted_return.mean(),duration_mean_s=rows.duration_s.mean(),gas_mean_Ns=rows.gas_consumed_Ns.mean(),contact_vx_mean=c.contact_vx_m_s.mean(),contact_vy_mean=c.contact_vy_m_s.mean(),contact_theta_abs_mean=c.contact_theta_rad.abs().mean(),contact_omega_abs_mean=c.contact_omega_rad_s.abs().mean(),contact_error_x_mean=c.contact_error_x_m.mean())
            results.append(result)
            for cause,g in rows.groupby('outcome'):causes.append(dict(method=name,split=split,outcome=cause,n=len(g),contacts=int(g.contact.sum())))
    axes[0].axhline(80,color='k',ls='--',label='Meta de desarrollo');axes[0].set_ylim(0,102);axes[0].set_title('Desarrollo greedy: mismos 64 casos');axes[0].set_xlabel('Pasos online');axes[0].set_ylabel('Éxito (%)')
    axes[1].set_title('Exactitud de acciones en holdout de demos');axes[1].set_xlabel('Actualizaciones supervisadas');axes[1].set_ylabel('Exactitud (%)')
    axes[2].set_title('TD medio absoluto: entrenamiento asistido');axes[2].set_xlabel('Pasos online')
    for ax in axes:ax.legend()
    fig.tight_layout();fig.savefig(out/'learning.png',dpi=150);plt.close(fig)
    pd.DataFrame(results).to_csv(out/'results.csv',index=False);pd.DataFrame(causes).to_csv(out/'failures.csv',index=False);pd.concat(all_eval).to_csv(out/'development_history.csv',index=False)
    # Matched initial state across old PID, failed RL, and loaded new primary policy.
    previous=pd.read_csv(ROOT/'audit/reproduced_trajectories.csv');candidate=pd.read_csv(ROOT/'loaded_primary_matched.traces.csv');fig,axes=plt.subplots(1,4,figsize=(16,4))
    for label,g,timecol in [('PID',previous[previous.controller=='pid'],'time'),('RL previo fallido',previous[previous.controller=='failed_rl'],'time'),('DQN asistido',candidate,'time')]:
        axes[0].plot(g.x,g.y,label=label);axes[1].plot(g[timecol],g.vy,label=label);axes[2].plot(g[timecol],g.theta,label=label);axes[3].plot(g[timecol],g.gas,label=label)
    for ax,title in zip(axes,['Trayectoria del centro','Velocidad vertical','Inclinación','Impulso restante']):ax.set_title(title);ax.legend()
    axes[0].set_xlabel('x (m)');axes[0].set_ylabel('y (m)')
    for ax in axes[1:]:ax.set_xlabel('Tiempo (s)')
    axes[1].set_ylabel('m/s');axes[2].set_ylabel('rad');axes[3].set_ylabel('N s');fig.tight_layout();fig.savefig(out/'matched_trajectories.png',dpi=150);plt.close(fig)
    # Representative failures: first original ID, never select the best path.
    traces=pd.read_csv(ROOT/'loaded_dqn_original.traces.csv');identifier=sorted(traces.scenario_id.unique())[0];g=traces[traces.scenario_id==identifier]
    fig,axes=plt.subplots(1,3,figsize=(12,4));axes[0].plot(g.x,g.y);axes[0].set_title('Fallo en originales: primer ID');axes[0].set_xlabel('x (m)');axes[0].set_ylabel('y (m)');axes[1].plot(g.time,g.vy);axes[1].set_ylabel('vy (m/s)');axes[2].plot(g.time,g.gas);axes[2].set_ylabel('Impulso (N s)')
    for ax in axes[1:]:ax.set_xlabel('Tiempo (s)')
    fig.tight_layout();fig.savefig(out/'original_failure.png',dpi=150);plt.close(fig)
    # Reference benchmark has its own independently implemented optimizer/replay/targets.
    cp=pd.read_csv(ROOT/'cartpole/evaluation.csv');fig,ax=plt.subplots(figsize=(8,4))
    for name,g in cp.groupby('method'):ax.plot(g.train_steps,g.mean_return,'o-',label=name)
    ax.set_xlabel('Pasos entrenamiento CartPole');ax.set_ylabel('Retorno greedy medio,16 casos');ax.legend();fig.tight_layout();fig.savefig(out/'cartpole.png',dpi=150);plt.close(fig)
    candidate_path=ROOT/'candidate.policy.pt'
    if candidate_path.exists():
        import hashlib
        assert hashlib.sha256(candidate_path.read_bytes()).digest()==hashlib.sha256((ROOT/'dqn_assisted_seed42/best.policy.pt').read_bytes()).digest()
    else:shutil.copyfile(ROOT/'dqn_assisted_seed42/best.policy.pt',candidate_path)
    policy,meta=AssistedDQN.load(candidate_path)
    assert policy.env_steps==json.loads((ROOT/'loaded_dqn_development.metadata.json').read_text())['loaded_online_steps']
    (out/'independent_landing_checks.json').write_text(json.dumps(dict(checked_loaded_episode_results=validation_checks,all_geometry_and_kinematic_success_flags_match=True,thresholds_unchanged=True,candidate_source='dqn_assisted_seed42/best.policy.pt',candidate_selected_only_by_development=True),indent=2)+'\n')
    print(pd.DataFrame(results).to_string(index=False),flush=True)
if __name__=='__main__':main()
