"""Summarize the second, frozen coverage intervention without environment steps."""
import json, shutil
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from thruster_landing.recovery_budget import ROOT
from report_recovery import independent_contact_valid

METHODS = [('DQN42','dqn_coverage_v2_seed42','coverage42'),
           ('Double42','double_coverage_v2_seed42','coverage_double42'),
           ('DQN43','dqn_coverage_v2_seed43','coverage43')]


def main():
    out=ROOT/'report_coverage';out.mkdir(exist_ok=True)
    summaries=[];failures=[];checked=0
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,directory,prefix in METHODS:
        ev=pd.read_csv(ROOT/directory/'evaluations.csv')
        online=ev[ev.phase=='online']
        for ax,condition in zip(axes,['level1','original']):
            series=online[online.condition==condition].groupby('online_steps').success.mean()*100
            ax.plot(series.index,series.values,'o-',label=name)
            ax.set_title('Desarrollo '+condition);ax.set_xlabel('Pasos online');ax.set_ylabel('Éxito greedy (%)');ax.set_ylim(0,102);ax.axhline(80,ls='--',color='k');ax.legend()
        for split in ['development','check','original']:
            path=ROOT/f'loaded_{prefix}_{split}.csv';df=pd.read_csv(path)
            meta=json.loads(path.with_suffix('.metadata.json').read_text())
            for row in df.itertuples():
                valid,contact=independent_contact_valid(row)
                assert valid==bool(row.success) and contact==bool(row.contact),(name,split,row.scenario_id)
                checked+=1
            for condition,g in df.groupby('condition'):
                c=g[g.contact]
                summaries.append(dict(method=name,checkpoint_steps=meta['loaded_online_steps'],split=split,condition=condition,n=len(g),successes=int(g.success.sum()),contacts=int(g.contact.sum()),mean_return=g['return'].mean(),mean_discounted_return=g.discounted_return.mean(),mean_duration_s=g.duration_s.mean(),mean_gas_Ns=g.gas_consumed_Ns.mean(),contact_vx_abs=c.contact_vx_m_s.abs().mean(),contact_vy=c.contact_vy_m_s.mean(),contact_x_abs=c.contact_error_x_m.mean(),contact_theta_abs=c.contact_theta_rad.abs().mean(),contact_omega_abs=c.contact_omega_rad_s.abs().mean()))
                for cause,h in g.groupby('outcome'):failures.append(dict(method=name,split=split,condition=condition,outcome=cause,n=len(h)))
    fig.tight_layout();fig.savefig(out/'learning.png',dpi=150);plt.close(fig)
    pd.DataFrame(summaries).to_csv(out/'results.csv',index=False);pd.DataFrame(failures).to_csv(out/'failures.csv',index=False)
    # Fixed representative rule: first case in each group, and first failure if any.
    episodes=pd.read_csv(ROOT/'loaded_coverage42_check.csv');trace=pd.read_csv(ROOT/'loaded_coverage42_check.traces.csv')
    ids=[sorted(g.scenario_id)[0] for _,g in episodes.groupby('condition')]
    bad=episodes[~episodes.success]
    if len(bad):ids.append(sorted(bad.scenario_id)[0])
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    for identifier in ids:
        g=trace[trace.scenario_id==identifier];r=episodes[episodes.scenario_id==identifier].iloc[0];label=identifier+' '+r.outcome
        axes[0].plot(g.x,g.y,label=label);axes[1].plot(g.time,g.vy,label=label);axes[2].plot(g.time,g.theta,label=label);axes[3].plot(g.time,g.gas,label=label)
    for ax,title in zip(axes,['Trayectoria del centro','vy (m/s)','θ (rad)','Gas (N s)']):ax.set_title(title);ax.legend(fontsize=6)
    axes[0].set_xlabel('x (m)');axes[0].set_ylabel('y (m)')
    for ax in axes[1:]:ax.set_xlabel('Tiempo (s)')
    fig.tight_layout();fig.savefig(out/'representative_trajectories.png',dpi=150);plt.close(fig)
    destination=ROOT/'candidate_coverage_v2.policy.pt'
    source=ROOT/'dqn_coverage_v2_seed42/best.policy.pt'
    if destination.exists():assert destination.read_bytes()==source.read_bytes()
    else:shutil.copyfile(source,destination)
    (out/'independent_checks.json').write_text(json.dumps(dict(episodes=checked,all_success_and_contact_flags_independently_confirmed=True,thresholds_unchanged=True,representative_ids=ids,selection='Grouped development only; check never used to select or tune.'),indent=2)+'\n')
    print(pd.DataFrame(summaries).to_string(index=False))

if __name__=='__main__':main()
