"""Common evaluation tables, paired comparisons and all-case diagnostic plots."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from thruster_landing.scope_evaluation import group_statistics,verify_protocol
from thruster_landing.evaluation import write_csv
from report_recovery import independent_contact_valid


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,default=Path('artifacts/scope_alignment_v1/final'));a=p.parse_args()
    protocol=json.loads(Path('configs/scope_final_v1_protocol.json').read_text());verify_protocol(protocol)
    rows=pd.read_csv(a.results/'episodes.csv');summaries=group_statistics(rows.where(pd.notna(rows),None).to_dict('records'))
    write_csv(a.results/'summary.csv',summaries)
    expected={m['id'] for m in protocol['methods']}
    for identifier,group in rows.groupby('scenario_id'):
        assert set(group.method)==expected and len(group)==len(expected)
        assert group.initial_state.nunique()==1 and group.wind_force_N.nunique()==1
    failures=[]
    for row in rows.itertuples():
        valid,contact=independent_contact_valid(row)
        assert valid==row.success and contact==row.contact,(row.method,row.scenario_id)
        if not row.success:
            violations=[]
            if row.contact:
                if abs(row.contact_vx_m_s)>.7:violations.append('vx')
                if not -1<=row.contact_vy_m_s<=0:violations.append('vy')
                if abs(row.contact_theta_rad)>.12:violations.append('theta')
                if abs(row.contact_omega_rad_s)>.25:violations.append('omega')
                if row.contact_error_x_m>1.5:violations.append('x')
            failures.append(dict(method=row.method,suite=row.suite,condition=row.condition,scenario_id=row.scenario_id,
                cause=row.termination_cause,contact=row.contact,kinematic_violations=','.join(violations)))
    write_csv(a.results/'failures.csv',failures)
    primary=['pid','dqn_assisted42','double_assisted42','dqn_scratch','double_dqn_scratch']
    pairs=[]
    for method in primary[1:]:
        merge=rows[rows.method==method].merge(rows[rows.method=='pid'],on='scenario_id',suffixes=('_method','_pid'))
        for row in merge.itertuples():
            both=row.success_method and row.success_pid
            pairs.append(dict(method=method,scenario_id=row.scenario_id,suite=row.suite_method,condition=row.condition_method,
                method_success=row.success_method,pid_success=row.success_pid,both_success=both,
                return_difference=row.return_method-row.return_pid,
                success_only_gas_difference_Ns=row.gas_consumed_Ns_method-row.gas_consumed_Ns_pid if both else None,
                success_only_contact_x_difference_m=row.contact_error_x_m_method-row.contact_error_x_m_pid if both else None))
    write_csv(a.results/'paired_vs_pid.csv',pairs)
    winds=rows[(rows.suite=='complement')&(rows.condition.str.startswith('wind'))]
    windpairs=[]
    for (method,pairid),group in winds.groupby(['method','paired_wind_id']):
        control=group[group.condition=='wind_control'].iloc[0]
        for row in group.itertuples():
            if row.condition=='wind_control':continue
            windpairs.append(dict(method=method,pair_id=pairid,wind_N=row.wind_force_N,success=row.success,
                calm_success=bool(control.success),success_change=int(row.success)-int(control.success),
                final_x_error_change_m=row.final_error_x_m-control.final_error_x_m,
                gas_difference_only_both_success_Ns=row.gas_consumed_Ns-control.gas_consumed_Ns if row.success and control.success else None))
    write_csv(a.results/'paired_wind.csv',windpairs)
    totals=[]
    for (method,suite),g in rows.groupby(['method','suite']):
        totals.append(dict(method=method,suite=suite,n=len(g),successes=int(g.success.sum()),contacts=int(g.contact.sum()),
            **{cause:int((g.termination_cause==cause).sum()) for cause in ['success','invalid_contact','body_contact','missed_platform','ceiling_exit','side_exit','time_limit']}))
    pd.DataFrame(totals).to_csv(a.results/'totals.csv',index=False)
    # All predefined groups and methods are retained, including zero success.
    names=[m['id'] for m in protocol['methods']];columns=list(dict.fromkeys(rows.suite+':'+rows.condition))
    rate=rows.assign(group=rows.suite+':'+rows.condition).pivot_table(index='method',columns='group',values='success',aggfunc='mean').reindex(index=names,columns=columns)*100
    fig,ax=plt.subplots(figsize=(16,5));im=ax.imshow(rate,vmin=0,vmax=100,cmap='viridis',aspect='auto')
    ax.set_xticks(range(len(columns)),columns,rotation=45,ha='right',fontsize=8);ax.set_yticks(range(len(names)),names)
    for i in range(len(names)):
        for j in range(len(columns)):ax.text(j,i,f'{rate.iloc[i,j]:.0f}',ha='center',va='center',color='white' if rate.iloc[i,j]<60 else 'black',fontsize=8)
    fig.colorbar(im,ax=ax,label='Éxito (%)');fig.tight_layout();fig.savefig(a.results/'success_by_group.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for name in names:
        c=rows[(rows.method==name)&rows.contact];axes[0].scatter(c.contact_vx_m_s,c.contact_vy_m_s,s=9,alpha=.45,label=name)
        vals=[rows[(rows.method==name)&(rows.success==success)].gas_consumed_Ns.mean() for success in [True,False]]
        axes[1].plot(['Éxitos','Fracasos'],vals,'o-',label=name)
    axes[0].axvline(-.7,ls='--',color='k');axes[0].axvline(.7,ls='--',color='k');axes[0].axhline(-1,ls='--',color='k');axes[0].axhline(0,ls='--',color='k')
    axes[0].set_xlabel('vx al contacto (m/s)');axes[0].set_ylabel('vy al contacto (m/s)');axes[0].legend(fontsize=7)
    axes[1].set_ylabel('Impulso medio consumido (N s)');axes[1].set_title('Condicional al resultado; no mide superioridad');axes[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(a.results/'contact_and_fuel.png',dpi=160);plt.close(fig)
    ids=['test-vertical-000','complement-new_dynamics-000','complement-wind_p40-000']
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for method in primary:
        data=pd.read_csv(a.results/(method+'.traces.csv'))
        for col,identifier in enumerate(ids):
            g=data[data.scenario_id==identifier];axes[0,col].plot(g.x,g.y,label=method);axes[1,col].plot(g.time,g.theta,label=method)
    for col,identifier in enumerate(ids):
        axes[0,col].set_title(identifier);axes[0,col].set_xlabel('x (m)');axes[0,col].set_ylabel('y (m)');axes[1,col].set_xlabel('Tiempo (s)');axes[1,col].set_ylabel('theta (rad)')
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(a.results/'representative_trajectories.png',dpi=160);plt.close(fig)
    (a.results/'independent_checks.json').write_text(json.dumps(dict(episodes_checked=len(rows),same_case_observations_and_gas_for_all_methods=True,
        independently_confirmed_success_and_contact_flags=True,representative_ids=ids,representative_rule='Fixed first IDs, no selection by reward/success'),indent=2)+'\n')
    print(pd.DataFrame(totals).to_string(index=False))

if __name__=='__main__':main()
