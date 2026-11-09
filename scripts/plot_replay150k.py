"""Standalone curves for capacity-only ablation, separated L1 and original validation."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path('artifacts/replay150k_weighted_seed42_100k_v1'));args=parser.parse_args();out=args.root/'analysis'
    p=pd.read_csv(out/'progress.csv');v=pd.read_csv(out/'validation.csv');q=pd.read_csv(out/'fixed_probe_values.csv')
    e=pd.read_csv(out/'episodes.csv');transitions=pd.read_csv(out/'successful_transition_exposure.csv');sampled=pd.read_csv(out/'successful_learning_samples.csv')
    def event_curve(events):
        delta={0:0,100000:0}
        for t,n in events:
            if 0<=t<=100000:delta[int(t)]=delta.get(int(t),0)+n
        times=sorted(delta);total=0;values=[]
        for t in times:total+=delta[t];values.append(total)
        return times,values
    fig,axes=plt.subplots(2,4,figsize=(16,8),sharex=True)
    for ai,alg in enumerate(['dqn','double_dqn']):
        for capacity in [50000,150000]:
            label=f'Replay {capacity//1000}k';g=p[(p.algorithm==alg)&(p.capacity==capacity)]
            wins=e[(e.algorithm==alg)&(e.capacity==capacity)&e.success]
            ts,ns=event_curve([(int(t),1) for t in wins.env_steps]);axes[ai,0].step(ts,ns,where='post',label=label)
            for ci,level in [(1,1),(2,0)]:
                curve=v[(v.algorithm==alg)&(v.capacity==capacity)&(v.source_level==level)].groupby('env_steps').success.mean()*100
                axes[ai,ci].plot(curve.index,curve,label=label)
            events=[(int(t),1) for t in wins.env_steps]+[(int(t)+capacity,-1) for t in wins.env_steps]
            ts,ns=event_curve(events);axes[ai,3].step(ts,ns,where='post',label=label)
        name='DQN' if alg=='dqn' else 'Double DQN'
        for ci,title in enumerate(['Éxitos de entrenamiento','Éxito greedy nivel 1 (%)','Éxito greedy originales (%)','Terminales de éxito retenidos']):
            ax=axes[ai,ci];ax.set_title(name+' — exploración ponderada\n'+title,fontsize=10);ax.set_xlabel('Pasos de entorno');ax.legend()
            if ci in [1,2]:ax.set_ylim(-2,100);ax.set_yticks([0,20,40,60,80,100])
            else:ax.set_ylim(-.05,max(1,p.training_successes.max()*1.1))
    fig.tight_layout();fig.savefig(out/'learning_retention.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True)
    for ai,alg in enumerate(['dqn','double_dqn']):
        for capacity in [50000,150000]:
            g=p[(p.algorithm==alg)&(p.capacity==capacity)];label=f'Replay {capacity//1000}k'
            tr=transitions[(transitions.algorithm==alg)&(transitions.capacity==capacity)]
            events=[(int(row.episode_end),1) for row in tr.itertuples()]+[(int(t)+capacity,-1) for t in tr.transition_step]
            ts,ns=event_curve(events);axes[ai,0].step(ts,ns,where='post',label=label)
            uses=sampled[(sampled.algorithm==alg)&(sampled.capacity==capacity)]
            for ci,filtered in [(1,uses[uses.terminal_success]),(2,uses)]:
                events=[(int(t),int(n)) for t,n in filtered.groupby('update_env_steps').size().items()]
                ts,ns=event_curve(events);axes[ai,ci].step(ts,ns,where='post',label=label)
        for ci,title in enumerate(['Transiciones de episodios exitosos retenidas','Muestreos de terminales exitosos','Muestreos de episodios exitosos (incl. terminales)']):
            axes[ai,ci].set_title(('DQN' if ai==0 else 'Double DQN')+'\n'+title,fontsize=10);axes[ai,ci].set_xlabel('Pasos de entorno');axes[ai,ci].legend();axes[ai,ci].set_ylim(bottom=-.05)
    fig.tight_layout();fig.savefig(out/'success_exposure.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ci,terminal in enumerate([True,False]):
        for capacity in [50000,150000]:
            g=q[(q.algorithm=='dqn')&(q.capacity==capacity)&(q.source=='early_success_episodes')&(q.terminal==terminal)].groupby('env_steps')[['q','target','td']].mean()
            axes[0,ci].plot(g.index,g.q,label=f'Q replay {capacity//1000}k');axes[0,ci].plot(g.index,g.target,'--',label=f'Target {capacity//1000}k')
            axes[1,ci].plot(g.index,g.td,label=f'Replay {capacity//1000}k')
        axes[0,ci].set_title('DQN — mismos estados y acciones\n'+('Terminales exitosas (3)' if terminal else 'No terminales de esos episodios (53)'),fontsize=10);axes[1,ci].set_ylabel('TD firmado = target − Q')
        for row in range(2):axes[row,ci].set_xlabel('Pasos de entrenamiento');axes[row,ci].legend()
    fig.tight_layout();fig.savefig(out/'fixed_success_q_td.png',dpi=150);plt.close(fig)
    # Fixed first ID; no best-episode selection. Plot COM paths, not support points.
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ai,alg in enumerate(['dqn','double_dqn']):
        for capacity in [50000,150000]:
            run=Path('artifacts/exploration_comparison_level1_seed42_100k_v1')/f'{alg}_weighted_variant' if capacity==50000 else args.root/f'{alg}_continued100k'
            traces=pd.read_csv(run/'final_validation.traces.csv')
            for ci,level in [(0,1),(1,0)]:
                selected=v[(v.algorithm==alg)&(v.capacity==capacity)&(v.env_steps==100000)&(v.source_level==level)]
                identifier=sorted(selected.scenario_id)[0];t=traces[traces.scenario_id==identifier];axes[ai,ci].plot(t.x,t.y,label=f'Replay {capacity//1000}k')
                axes[ai,ci].set_title(('DQN' if ai==0 else 'Double DQN')+' — '+('Nivel 1' if level==1 else 'Originales')+'\nPrimer ID, trayectoria del centro')
        for ax in axes[ai]:ax.plot([-2.5,2.5],[0,0],'k-',lw=3);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.legend()
    fig.tight_layout();fig.savefig(out/'fixed_final_trajectories.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
