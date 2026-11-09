"""Presentation-only figures from frozen CSV; no simulation or learning."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path('tmp/pdfs/mpl_cache').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

d=pd.read_csv('artifacts/scope_alignment_v1/final/episodes.csv')
methods=['pid','dqn_assisted42','double_assisted42','dqn_assisted43','dqn_scratch','double_dqn_scratch']
labels=['PID','DQN\nasistido42','Double\nasistido42','DQN\nasistido43','DQN\ndesde cero','Double\ndesde cero']
groups=list(dict.fromkeys(zip(d.suite,d.condition)))
group_labels=['Reservados: vertical','Reservados: desplazamiento','Reservados: inclinación','Reservados: viento','Complemento: L1','Complemento: altura','Complemento: desplazamiento nuevo','Complemento: dinámica nueva','Viento: control sin perturbación','Viento: -20 N','Viento: +20 N','Viento: -40 N','Viento: +40 N']
values=np.zeros((len(groups),len(methods)));counts={}
for i,(suite,condition) in enumerate(groups):
 for j,method in enumerate(methods):
  g=d[(d.suite==suite)&(d.condition==condition)&(d.method==method)];values[i,j]=100*g.success.mean();counts[i,j]=(int(g.success.sum()),len(g))
fig,ax=plt.subplots(figsize=(8.2,5.9));ax.imshow(values,cmap='YlGnBu',vmin=0,vmax=100,aspect='auto')
ax.set_yticks(range(len(groups)),group_labels,fontsize=9);ax.set_xticks(range(6),labels,fontsize=9);ax.xaxis.tick_top();ax.tick_params(length=0,pad=8)
for i in range(len(groups)):
 for j in range(6):
  n,total=counts[i,j];ax.text(j,i,f'{n}/{total}',ha='center',va='center',fontsize=9,color='white' if values[i,j]>60 else '#19303a')
for edge in ax.spines.values():edge.set_visible(False)
fig.tight_layout(pad=1);fig.savefig('tmp/pdfs/exitos_pdf.pdf');plt.close(fig)
colors=['#193a4b','#168d91','#cc7b35','#7666a7','#a94b55','#8a785b']
fig,axes=plt.subplots(2,1,figsize=(8.2,8.0),layout='constrained')
for method,label,color in zip(methods,labels,colors):
 g=d[(d.method==method)&d.contact];axes[0].scatter(g.contact_vx_m_s,g.contact_vy_m_s,s=7,alpha=.5,color=color,label=label.replace('\n',' '))
axes[0].set(xlabel='Velocidad horizontal al contacto (m/s)',ylabel='Velocidad vertical al contacto (m/s)')
for value in [-.7,.7]:axes[0].axvline(value,color='#333',linestyle='--',linewidth=.7)
for value in [-1,0]:axes[0].axhline(value,color='#333',linestyle='--',linewidth=.7)
axes[0].legend(fontsize=8,ncol=2,loc='lower left')
y=np.arange(6)
for i,flag in enumerate([True,False]):
 vals=[d[(d.method==m)&(d.success==flag)].gas_consumed_Ns.mean() for m in methods]
 axes[1].barh(y+(i-.5)*.3,vals,height=.27,color='#168d91' if flag else '#c8946a',label='Éxitos' if flag else 'Fracasos')
axes[1].set_yticks(y,[x.replace('\n',' ') for x in labels],fontsize=9);axes[1].invert_yaxis();axes[1].set_xlabel('Consumo medio condicionado al resultado (N·s)');axes[1].legend(fontsize=9)
axes[1].set_title('Un fracaso de bajo consumo no es un aterrizaje eficiente',fontsize=10)
for ax in axes:
 ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15,axis='x');ax.set_axisbelow(True)
fig.savefig('tmp/pdfs/contacto_pdf.pdf');plt.close(fig)
print('Presentation figures regenerated from existing2328 rows; no new episodes')
