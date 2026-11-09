"""Cross-evaluate final models on identical uniform batches of each replay source."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from thruster_landing.checkpoints import read_checkpoint
from thruster_landing.q_diagnostics import load_model
from thruster_landing.offline_diagnostics import subset
from thruster_landing.beta_diagnostics import measure_groups
from thruster_landing.experiments import digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('artifacts/replay150k_weighted_seed42_100k_v1'));a=p.parse_args();out=a.root/'replay_diagnostics';out.mkdir(exist_ok=False)
    group_rows=[];samples=[];batch_rows=[];sources={}
    for alg in ['dqn','double_dqn']:
        paths={50000:Path('artifacts/exploration_comparison_level1_seed42_100k_v1')/f'{alg}_weighted_variant/last.resume.pt',150000:a.root/f'{alg}_continued100k/last.resume.pt'}
        for source_capacity,source in paths.items():
            arrays=read_checkpoint(source,'resume')['agent']['replay']['arrays'];rng=np.random.default_rng(20261107);indices=[rng.choice(len(arrays['rewards']),64,replace=False).tolist() for _ in range(128)]
            sources[f'{alg}_{source_capacity}']=dict(checkpoint_sha256=digest(source),indices=indices)
            for model_capacity,model in paths.items():
                agent,_=load_model(model)
                for n,index in enumerate(indices):
                    batch=subset(arrays,index);groups,full,detail=measure_groups(agent,batch,10.)
                    common=dict(algorithm=alg,model_capacity=model_capacity,source_capacity=source_capacity,batch=n)
                    group_rows += [{**common,**row} for row in groups];batch_rows.append({**common,**full})
                    for i,source_index in enumerate(index):
                        row={**common,'source_index':source_index,'terminal':bool(batch['terminated'][i]),'terminal_success':bool(batch['terminated'][i]) and float(batch['rewards'][i])>0};row.update({k:float(v[i]) for k,v in detail.items()});samples.append(row)
        print(alg,'cross-replay diagnostics done',flush=True)
    pd.DataFrame(group_rows).to_csv(out/'gradients.csv',index=False);pd.DataFrame(batch_rows).to_csv(out/'batches.csv',index=False);df=pd.DataFrame(samples);df.to_csv(out/'samples.csv',index=False)
    summary=[]
    for key,g in df.groupby(['algorithm','model_capacity','source_capacity','terminal']):
        for field in ['q','target','td','reward','loss']:
            v=g[field];summary.append(dict(zip(['algorithm','model_capacity','source_capacity','terminal'],key))|dict(quantity=field,n=len(v),mean=v.mean(),p05=v.quantile(.05),p50=v.quantile(.5),p95=v.quantile(.95),mean_abs=v.abs().mean()))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False);(out/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
if __name__=='__main__':main()
