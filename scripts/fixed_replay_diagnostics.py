"""Congelar muestra o medir todos los checkpoints, sin entrenar ni ejecutar entornos."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import torch
from thruster_landing.evaluation import write_csv
from thruster_landing.q_diagnostics import create_probe,diagnose_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--create-from',type=Path)
    parser.add_argument('--runs',type=Path,nargs='*')
    parser.add_argument('--output',type=Path,default=Path('artifacts/lr_ablation_analysis'))
    args=parser.parse_args()
    if args.create_from:
        metadata=create_probe(args.create_from,args.probe)
        print(f"Muestra fija: {len(metadata['transition_ids'])} transiciones")
    if not args.runs:
        return
    probe=torch.load(args.probe,map_location='cpu',weights_only=True)
    rows,gradients=[],[]
    for run in args.runs:
        checkpoints=[run/'initial.policy.pt',*sorted(run.glob('step_*.resume.pt'))]
        for checkpoint in checkpoints:
            detail,gradient=diagnose_checkpoint(checkpoint,probe)
            for row in detail:
                row['run']=run.name
                rows.append(row)
            gradient['run']=run.name
            gradients.append(gradient)
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/'fixed_transitions.csv',rows)
    write_csv(args.output/'fixed_batch_gradients.csv',gradients)
    frame=pd.DataFrame(rows)
    summaries=[]
    for (run,step,group),values in frame.groupby(['run','env_steps','group'],sort=False):
        assert set(values.transition_id)==set(frame[(frame.run==run)&(frame.env_steps==0)&(frame.group==group)].transition_id)
        for column in ('q','reward','target','td_target_minus_q','bootstrap','terminal_error_q_minus_reward'):
            data=values[column].dropna()
            if not len(data):continue
            summaries.append({'run':run,'env_steps':step,'group':group,'quantity':column,'n':len(data),
                              'mean':data.mean(),'min':data.min(),'p05':data.quantile(.05),'p25':data.quantile(.25),
                              'p50':data.quantile(.50),'p75':data.quantile(.75),'p95':data.quantile(.95),'max':data.max()})
    write_csv(args.output/'fixed_transition_summary.csv',summaries)
    (args.output/'probe_manifest.json').write_text(json.dumps({'probe_sha256':hashlib.sha256(args.probe.read_bytes()).hexdigest(),
                 'metadata':probe['metadata'],'td_definition':'target - Q(s,a)',
                 'terminal_error_definition':'Q(s,a) - reward','fixed_batch_gradients':'offline measurements; no optimizer step'},indent=2)+'\n')
    print(args.output)


if __name__=='__main__':
    main()
