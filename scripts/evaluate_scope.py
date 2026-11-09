"""Execute the frozen final protocol once, without training or checkpoint selection."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time

from thruster_landing.envs import EnvConfig
from thruster_landing.agents.pid import PIDConfig
from thruster_landing.evaluation import write_csv
from thruster_landing.scope_evaluation import (evaluate_method,load_method,verify_protocol,sha256)
from prepare_scope_evaluation import exposure


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=Path('configs/scope_final_v1_protocol.json'))
    p.add_argument('--output',type=Path,default=Path('artifacts/scope_alignment_v1/final'))
    a=p.parse_args();protocol=json.loads(a.protocol.read_text());verify_protocol(protocol)
    root=Path(protocol['artifacts_root']);record=json.loads((root/'protocol_sha256.json').read_text())
    assert record[str(a.protocol)]==sha256(a.protocol)
    a.output.mkdir(parents=True,exist_ok=False)
    config=EnvConfig.from_json(protocol['env_config']);pid_config=PIDConfig.from_json(protocol['pid_config'])
    reserved=json.loads(Path(protocol['reserved_manifest']).read_text());complement=json.loads(Path(protocol['complement_manifest']).read_text())
    cases=[]
    for case in reserved['scenarios']:
        cases.append({**case,'suite':'reserved','exposure':exposure(case['options']['initial_state'],case['options']['wind_force'])})
    cases.extend(complement['scenarios']);manifest=dict(split='test',scenarios=cases)
    audit=[];all_rows=[];start=time.perf_counter()
    for method in protocol['methods']:
        verify_protocol(protocol)
        agent=load_method(method,config,pid_config)
        rows,traces,checks=evaluate_method(agent,method,config,manifest)
        write_csv(a.output/(method['id']+'.episodes.csv'),rows)
        write_csv(a.output/(method['id']+'.traces.csv'),traces)
        audit.append(checks);all_rows.extend(rows)
        print(method['id'],'success',sum(r['success'] for r in rows),'/',len(rows),flush=True)
    verify_protocol(protocol)
    write_csv(a.output/'episodes.csv',all_rows)
    (a.output/'execution.json').write_text(json.dumps(dict(protocol_sha256=sha256(a.protocol),
        completed_at_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-start,
        episodes=len(all_rows),environment_steps=sum(r['steps'] for r in all_rows),
        training_updates=0,checkpoints_changed=False,methods=audit),indent=2)+'\n')

if __name__=='__main__':main()
