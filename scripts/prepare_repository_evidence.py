"""Inventory the explicit public evidence selection, without modifying experiments."""
import json
from pathlib import Path
from thruster_landing.scope_evaluation import sha256,verify_protocol


def selected_evidence():
    protocol=json.loads(Path('configs/scope_final_v1_protocol.json').read_text())
    verify_protocol(protocol)
    files={Path(p) for p in protocol['sha256'] if p.startswith('artifacts/')}
    scope=Path('artifacts/scope_alignment_v1')
    files.update(p for p in (scope/'final').iterdir() if p.is_file() and p.suffix!='.log')
    for name in ['methodology_audit.json','protocol_sha256.json','animation_verification.json',
                 'preservation_verified.json','reserved_exclusion_check.json']:
        files.add(scope/name)
    recovery=Path('artifacts/recovery_v1')
    for folder in ['report','report_coverage','audit']:
        files.update(p for p in (recovery/folder).iterdir() if p.is_file() and p.suffix in ['.json','.csv','.png'])
    files.update(p for p in recovery.iterdir() if p.is_file() and ((p.suffix=='.csv' and not p.name.endswith('.traces.csv')) or (p.suffix=='.json' and p.name!='reference_before.json')))
    for directory in ['dqn_assisted_seed42','double_dqn_assisted_seed42','dqn_assisted_seed43',
                      'dqn_coverage_v2_seed42','double_coverage_v2_seed42','dqn_coverage_v2_seed43']:
        for name in ['metadata.json','metrics.csv','evaluations.csv','episodes.csv']:
            files.add(recovery/directory/name)
    for name in ['candidate.policy.pt','candidate_coverage_v2.policy.pt','dqn_assisted_seed42/best.policy.pt',
                 'loaded_primary_matched.traces.csv','loaded_dqn_original.traces.csv','loaded_coverage42_check.traces.csv',
                 'cartpole/decisions.json','cartpole/evaluation.csv']:
        files.add(recovery/name)
    for directory in ['demonstrations_v1','demonstrations_v2']:
        for name in ['metadata.json','episodes.csv']:files.add(recovery/directory/name)
    files.update(p for p in Path('output/pdf').rglob('*') if p.is_file() and p.suffix in ['.pdf','.tex','.zip','.json'])
    return sorted(files)


def main():
    files=selected_evidence()
    assert all(p.exists() for p in files)
    assert all(p.stat().st_size<100_000_000 for p in files)
    manifest=dict(version='publicacion_entrega_v1',purpose='Frozen inference, final evidence, report and selected development provenance',
        excludes=['private_data/','tmp/','virtualenvs','resume checkpoints','replays','demonstrations.pt','historical raw campaigns not selected'],
        bytes=sum(p.stat().st_size for p in files),files={str(p):sha256(p) for p in files})
    Path('entrega/publicacion_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Public evidence files',len(files),'uncompressed bytes',manifest['bytes'])

if __name__=='__main__':main()
