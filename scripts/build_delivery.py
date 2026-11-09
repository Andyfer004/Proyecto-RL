"""Package a self-contained selected delivery; never train or rerun final evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from thruster_landing.scope_evaluation import verify_protocol,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('artifacts/delivery_v1/proyecto_rl_entrega_v1.zip'));a=p.parse_args()
    protocol=json.loads(Path('configs/scope_final_v1_protocol.json').read_text());verify_protocol(protocol)
    files={Path('README.md'),Path('pyproject.toml'),Path('.gitignore')}
    for name in ['src','scripts','tests','configs','docs','entrega','artifacts/scope_alignment_v1','artifacts/recovery_v1','output/pdf']:
        files.update(p for p in Path(name).rglob('*') if p.is_file() and p.suffix!='.log' and not any(x in p.parts for x in ['__pycache__','.pytest_cache','matplotlib_cache']))
    files.update(Path(p) for p in protocol['sha256'])
    for name in ['closure_reference.json','versions.json','verification.txt']:
        path=Path('artifacts/delivery_v1')/name
        if path.exists():files.add(path)
    # Explicit allowlist roots only; no .git, environments, credentials or assistant logs.
    hashes={str(path):sha256(path) for path in sorted(files)}
    manifest=dict(version='entrega_v1',scope='Academic2D; frozen final evidence; assisted methods explicitly labelled',
        exclusions='No virtualenv/git/credentials. Most historical experiment raw artifacts excluded; selected final/recovery evidence included.',
        files=hashes)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(files):archive.write(path,'Proyecto-RL/'+str(path))
        archive.writestr('Proyecto-RL/MANIFEST_ENTREGA.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(a.output) as archive:
        assert archive.testzip() is None
        for path,h in hashes.items():assert hashlib.sha256(archive.read('Proyecto-RL/'+path)).hexdigest()==h,path
    a.output.with_suffix('.sha256').write_text(sha256(a.output)+'  '+a.output.name+'\n')
    print(a.output,'files',len(files),'bytes',a.output.stat().st_size,'archive hashes verified')

if __name__=='__main__':main()
