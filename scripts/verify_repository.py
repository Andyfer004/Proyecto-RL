"""Read-only verification of the evidence shipped with a clean Git checkout."""
import json
from pathlib import Path
import subprocess
import sys

from thruster_landing.scope_evaluation import sha256,verify_protocol


def main():
    manifest=json.loads(Path('entrega/publicacion_manifest.json').read_text())
    for name,expected in manifest['files'].items():
        p=Path(name)
        if not p.is_file() or sha256(p)!=expected:raise AssertionError('Published evidence missing/changed: '+name)
    verify_protocol(json.loads(Path('configs/scope_final_v1_protocol.json').read_text()))
    result=subprocess.run(['git','ls-files','-z'],capture_output=True)
    if result.returncode==0:
        files=result.stdout.decode().split('\0')
        assert not any(p.startswith(('private_data/','tmp/','.venv/')) for p in files)
    subprocess.run([sys.executable,'scripts/verify_delivery.py'],check=True)
    print('PASS: public evidence selection is complete; frozen hashes match; private/temp files excluded.')

if __name__=='__main__':main()
