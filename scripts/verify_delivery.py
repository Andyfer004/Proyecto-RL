"""Read-only verification of the delivered report and frozen experimental evidence.
No environment steps, policy inference, optimizer updates or reevaluation.
"""
import hashlib
import json
from pathlib import Path
import re
from html.parser import HTMLParser

import numpy as np
import pandas as pd
import torch
from thruster_landing.scope_evaluation import verify_protocol,sha256


class Links(HTMLParser):
    def __init__(self):
        super().__init__();self.paths=[]
    def handle_starttag(self,tag,attrs):
        for name,value in attrs:
            if name in ['src','href'] and value:self.paths.append(value)


def main():
    protocol=json.loads(Path('configs/scope_final_v1_protocol.json').read_text())
    verify_protocol(protocol)
    root=Path('artifacts/scope_alignment_v1');record=json.loads((root/'protocol_sha256.json').read_text())
    assert record['configs/scope_final_v1_protocol.json']==sha256('configs/scope_final_v1_protocol.json')
    data=pd.read_csv(root/'final/episodes.csv');assert len(data)==2328
    expected={'pid':(98,287),'dqn_assisted42':(96,275),'double_assisted42':(94,266),
              'dqn_assisted43':(100,277),'dqn_scratch':(0,0),'double_dqn_scratch':(0,0)}
    for method,values in expected.items():
        for suite,successes,n in [('reserved',values[0],100),('complement',values[1],288)]:
            g=data[(data.method==method)&(data.suite==suite)];assert len(g)==n and g.success.sum()==successes
    for _,g in data.groupby('scenario_id'):
        assert len(g)==6 and g.initial_state.nunique()==1 and g.wind_force_N.nunique()==1
    contact_columns=[c for c in data if c.startswith('contact_')]
    assert data.loc[~data.contact,contact_columns].isna().all().all()
    pairs=pd.read_csv(root/'final/paired_vs_pid.csv')
    expected_pairs={'dqn_assisted42':(368,-65.556631),'double_assisted42':(357,-63.270023)}
    for method,(n,mean) in expected_pairs.items():
        g=pairs[(pairs.method==method)&pairs.both_success];assert len(g)==n
        assert np.isclose(g.success_only_gas_difference_Ns.mean(),mean,atol=1e-5)
    methodology=json.loads((root/'methodology_audit.json').read_text())
    for m in methodology:
        assert m['classification_updates']==10000 and m['offline_td_margin_updates']==10000
        assert m['online_td_margin_updates'] in [4969,7469]
        method=next(x for x in protocol['methods'] if x['id']==m['method'])
        payload=torch.load(method['checkpoint'],map_location='cpu',weights_only=True)
        assert payload['updates']==m['total_updates'] and payload['env_steps']==m['checkpoint_online_steps']
    cfg=json.loads(Path('configs/env.json').read_text());m=cfg['mass'];inertia=m*(cfg['width']**2+cfg['height']**2)/12
    assert np.isclose(inertia,37.6666666667) and np.isclose(360*.05/inertia,.4778761062)
    assert np.isclose(3000/m-cfg['gravity'],20.19)
    # Independent numeric targets and SmoothL1 examples, no neural update.
    assert np.isclose(1+.9*max([5,1]),5.5) and np.isclose(1+.9*np.array([5,1])[np.argmax([2,4])],1.9)
    assert np.isclose(.5*2**2/10,.2) and np.isclose(20-10/2,15.)
    report=Path('docs/informe_final.md').read_text()
    assert 'No se aisló el beneficio de las actualizaciones TD frente a la imitación por sí sola' in report
    assert '<!--' not in report and report.count('$$')%2==0
    for p in map(Path,['README.md','docs/informe_final.md','docs/demostracion.md','docs/metodologia.md']):
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if not target.startswith(('http:','https:','#')):assert (p.parent/target.split('#')[0]).exists(),(p,target)
    parser=Links();parser.feed(Path('docs/demostracion.html').read_text())
    for target in parser.paths:
        if not target.startswith(('http:','https:','#')):assert (Path('docs')/target).exists(),target
    closure=Path('artifacts/delivery_v1/closure_reference.json')
    if closure.exists():
        for path,h in json.loads(closure.read_text())['sha256'].items():assert sha256(path)==h,path
    print('PASS: frozen hashes,2328 episode rows,method counters,paired fuel,math examples and delivery links. No simulation/training.')

if __name__=='__main__':main()
