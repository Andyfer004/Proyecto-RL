from dataclasses import replace
import json
import pytest
from thruster_landing.envs import EnvConfig
from thruster_landing.reward_audit import level1_only, rollout


def case():
    return dict(id='synthetic-level1',seed=42,condition='level1',source_level=1,
                options=dict(initial_state=dict(x=0.,y=2.,vx=0.,vy=-.2,theta=0.,omega=0.),wind_force=0.))


def test_audit_rejects_final_test_and_filters_levels():
    with pytest.raises(ValueError):level1_only(dict(split='test',scenarios=[case()]))
    assert level1_only(dict(split='validation',scenarios=[case(),{**case(),'source_level':2}]))['scenarios']==[case()]
    with pytest.raises(ValueError):rollout(EnvConfig(),case(),'test','off')


def test_partial_truncation_return_components_and_no_fake_contact():
    # Synthetic fixture only: the audit itself uses the frozen 25s environment.
    row,steps=rollout(replace(EnvConfig(),max_time=.075),case(),'validation','off')
    assert len(steps)==2 and row['truncated'] and not row['terminated']
    assert row['contact_state'] is None and row['contact_vy_m_s'] is None
    assert steps[0]['elapsed_s']==pytest.approx(.05)
    assert steps[1]['elapsed_s']==pytest.approx(.025)
    assert row['discounted_return']==pytest.approx(steps[0]['reward']+.99*steps[1]['reward'])
    assert row['reward_terminal']==0 and row['discounted_time']==pytest.approx(-.2*(.05+.99*.025))
    assert steps[0]['after_y']==steps[1]['before_y']


def test_real_contact_keeps_contact_state_and_single_terminal_penalty():
    row,steps=rollout(EnvConfig(),case(),'train','off')
    assert row['terminated'] and not row['success'] and row['contact_state'] is not None
    assert row['reward_terminal']==-100 and row['outcome']=='invalid_contact'
    assert sum(s['reward_terminal']!=0 for s in steps)==1
    assert row['discounted_terminal']==pytest.approx(-100*.99**(len(steps)-1))
    assert json.loads(row['contact_state'])==json.loads(row['final_state'])
