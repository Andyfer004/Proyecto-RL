import copy
import json
from pathlib import Path
import numpy as np
import pytest

from thruster_landing.envs import EnvConfig,ThrusterLandingEnv
from thruster_landing.scope_evaluation import (decode_observation,evaluate_method,group_statistics,verify_protocol,sha256)


def test_pid_observation_decoder_uses_only_available_values():
    c=EnvConfig();obs=np.array([.1,.2,.3,-.4,np.sin(.1),np.cos(.1),.2,.5,-.3],np.float32)
    s,w=decode_observation(obs,c)
    assert s.x==pytest.approx(1.2) and s.y==pytest.approx(6.)
    assert s.vx==pytest.approx(3) and s.vy==pytest.approx(-4)
    assert s.theta==pytest.approx(.1) and s.omega==pytest.approx(.6)
    assert s.gas==9000 and w==pytest.approx(-30)


def test_aggregation_never_turns_no_contact_into_landing_metric_or_low_fuel_success():
    def row(success,contact,fuel):
        return dict(method='m',suite='s',condition='c',success=success,contact=contact,
            termination_cause='success' if success else 'side_exit',gas_exhausted=False,
            gas_consumed_Ns=fuel,duration_s=1.,**{'return':1.},discounted_return=1.,final_error_x_m=0.,
            **{key:(.2 if contact else None) for key in ['contact_error_x_m','contact_vx_m_s','contact_vy_m_s','contact_theta_rad','contact_omega_rad_s']})
    result=group_statistics([row(True,True,2000),row(False,False,50)])
    success=next(r for r in result if r['status']=='success');failure=next(r for r in result if r['status']=='failure')
    assert success['mean_gas_consumed_Ns']==2000 and failure['mean_gas_consumed_Ns']==50
    assert failure['contacts']==0 and failure['mean_contact_vy_m_s'] is None


def test_frozen_protocol_rejects_changed_files(tmp_path):
    p=tmp_path/'config.json';p.write_text('{}');protocol=dict(sha256={str(p):sha256(p)})
    verify_protocol(protocol);p.write_text('{"change":1}')
    with pytest.raises(ValueError,match='Frozen'):verify_protocol(protocol)


def test_neural_common_evaluation_uses_greedy_and_masks_teacher(monkeypatch):
    from thruster_landing.agents.dqn import DQNAgent,DQNConfig
    from thruster_landing.agents.pid import PIDController
    agent=DQNAgent(DQNConfig());observations=[]
    def act(obs,explore):
        assert explore is False
        observations.append(obs.copy())
        with pytest.raises(AssertionError,match='Teacher'):
            PIDController(EnvConfig()).act(None)
        return 0
    monkeypatch.setattr(agent,'act',act)
    case=dict(id='synthetic',seed=777,condition='unit',suite='synthetic',exposure='unit',options=dict(initial_state=dict(x=0.,y=1.01,vx=0.,vy=-.1,theta=0.,omega=0.),wind_force=0.))
    rows,traces,audit=evaluate_method(agent,dict(id='unit_dqn',kind='scratch'),EnvConfig(),dict(scenarios=[case]))
    assert len(rows)==1 and audit['agent_state_unchanged'] and audit['updates_during_evaluation']==0
    assert len(observations)>0 and observations[0].shape==(9,)
    assert rows[0]['initial_state'] and rows[0]['contact']
