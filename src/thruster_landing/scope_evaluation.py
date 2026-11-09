"""Frozen, read-only common evaluation of original and PID-assisted methods."""
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .agents.pid import PIDConfig, PIDController
from .checkpoints import load_inference
from .envs import EnvConfig, ThrusterLandingEnv
from .envs.physics import State
from .recovery_agent import AssistedDQN
from .rl_evaluation import episode_metrics
from .training_telemetry import outcome_details


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_protocol(protocol):
    """Reject any changed artifact/configuration/evaluator before executing cases."""
    for path, expected in protocol['sha256'].items():
        if sha256(path) != expected:
            raise ValueError(f'Frozen protocol mismatch: {path}')


def decode_observation(obs, config):
    """PID receives exactly the physical information present in the float32 Box."""
    return State(x=float(obs[0])*config.position_scale,
                 y=float(obs[1])*config.height_scale+config.foot_drop,
                 vx=float(obs[2])*config.velocity_scale,
                 vy=float(obs[3])*config.velocity_scale,
                 theta=float(np.arctan2(obs[4],obs[5])),
                 omega=float(obs[6])*config.omega_scale,
                 gas=float(obs[7])*config.gas_capacity), float(obs[8])*config.wind_scale


class ObservedPID:
    def __init__(self, env_config, pid_config):
        self.config=env_config
        self.pid=PIDController(env_config,pid_config)

    def reset(self):
        self.pid.reset()

    def act(self, observation):
        state,wind=decode_observation(observation,self.config)
        return self.pid.act(state,wind)


def load_method(method, config, pid_config):
    if method['kind']=='pid':
        return ObservedPID(config,pid_config)
    if method['kind']=='assisted':
        agent,_=AssistedDQN.load(method['checkpoint'])
        assert agent.config['algorithm']==method['algorithm']
        assert agent.demos is None
    else:
        agent,experiment=load_inference(method['checkpoint'])
        assert agent.config.algorithm==method['algorithm']
        assert experiment['env_config']==json.loads(json.dumps(config.to_dict()))
    return agent


def nested_equal(first,second):
    if isinstance(first,torch.Tensor):
        return isinstance(second,torch.Tensor) and torch.equal(first,second)
    if isinstance(first,np.ndarray):
        return np.array_equal(first,second)
    if isinstance(first,dict):
        return first.keys()==second.keys() and all(nested_equal(v,second[k]) for k,v in first.items())
    if isinstance(first,(tuple,list)):
        return len(first)==len(second) and all(nested_equal(a,b) for a,b in zip(first,second))
    return first==second


def agent_snapshot(agent):
    return copy.deepcopy(dict(online=agent.online.state_dict(),target=agent.target.state_dict(),
        optimizer=agent.optimizer.state_dict(),rng=agent.rng.bit_generator.state,
        replay=agent.replay.state_dict(),env_steps=agent.env_steps,updates=agent.updates,
        torch_rng=torch.get_rng_state()))


def evaluate_method(agent, method, config, manifest, trace_stride=1):
    """Same cases, initial gas, decision period, observation and termination rules."""
    env=ThrusterLandingEnv(config);rows=[];traces=[]
    before=agent_snapshot(agent) if method['kind']!='pid' else None
    def forbidden_teacher(*args,**kwargs):
        raise AssertionError('Teacher query during neural inference')
    guard=patch.object(PIDController,'act',forbidden_teacher) if before is not None else None
    if guard is not None:guard.start()
    try:
        for scenario in manifest['scenarios']:
            obs,info=env.reset(seed=scenario['seed'],options=scenario['options'])
            assert info['state']['gas']==config.gas_capacity
            initial=info['state'];total=discounted=0.;parts={};steps=0
            if before is None:agent.reset()
            def trace(action):
                traces.append(dict(method=method['id'],scenario_id=scenario['id'],condition=scenario['condition'],
                    suite=scenario['suite'],time=info['time'],action=action,outcome=info['outcome'],wind_force_N=info['wind_force'],**info['state']))
            trace(-1)
            while True:
                action=agent.act(obs) if before is None else agent.act(obs,explore=False)
                if not env.action_space.contains(action):raise AssertionError('Invalid action')
                obs,reward,terminated,truncated,info=env.step(action)
                total+=reward;discounted+=.99**steps*reward;steps+=1
                for key,value in info['reward_components'].items():parts[key]=parts.get(key,0.)+value
                if steps%trace_stride==0 or terminated or truncated:trace(int(action))
                if terminated or truncated:
                    row=episode_metrics(info,initial,total,steps,scenario,method['id'],'test')
                    row.update(method=method['id'],suite=scenario['suite'],exposure=scenario['exposure'],
                        paired_wind_id=scenario.get('paired_wind_id',''),discounted_return=discounted,
                        terminated=terminated,truncated=truncated,**outcome_details(info,env),
                        **{'reward_'+key:value for key,value in parts.items()})
                    rows.append(row);break
    finally:
        env.close()
        if guard is not None:guard.stop()
    if before is not None and not nested_equal(before,agent_snapshot(agent)):
        raise AssertionError('Evaluation mutated network/optimizer/replay/RNG/counters')
    return rows,traces,dict(method=method['id'],episodes=len(rows),environment_steps=sum(r['steps'] for r in rows),
        neural_only=before is not None,teacher_queries=0 if before is not None else None,
        updates_during_evaluation=0,exploration=False,agent_state_unchanged=before is not None)


def group_statistics(rows):
    """Metrics use contacts only; fuel/duration separated by success/failure/cause."""
    from collections import defaultdict
    groups=defaultdict(list)
    for row in rows:
        for status in ['all','success' if row['success'] else 'failure','cause:'+row['termination_cause']]:
            groups[(row['method'],row['suite'],row['condition'],status)].append(row)
    result=[]
    for (method,suite,condition,status),group in sorted(groups.items()):
        contacts=[r for r in group if r['contact']]
        r=dict(method=method,suite=suite,condition=condition,status=status,n=len(group),
            successes=sum(x['success'] for x in group),contacts=len(contacts),
            success_rate=sum(x['success'] for x in group)/len(group),
            gas_exhausted=sum(x['gas_exhausted'] for x in group))
        for key in ['gas_consumed_Ns','duration_s','return','discounted_return','final_error_x_m']:
            values=np.array([x[key] for x in group]);r['mean_'+key]=float(values.mean());r['median_'+key]=float(np.median(values))
        for key in ['contact_error_x_m','contact_vx_m_s','contact_vy_m_s','contact_theta_rad','contact_omega_rad_s']:
            values=np.array([x[key] for x in contacts]);r['mean_'+key]=float(values.mean()) if len(values) else None
            r['mean_abs_'+key]=float(np.abs(values).mean()) if len(values) else None
            r['p90_abs_'+key]=float(np.quantile(np.abs(values),.9)) if len(values) else None
        result.append(r)
    return result
