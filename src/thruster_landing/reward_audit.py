"""Paired behaviour audit on existing level-1 cases; never learns or uses PID labels."""
import hashlib
import json
import numpy as np
from .envs import ThrusterLandingEnv
from .envs.physics import State, geometry
from .envs.rewards import components
from .rl_evaluation import episode_metrics

BEHAVIOURS = ('pid', 'off', 'up', 'dqn', 'double_dqn')


def level1_only(manifest):
    if manifest.get('split') not in ('train', 'validation'):
        raise ValueError('Solo entrenamiento y validación; prueba final excluida')
    cases = [s for s in manifest['scenarios'] if s.get('source_level') == 1]
    if not cases:
        raise ValueError('Faltan casos de nivel 1')
    return {**manifest, 'scenarios': cases}


def inference_fingerprint(agent):
    hasher = hashlib.sha256()
    for network in (agent.online, agent.target):
        for key, value in network.state_dict().items():
            hasher.update(key.encode()); hasher.update(value.detach().cpu().numpy().tobytes())
    hasher.update(json.dumps(agent.rng.bit_generator.state, sort_keys=True).encode())
    hasher.update(str((agent.env_steps, agent.updates, len(agent.replay))).encode())
    return hasher.hexdigest()


def signal_tags(before, after, outcome, config):
    """Non-exclusive kinematic observations, not assertions about action causality."""
    h0 = float(geometry(before, config)[:, 1].min())
    h1 = float(geometry(after, config)[:, 1].min())
    return {
        'tilt_correction': outcome == 'flying' and abs(after.theta) < abs(before.theta) - 1e-9,
        'precontact_braking': outcome == 'flying' and h0 < .5 and before.vy < 0 and after.vy > before.vy + 1e-8,
        'horizontal_approach': outcome == 'flying' and abs(after.x) < abs(before.x) - 1e-9,
        'descent': outcome == 'flying' and h1 < h0 - 1e-9,
        'ascent': outcome == 'flying' and h1 > h0 + 1e-9,
        'landing': outcome == 'success',
    }


def rollout(config, scenario, split, behaviour, pid=None, policy=None, gamma=.99):
    if behaviour not in BEHAVIOURS or scenario.get('source_level') != 1 or split not in ('train', 'validation'):
        raise ValueError('Conducta o caso fuera del protocolo de nivel 1')
    env = ThrusterLandingEnv(config)
    if behaviour == 'pid':
        pid.reset()
    steps, totals, discounted = [], {}, {}
    try:
        observation, info = env.reset(seed=scenario['seed'], options=scenario['options'])
        initial = info['state']; max_height = env.state.y; k = 0
        while True:
            before = env.state
            action = (pid.act(before, info['wind_force']) if behaviour == 'pid' else
                      0 if behaviour == 'off' else 7 if behaviour == 'up' else policy.act(observation, explore=False))
            observation, reward, terminated, truncated, info = env.step(action)
            after = env.state; parts = info['reward_components']; discount = gamma ** k
            # Independently recompute all components using actual event-localized elapsed time.
            dt = info['time'] - (steps[-1]['time_after_s'] if steps else 0.)
            expected = components(before, after, dt, info['outcome'], config)
            assert all(abs(parts[key] - expected[key]) < 1e-10 for key in expected)
            assert abs(sum(parts.values()) - reward) < 1e-10
            row = dict(reward_version='v1', split=split, scenario_id=scenario['id'], seed=scenario['seed'],
                       behaviour=behaviour, step=k, time_after_s=info['time'], elapsed_s=dt,
                       action=action, outcome=info['outcome'], terminated=terminated, truncated=truncated,
                       reward=reward, discount=discount, discounted_reward=discount * reward,
                       height_before_m=before.y, height_after_m=after.y,
                       clearance_before_m=float(geometry(before, config)[:, 1].min()),
                       clearance_after_m=float(geometry(after, config)[:, 1].min()))
            for key, value in before.to_dict().items(): row['before_' + key] = value
            for key, value in after.to_dict().items(): row['after_' + key] = value
            row.update(signal_tags(before, after, info['outcome'], config))
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.) + value
                discounted[key] = discounted.get(key, 0.) + discount * value
                row['reward_' + key] = value
                row['discounted_' + key] = discount * value
            steps.append(row); k += 1; max_height = max(max_height, after.y)
            if terminated or truncated: break
        result = episode_metrics(info, initial, sum(totals.values()), k, scenario, behaviour, split)
        result.update(reward_version='v1', behaviour=behaviour, discounted_return=sum(discounted.values()),
                      max_center_height_m=max_height, terminated=terminated, truncated=truncated,
                      termination_cause=('ceiling_exit' if info['outcome'] == 'out_of_bounds' and
                                         geometry(after, config)[:, 1].max() >= config.ceiling - 1e-6 else
                                         'side_exit' if info['outcome'] == 'out_of_bounds' else info['outcome']),
                      contact_state=json.dumps(info['state'], sort_keys=True) if info['contact'] else None,
                      source_level=1)
        result.update({'reward_' + key: value for key, value in totals.items()})
        result.update({'discounted_' + key: value for key, value in discounted.items()})
        assert abs(sum(row['discounted_reward'] for row in steps) - result['discounted_return']) < 1e-8
        assert abs(sum(row['reward'] for row in steps) - result['return']) < 1e-8
        return result, steps
    finally:
        env.close()
