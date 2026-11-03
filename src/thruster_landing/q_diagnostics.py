"""Sondas offline fijas: TD=target-Q; error terminal=Q-recompensa."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .agents.dqn import DQNAgent, DQNConfig, compute_targets
from .checkpoints import atomic_save, read_checkpoint


def transition_id(arrays, index):
    parts=[]
    for key in sorted(arrays):
        parts.append(key.encode()+arrays[key][index].contiguous().numpy().tobytes())
    return hashlib.sha256(b''.join(parts)).hexdigest()


def create_probe(source_checkpoint, output, count_per_group=64, seed=20261102):
    output=Path(output)
    if output.exists():
        raise ValueError('No se sobrescribe la muestra fija')
    payload=read_checkpoint(source_checkpoint,'resume')
    arrays=payload['agent']['replay']['arrays']
    rng=np.random.default_rng(seed)
    indices=[]
    for terminal in (False,True):
        available=torch.where(arrays['terminated']==terminal)[0].numpy()
        if len(available)<count_per_group:
            raise ValueError('Replay insuficiente para ambos grupos')
        indices.extend(sorted(rng.choice(available,count_per_group,replace=False).tolist()))
    sample={key:value[indices].clone() for key,value in arrays.items()}
    identifiers=[transition_id(arrays,i) for i in indices]
    assert len(set(identifiers))==len(identifiers)
    metadata={'schema_version':1,'source_checkpoint':str(source_checkpoint),
              'source_sha256':hashlib.sha256(Path(source_checkpoint).read_bytes()).hexdigest(),
              'seed':seed,'count_per_group':count_per_group,'replay_indices':indices,
              'transition_ids':identifiers,'role':'offline diagnostic only; never trained on probe',
              'experiment_train_manifest_sha256':payload['experiment']['train_sha256']}
    atomic_save({'arrays':sample,'metadata':metadata},output)
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def load_model(path):
    payload=read_checkpoint(path)
    if payload['kind']=='resume':
        state=payload['agent']
        agent=DQNAgent(DQNConfig(**{**state['config'],'device':'cpu'}))
        agent.online.load_state_dict(state['online'])
        agent.target.load_state_dict(state['target'])
        step=state['env_steps']
    else:
        agent=DQNAgent(DQNConfig(**{**payload['config'],'device':'cpu'}))
        agent.online.load_state_dict(payload['online'])
        agent.target.load_state_dict(payload['online'])  # baseline inicial, sincronización exacta
        step=payload['env_steps']
        if step!=0:
            raise ValueError('Para pasos posteriores se requiere red objetivo del resume checkpoint')
    return agent,step


def diagnose_checkpoint(path, probe):
    agent,step=load_model(path)
    batch=probe['arrays']
    count=len(batch['rewards'])
    q=agent.online(batch['observations']).gather(1,batch['actions'].unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_target=agent.target(batch['next_observations'])
        next_online=agent.online(batch['next_observations'])
        target=compute_targets(batch['rewards'],batch['terminated'],next_online,next_target,
                               agent.config.gamma,agent.config.algorithm=='double_dqn')
        next_value=next_target.gather(1,next_online.argmax(1,keepdim=True)).squeeze(1) if agent.config.algorithm=='double_dqn' else next_target.max(1).values
        bootstrap=agent.config.gamma*torch.where(batch['terminated'],torch.zeros_like(next_value),next_value)
    assert q.shape==target.shape==batch['rewards'].shape==batch['terminated'].shape==(count,)
    assert torch.allclose(target,batch['rewards']+bootstrap)
    assert torch.equal(target[batch['terminated']],batch['rewards'][batch['terminated']])
    agent.optimizer.zero_grad(set_to_none=True)
    loss=F.smooth_l1_loss(q,target)
    loss.backward()
    before=torch.nn.utils.clip_grad_norm_(agent.online.parameters(),agent.config.gradient_norm_limit,error_if_nonfinite=True)
    after=torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in agent.online.parameters() if p.grad is not None]))
    assert all(p.grad is None for p in agent.target.parameters())
    # No optimizer.step(): solo medir sobre idéntico batch, sin actualizar pesos.
    values={'q':q.detach(),'reward':batch['rewards'],'target':target,'td_target_minus_q':target-q.detach(),
            'bootstrap':bootstrap,'terminal_error_q_minus_reward':q.detach()-batch['rewards']}
    rows=[]
    for index,identifier in enumerate(probe['metadata']['transition_ids']):
        terminal=bool(batch['terminated'][index])
        row={'env_steps':step,'transition_id':identifier,'group':'terminal' if terminal else 'nonterminal',
             'action':int(batch['actions'][index]),'terminated':terminal,'truncated':bool(batch['truncated'][index])}
        row.update({key:float(value[index]) for key,value in values.items()
                    if key!='terminal_error_q_minus_reward' or terminal})
        rows.append(row)
    return rows,{'env_steps':step,'probe_size':count,'loss_fixed_batch':float(loss.detach()),
                 'gradient_norm_before_clip':float(before),'gradient_norm_after_clip':float(after)}
