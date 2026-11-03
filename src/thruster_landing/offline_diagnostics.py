"""Diagnóstico offline: gradientes proporcionados, copias Adam y datos congelados."""
import copy
import hashlib
import numpy as np
import torch
from torch.nn import functional as F
from .agents.dqn import compute_targets


def shared_parameters(model):
    # Sequential: Linear, ReLU, Linear, ReLU, Linear. La última capa es específica por acción.
    return tuple(model[0].parameters()) + tuple(model[2].parameters())


def vector_gradient(loss, parameters, retain_graph=True):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph)
    return torch.cat([gradient.reshape(-1) for gradient in gradients])


def selected_values(model, batch):
    return model(batch['observations']).gather(1, batch['actions'].unsqueeze(1)).squeeze(1)


@torch.no_grad()
def targets_for(agent, batch):
    target = agent.target(batch['next_observations'])
    online = agent.online(batch['next_observations'])
    return compute_targets(batch['rewards'], batch['terminated'], online, target,
                           agent.config.gamma, agent.config.algorithm == 'double_dqn')


def decompose(agent, batch):
    q = selected_values(agent.online, batch)
    target = targets_for(agent, batch)
    assert q.shape == target.shape == batch['terminated'].shape == (len(q),)
    losses = F.smooth_l1_loss(q, target, reduction='none', beta=1.0)
    parameters = shared_parameters(agent.online)
    full = vector_gradient(losses.mean(), parameters)
    groups, gradients = [], {}
    for name, mask in (('terminal', batch['terminated']), ('nonterminal', ~batch['terminated'])):
        count = int(mask.sum())
        fraction = count / len(q)
        gradient = vector_gradient(losses[mask].mean(), parameters) if count else torch.zeros_like(full)
        contribution = fraction * gradient
        gradients[name] = gradient
        groups.append({'group': name, 'count': count, 'proportion': fraction,
            'q_mean': float(q[mask].detach().mean()) if count else None,
            'target_mean': float(target[mask].mean()) if count else None,
            'td_signed_mean': float((target[mask] - q[mask].detach()).mean()) if count else None,
            'loss_per_sample_mean': float(losses[mask].detach().mean()) if count else None,
            'huber_linear_count': int(((q.detach()-target).abs()[mask] >= 1).sum()),
            'huber_abs_td_ge_10_count': int(((q.detach()-target).abs()[mask] >= 10).sum()),
            'shared_gradient_mean_norm': float(gradient.norm()),
            'shared_weighted_contribution_norm': float(contribution.norm()),
            'projection_share_full_gradient': float(torch.dot(contribution,full)/full.square().sum()) if full.norm()>0 else None})
    weighted = sum(row['proportion'] * gradients[row['group']] for row in groups)
    error = float((weighted - full).norm())
    assert error <= 2e-5 * max(1., float(full.norm()))
    first, second = gradients['terminal'], gradients['nonterminal']
    cosine = float(F.cosine_similarity(first,second,dim=0)) if first.norm()>0 and second.norm()>0 else None
    summary = {'terminal_count': groups[0]['count'], 'nonterminal_count': groups[1]['count'],
               'cosine_shared_groups': cosine, 'shared_full_gradient_norm': float(full.norm()),
               'weighted_sum_verification_error': error}
    detail = {'q': q.detach(), 'target': target, 'td': target-q.detach(), 'loss': losses.detach()}
    return groups, summary, detail


def subset(arrays, indices):
    return {key:value[indices] for key,value in arrays.items()}


def state_action_key(arrays, index):
    return hashlib.sha256(arrays['observations'][index].numpy().tobytes()+arrays['actions'][index].numpy().tobytes()).hexdigest()


def split_terminal_groups(arrays, rng, max_groups=1024, fraction=.75):
    groups={}
    for index in torch.where(arrays['terminated'])[0].tolist():
        key=state_action_key(arrays,index)
        groups.setdefault(key,[]).append(index)
    keys=sorted(groups)
    rng.shuffle(keys)
    keys=keys[:max_groups]
    if len(keys)<8:
        raise ValueError('Insuficientes grupos terminales para ajuste y holdout')
    cut=int(len(keys)*fraction)
    training,holdout=keys[:cut],keys[cut:]
    assert not set(training)&set(holdout)
    train_indices=[groups[key][0] for key in training]
    held_indices=[groups[key][0] for key in holdout]
    excluded_indices={index for key in holdout for index in groups[key]}
    # Comprobar que un mismo (s,a) no recibe etiquetas incompatibles en este conjunto.
    for key in keys:
        rewards=arrays['rewards'][groups[key]]
        assert float(rewards.max()-rewards.min())<1e-4
    return train_indices,held_indices,excluded_indices,{'train_state_action_hashes':training,'holdout_state_action_hashes':holdout,
           'train_indices':train_indices,'holdout_indices':held_indices,'holdout_all_duplicate_indices':sorted(excluded_indices)}


@torch.no_grad()
def terminal_errors(model, batch):
    error=selected_values(model,batch)-batch['rewards']
    return {'mean_signed_error':float(error.mean()), 'mae':float(error.abs().mean()),
            'rmse':float(error.square().mean().sqrt()),'p05_error':float(error.quantile(.05)),
            'p50_error':float(error.quantile(.5)),'p95_error':float(error.quantile(.95))}


def copied_optimizer(model, saved_state, learning_rate, mode):
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    if mode=='saved_Adam_state':
        optimizer.load_state_dict(copy.deepcopy(saved_state))
    elif mode!='fresh_Adam_state':
        raise ValueError(mode)
    return optimizer


def one_update_copy(model, optimizer_state, learning_rate, mode, batch, target, mask, clip_limit):
    clone=copy.deepcopy(model)
    optimizer=copied_optimizer(clone,optimizer_state,learning_rate,mode)
    optimizer.zero_grad(set_to_none=True)
    # Mantener el denominador B: aislar la contribución real, sin reponderar el grupo.
    loss=F.smooth_l1_loss(selected_values(clone,batch)[mask],target[mask],beta=1.0,reduction='sum')/len(target)
    loss.backward()
    before=torch.nn.utils.clip_grad_norm_(clone.parameters(),clip_limit,error_if_nonfinite=True)
    after=torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in clone.parameters()]))
    optimizer.step()
    return clone,{'update_loss':float(loss.detach()),'gradient_norm_before_clip':float(before),'gradient_norm_after_clip':float(after)}
