"""Diagnósticos beta explícitos, sin pasos de optimizador ni cambios de modelos."""
import torch
from .offline_diagnostics import shared_parameters,selected_values,targets_for,vector_gradient


def measure_groups(agent,batch,beta):
    q=selected_values(agent.online,batch)
    target=targets_for(agent,batch)
    losses=torch.nn.SmoothL1Loss(beta=beta,reduction='none')(q,target)
    derivative=torch.autograd.grad(losses.sum(),q,retain_graph=True)[0]
    shared=shared_parameters(agent.online)
    full=vector_gradient(losses.mean(),shared)
    gradients={};rows=[]
    for group,mask in (('terminal',batch['terminated']),('nonterminal',~batch['terminated'])):
        n=int(mask.sum());fraction=n/len(q)
        gradient=vector_gradient(losses[mask].mean(),shared) if n else torch.zeros_like(full)
        gradients[group]=gradient
        rows.append({'group':group,'n':n,'proportion':fraction,
            'q_mean':float(q[mask].detach().mean()) if n else None,
            'target_mean':float(target[mask].mean()) if n else None,
            'td_signed_mean':float((target[mask]-q[mask].detach()).mean()) if n else None,
            'loss_mean':float(losses[mask].detach().mean()) if n else None,
            'mean_abs_loss_derivative':float(derivative[mask].abs().mean()) if n else None,
            'mean_signed_loss_derivative':float(derivative[mask].mean()) if n else None,
            'linear_region_count':int(((q.detach()-target).abs()[mask]>=beta).sum()),
            'abs_td_lt1_count':int(((q.detach()-target).abs()[mask]<1).sum()),
            'abs_td_between1_and10_count':int((((q.detach()-target).abs()[mask]>=1)&((q.detach()-target).abs()[mask]<10)).sum()),
            'abs_td_ge10_count':int(((q.detach()-target).abs()[mask]>=10).sum()),
            'shared_group_gradient_norm':float(gradient.norm()),
            'shared_weighted_gradient_norm':float((fraction*gradient).norm()),
            'projection_share_full':float(torch.dot(fraction*gradient,full)/full.square().sum()) if full.norm()>0 else None})
    combined=sum(row['proportion']*gradients[row['group']] for row in rows)
    residual=float((combined-full).norm())
    assert residual<2e-5*max(1.,float(full.norm()))
    a,b=gradients['terminal'],gradients['nonterminal']
    cosine=float(torch.nn.functional.cosine_similarity(a,b,dim=0)) if a.norm()>0 and b.norm()>0 else None
    return rows,{'cosine_shared':cosine,'shared_full_gradient_norm':float(full.norm()),'identity_error':residual}, {
        'q':q.detach(),'reward':batch['rewards'],'target':target,'td':target-q.detach(),
        'loss':losses.detach(),'derivative':derivative.detach()}
