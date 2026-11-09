"""Versioned assisted DQN: landing scales, train-only demos, margin and n-step TD.
This is not vanilla DQN, not exact DQfD (no PER), and not learning from scratch.
"""
import copy
import numpy as np
import torch
from torch import nn
from .agents.dqn import QNetwork,compute_targets
from .agents.replay import ReplayBuffer


def landing_features(obs,config):
    return np.asarray(obs,dtype=np.float32)*np.asarray(config['observation_multipliers'],dtype=np.float32)


def large_margin(q,actions,margin):
    offset=torch.full_like(q,margin);offset.scatter_(1,actions[:,None],0.)
    return ((q+offset).max(1).values-q.gather(1,actions[:,None]).squeeze(1)).mean()


class AssistedDQN:
    def __init__(self,config,demos=None):
        self.config=copy.deepcopy(config);torch.set_num_threads(1);torch.manual_seed(config['seed']);self.rng=np.random.default_rng(config['seed'])
        self.online=QNetwork(9,8,tuple(config['architecture']));self.target=copy.deepcopy(self.online).eval();self.target.requires_grad_(False)
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=config['learning_rate']);self.replay=ReplayBuffer(config['online_capacity'],9,config['seed']+1)
        self.updates=0;self.env_steps=0;self.demos=demos
        self.train_indices=torch.where(demos['train_mask'])[0] if demos is not None else None
    def act(self,obs,explore=False):
        if explore and self.rng.random()<self.config['epsilon']:return int(self.rng.integers(8))
        with torch.inference_mode():return int(self.online(torch.from_numpy(landing_features(obs,self.config))).argmax())
    def sync(self):self.target.load_state_dict(self.online.state_dict())
    def sample_demo(self,n):
        index=self.train_indices[torch.randint(len(self.train_indices),(n,))]
        return {k:v[index] for k,v in self.demos.items() if isinstance(v,torch.Tensor) and k not in ['train_mask','episode_ids']}
    def classification_update(self):
        batch=self.sample_demo(256);q=self.online(batch['observations']);loss=nn.functional.cross_entropy(q,batch['actions']);self._step(loss)
        return float(loss.detach())
    def _step(self,loss):
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite assisted loss')
        self.optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(self.online.parameters(),self.config['gradient_norm_limit'],error_if_nonfinite=True);self.optimizer.step();self.updates+=1
    def learn(self,offline=False):
        c=self.config;n=c['batch_size'] if offline else int(c['batch_size']*c['demo_fraction']);demo=self.sample_demo(n)
        if offline:batch={k:demo[k] for k in ['observations','actions','rewards','next_observations','terminated']}
        else:
            online=self.replay.sample(c['batch_size']-n);batch={k:torch.cat([demo[k],online[k]],dim=0) for k in ['observations','actions','rewards','next_observations','terminated']}
        q=self.online(batch['observations']).gather(1,batch['actions'][:,None]).squeeze(1)
        with torch.no_grad():
            nt=self.target(batch['next_observations']);no=self.online(batch['next_observations']) if c['algorithm']=='double_dqn' else nt
            targets=compute_targets(batch['rewards'],batch['terminated'],no,nt,c['gamma'],c['algorithm']=='double_dqn')
            nt_n=self.target(demo['n_next_observations']);no_n=self.online(demo['n_next_observations']) if c['algorithm']=='double_dqn' else nt_n
            next_n=nt_n.gather(1,no_n.argmax(1,keepdim=True)).squeeze(1) if c['algorithm']=='double_dqn' else nt_n.max(1).values
            n_targets=demo['n_rewards']+demo['n_gamma']*torch.where(demo['n_terminated'],0.,next_n)
        assert q.shape==targets.shape==(len(batch['actions']),)
        one=nn.functional.smooth_l1_loss(q,targets,beta=c['smooth_l1_beta'])
        dq=self.online(demo['observations']);chosen=dq.gather(1,demo['actions'][:,None]).squeeze(1)
        multi=nn.functional.smooth_l1_loss(chosen,n_targets,beta=c['smooth_l1_beta'])
        supervised=large_margin(dq,demo['actions'],c['margin']);loss=one+c['n_step_weight']*multi+c['margin_weight']*supervised
        self._step(loss)
        return dict(loss=float(loss.detach()),td_loss=float(one.detach()),nstep_loss=float(multi.detach()),margin_loss=float(supervised.detach()),mean_abs_td=float((targets-q).detach().abs().mean()),q_mean=float(q.detach().mean()))
    def add(self,obs,a,r,next_obs,terminated,truncated):
        self.replay.add(landing_features(obs,self.config),a,r,landing_features(next_obs,self.config),terminated,truncated,source_level=1)
    def save(self,path,metadata=None):
        torch.save(dict(format='assisted_dqn_v1',config=self.config,online=self.online.state_dict(),target=self.target.state_dict(),optimizer=self.optimizer.state_dict(),updates=self.updates,env_steps=self.env_steps,rng=self.rng.bit_generator.state,metadata=metadata or {}),path)
    @classmethod
    def load(cls,path):
        p=torch.load(path,weights_only=True,map_location='cpu')
        if p['format']!='assisted_dqn_v1':raise ValueError('Wrong candidate format')
        agent=cls(p['config']);agent.online.load_state_dict(p['online']);agent.target.load_state_dict(p['target']);agent.optimizer.load_state_dict(p['optimizer']);agent.updates=p['updates'];agent.env_steps=p['env_steps'];agent.rng.bit_generator.state=p['rng'];return agent,p['metadata']
