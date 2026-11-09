"""Project DQN and an independent adaptation of the official PyTorch DQN tutorial."""
from collections import deque
import copy,json,math,random,time
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from thruster_landing.agents.dqn import DQNAgent,DQNConfig
from thruster_landing.evaluation import write_csv
from thruster_landing.recovery_budget import Budget,ROOT

class TutorialReference:
    """Independent replay/targets/optimization, guided by PyTorch tutorial (not imported agent)."""
    def __init__(self):
        torch.manual_seed(42);self.net=nn.Sequential(nn.Linear(4,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,2));self.target=copy.deepcopy(self.net)
        self.opt=torch.optim.AdamW(self.net.parameters(),lr=3e-4,amsgrad=True);self.memory=deque(maxlen=10000);self.rng=random.Random(42);self.steps=0
    def act(self,obs,explore=True):
        epsilon=.01+(.9-.01)*math.exp(-self.steps/2500)
        if explore and self.rng.random()<epsilon:return self.rng.randrange(2)
        with torch.no_grad():return int(self.net(torch.tensor(obs,dtype=torch.float32)).argmax())
    def learn(self):
        if len(self.memory)<128:return
        items=self.rng.sample(self.memory,128)
        s=torch.tensor(np.stack([t[0] for t in items]));a=torch.tensor([t[1] for t in items]);r=torch.tensor([t[2] for t in items]);ns=torch.tensor(np.stack([t[3] for t in items]));done=torch.tensor([t[4] for t in items])
        current=self.net(s).gather(1,a[:,None]).view(-1)
        with torch.no_grad():target=r+.99*torch.where(done,0.,self.target(ns).max(1).values)
        loss=nn.SmoothL1Loss()(current,target);self.opt.zero_grad();loss.backward();nn.utils.clip_grad_value_(self.net.parameters(),100);self.opt.step()
        with torch.no_grad():
            for p,t in zip(self.net.parameters(),self.target.parameters()):t.mul_(.995).add_(p,alpha=.005)


def main():
    out=ROOT/'cartpole';out.mkdir(exist_ok=False);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    results=[];episodes=[]
    for method in ['project','tutorial_reference']:
        b=Budget('CartPole_'+method);start=time.perf_counter()
        config=DQNConfig(seed=42,learning_rate=3e-4,gamma=.99,buffer_capacity=10000,batch_size=128,learning_starts=1000,train_frequency=1,target_update_frequency=500,epsilon_start=.9,epsilon_end=.01,epsilon_decay_steps=10000,total_steps=50000,smooth_l1_beta=1.)
        agent=DQNAgent(config,observation_dim=4,action_count=2) if method=='project' else TutorialReference()
        env=gym.make('CartPole-v1');eval_env=gym.make('CartPole-v1');obs,_=env.reset(seed=42);episode_return=0;episode_num=0
        try:
            for step in range(0,50001):
                if step%10000==0:
                    returns=[]
                    for seed in range(800000,800016):
                        state,_=eval_env.reset(seed=seed);ret=0
                        while True:
                            state,r,t,tr,_=b.step(eval_env,agent.act(state,explore=False));ret+=r
                            if t or tr:break
                        returns.append(ret)
                    row=dict(method=method,train_steps=step,mean_return=float(np.mean(returns)),min_return=min(returns),max_return=max(returns),returns=returns);results.append(row);print(row,flush=True)
                    if step==50000:break
                action=agent.act(obs,explore=True);next_obs,r,t,tr,_=b.step(env,action);episode_return+=r
                if method=='project':
                    agent.replay.add(obs,action,r,next_obs,t,tr);agent.env_steps+=1
                    if agent.env_steps>=1000:agent.learn()
                    if agent.env_steps>=1000 and agent.env_steps%500==0:agent.sync_target()
                else:agent.memory.append((obs.copy(),action,float(r),next_obs.copy(),bool(t)));agent.steps+=1;agent.learn()
                obs=next_obs
                if t or tr:
                    episodes.append(dict(method=method,step=step+1,return_=episode_return));episode_num+=1;obs,_=env.reset(seed=42+episode_num);episode_return=0
            torch.save(dict(method=method,online=(agent.online if method=='project' else agent.net).state_dict()),out/(method+'.pt'))
            print(method,'wall_seconds',time.perf_counter()-start,flush=True)
        finally:env.close();eval_env.close();b.close()
    write_csv(out/'evaluation.csv',[{**r,'returns':json.dumps(r['returns'])} for r in results]);write_csv(out/'episodes.csv',episodes)
    (out/'decisions.json').write_text(json.dumps(dict(project_config=config.to_dict(),reference='Independent adaptation of https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html ; AdamW amsgrad3e-4, soft tau.005, exponential epsilon2500; no shared learner/replay/target functions.',train_budget_per_method=50000),indent=2)+'\n')
if __name__=='__main__':main()
