"""Read-only decision telemetry; no RNG draws and no effect on actions or replay."""
import csv
import numpy as np
from .envs.physics import geometry


def outcome_details(info, env):
    state=info['state'];outcome=info['outcome']
    cause=('ceiling_exit' if outcome=='out_of_bounds' and geometry(env.state,env.config)[:,1].max()>=env.config.ceiling-1e-6
           else 'side_exit' if outcome=='out_of_bounds' else outcome)
    return dict(termination_cause=cause,gas_exhausted=state['gas']<=1e-8,contact=info['contact'])


class TrainingTelemetry:
    def __init__(self, runner, output):
        self.data=runner.setdefault('decision_telemetry',dict(histograms={m:[0]*8 for m in ['exploration','exploitation']},first_success_step=None))
        self.stream=(output/'decisions.csv').open('w',newline='')
        fields=['env_steps','episode','scenario_id','mode','action','epsilon_used','reward','outcome','termination_cause','gas_exhausted','contact','success','duration_s','gas_remaining_Ns','x','y','vx','vy','theta','omega']
        self.writer=csv.DictWriter(self.stream,fieldnames=fields);self.writer.writeheader()

    def begin_episode(self, info):
        self.data['episode']=dict(exploration_steps=0,exploitation_steps=0,max_center_height_m=info['state']['y'],gas_exhausted=False,first_gas_empty_s=None)

    def record(self, agent, runner, action, mode, epsilon_used, reward, info, env):
        assert mode in ('exploration','exploitation')
        self.data['histograms'][mode][action]+=1
        ep=self.data['episode'];ep[mode+'_steps']+=1
        ep['max_center_height_m']=max(ep['max_center_height_m'],info['state']['y'])
        details=outcome_details(info,env)
        if details['gas_exhausted']:
            ep['gas_exhausted']=True
            if ep['first_gas_empty_s'] is None:ep['first_gas_empty_s']=info['time']
        if info['is_success'] and self.data['first_success_step'] is None:self.data['first_success_step']=agent.env_steps
        self.writer.writerow(dict(env_steps=agent.env_steps,episode=len(runner['episodes'])+1,scenario_id=runner['scenario']['id'],mode=mode,action=action,epsilon_used=epsilon_used,reward=reward,outcome=info['outcome'],**details,success=info['is_success'],duration_s=info['time'],gas_remaining_Ns=info['state']['gas'],**{k:info['state'][k] for k in ['x','y','vx','vy','theta','omega']}))

    def metrics(self, agent):
        row={f'{mode}_action_{i}_count':n for mode,counts in self.data['histograms'].items() for i,n in enumerate(counts)}
        row['first_training_success_step']=self.data['first_success_step']
        arrays=agent.replay.arrays
        row['replay_successful_terminal_transitions']=int(np.sum(arrays['terminated'][:len(agent.replay)] & (arrays['rewards'][:len(agent.replay)]>0)))
        return row

    def close(self):
        self.stream.close()
