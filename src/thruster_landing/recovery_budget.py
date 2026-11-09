"""Shared, conservative intervention limits: every actual env.step, including eval/demo."""
import json,time,os
from pathlib import Path

ROOT=Path(os.environ.get('THRUSTER_RECOVERY_ROOT','artifacts/recovery_v1'))

class BudgetExceeded(RuntimeError):pass

class Budget:
    def __init__(self,stage):
        self.path=ROOT/'budget.json';self.data=json.loads(self.path.read_text());self.stage=stage;self.start=self.data['actual_steps'];self.wall=time.time();self.closed=False
        self.guard()
    def guard(self):
        if self.data.get('status')=='complete':raise BudgetExceeded('Intervention already closed; initialize a new budget for another run')
        if time.time()>=self.data['deadline_epoch'] or self.data['actual_steps']>=self.data['step_limit']:
            raise BudgetExceeded('Intervention budget exhausted')
    def step(self,env,action):
        self.guard();self.data['actual_steps']+=1
        if self.data['actual_steps']%1000==0:self.flush()
        return env.step(action)
    def flush(self):
        self.data['last_progress_epoch']=time.time();self.path.write_text(json.dumps(self.data,indent=2)+'\n')
    def close(self):
        if not self.closed:
            self.data['stages'].append(dict(stage=self.stage,steps=self.data['actual_steps']-self.start,wall_seconds=time.time()-self.wall));self.flush();self.closed=True
