"""Count verification env interactions too; no changes to environment files or RNG."""
from thruster_landing.recovery_budget import Budget
from thruster_landing.envs import ThrusterLandingEnv
_budget=None
_original=None

def pytest_sessionstart(session):
    global _budget,_original
    _budget=Budget('verification_pytest_environment_steps');_original=ThrusterLandingEnv.step
    def measured_step(self,action):
        _budget.guard();_budget.data['actual_steps']+=1
        if _budget.data['actual_steps']%1000==0:_budget.flush()
        return _original(self,action)
    ThrusterLandingEnv.step=measured_step

def pytest_sessionfinish(session,exitstatus):
    ThrusterLandingEnv.step=_original;_budget.close()
