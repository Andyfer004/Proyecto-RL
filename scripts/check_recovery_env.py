"""Official Gymnasium checker, optionally included in the shared intervention budget."""
import argparse
from gymnasium.utils.env_checker import check_env
from thruster_landing.envs import ThrusterLandingEnv
from thruster_landing.recovery_budget import Budget


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--count-budget',action='store_true');a=p.parse_args();b=Budget('official_gymnasium_checker') if a.count_budget else None;original=ThrusterLandingEnv.step
    try:
        if b is not None:
            def measured(self,action):
                b.guard();b.data['actual_steps']+=1
                return original(self,action)
            ThrusterLandingEnv.step=measured
        env=ThrusterLandingEnv();check_env(env,skip_render_check=True);env.close();print('Gymnasium checker: valid')
    finally:
        ThrusterLandingEnv.step=original
        if b is not None:b.close()
if __name__=='__main__':main()
