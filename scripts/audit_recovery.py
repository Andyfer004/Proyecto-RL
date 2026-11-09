"""Independent dynamics/contact checks plus matched PID and failed RL reproduction."""
import json,time
from dataclasses import replace
from pathlib import Path
import numpy as np
from thruster_landing.envs import EnvConfig,ThrusterLandingEnv
from thruster_landing.envs.physics import State,integrate
from thruster_landing.agents.pid import PIDConfig,PIDController
from thruster_landing.checkpoints import load_inference
from thruster_landing.evaluation import write_csv
from thruster_landing.recovery_budget import Budget,ROOT


def main():
    out=ROOT/'audit';out.mkdir(exist_ok=False);b=Budget('system_audit_reproduction');c=EnvConfig();checks=[]
    try:
        # Independent ODE, explicit action forces/torques. No call to wrench/rotation.
        def rhs(z,F,M):
            x,y,vx,vy,theta,omega,gas=z
            return np.array([vx,vy,(-F*np.sin(theta)+30)/100,F*np.cos(theta)/100-9.81,omega,M/(100*(1.4**2+1.6**2)/12),-F])
        def rk4(z,F,M,dt):
            k1=rhs(z,F,M);k2=rhs(z+dt*k1/2,F,M);k3=rhs(z+dt*k2/2,F,M);k4=rhs(z+dt*k3,F,M)
            return z+dt*(k1+2*k2+2*k3+k4)/6
        for action,F,M in [(0,0,0),(1,1800,0),(2,600,-360),(3,600,360),(4,1200,0),(5,2400,-360),(6,2400,360),(7,3000,0)]:
            initial=State(.2,6.,.1,-1.,.2,.05);z=np.array(list(initial.to_dict().values()),float)
            for _ in range(500):z=rk4(z,F,M,.0001)
            errors=[]
            for n in [10,50]:
                s=initial
                for _ in range(n):s=integrate(s,action,.05/n,30.,c)
                delta=np.array(list(s.to_dict().values()))-z;errors.append(float(np.max(np.abs(delta[:6]))))
            assert errors[0]<.01 and errors[1]<=errors[0]+1e-10
            checks.append(dict(action=action,force_N=F,torque_Nm=M,error_default_max_SI=errors[0],error_refined_max_SI=errors[1]))
        # Exact independent free-fall touchdown time and velocity.
        env=ThrusterLandingEnv(c);obs,info=env.reset(seed=42,options=dict(initial_state=dict(x=0.,y=2.,vx=0.,vy=-.2,theta=0.,omega=0.),wind_force=0.))
        while True:
            obs,r,t,tr,info=b.step(env,0)
            if t or tr:break
        expected_t=(-.2+np.sqrt(.2**2+2*9.81))/9.81;expected_v=-np.sqrt(.2**2+2*9.81)
        assert abs(info['time']-expected_t)<1e-9 and abs(info['state']['vy']-expected_v)<1e-9 and abs(info['state']['y']-1)<1e-9
        env.close();check_contact=dict(expected_time_s=expected_t,measured_time_s=info['time'],expected_vy=expected_v,measured_vy=info['state']['vy'])
        # Existing L1 validation, never final test. Reproduce both from identical state.
        scenario=json.loads(Path('configs/level1_validation_scenarios.json').read_text())['scenarios'][0]
        policy,_=load_inference('artifacts/replay150k_weighted_seed42_100k_v1/dqn_continued100k/last.policy.pt')
        pid=PIDController(c,PIDConfig.from_json('configs/pid.json'));rows=[];results=[]
        for mode in ['pid','failed_rl']:
            env=ThrusterLandingEnv(c);obs,info=env.reset(seed=scenario['seed'],options=scenario['options']);pid.reset();ret=0.;k=0;start=time.perf_counter()
            rows.append(dict(controller=mode,step=0,time=0.,**info['state'],action=-1,outcome='flying',reward=0.))
            while True:
                action=pid.act(env.state,info['wind_force']) if mode=='pid' else policy.act(obs,explore=False)
                obs,r,t,tr,info=b.step(env,action);k+=1;ret+=r;rows.append(dict(controller=mode,step=k,time=info['time'],**info['state'],action=action,outcome=info['outcome'],reward=r))
                if t or tr:break
            s=info['state'];th=s['theta'];feet=[(s['x']+sx*np.cos(th)+np.sin(th),s['y']+sx*np.sin(th)-np.cos(th)) for sx in [-.6,.6]]
            body_y=[s['y']+sx*np.sin(th)+sy*np.cos(th) for sx in [-.7,.7] for sy in [-.8,.8]]
            independently_valid=(min(y for x,y in feet)<=1e-6 and max(y for x,y in feet)<=.15 and min(body_y)>1e-6 and all(abs(x)<=2.5 for x,y in feet) and abs(s['x'])<=1.5 and abs(s['vx'])<=.7 and -1<=s['vy']<=0 and abs(th)<=.12 and abs(s['omega'])<=.25)
            assert independently_valid==info['is_success']
            results.append(dict(controller=mode,steps=k,outcome=info['outcome'],success=info['is_success'],return_=ret,wall_seconds=time.perf_counter()-start,final_state=s,contact=info['contact']))
            env.close()
        write_csv(out/'reproduced_trajectories.csv',rows)
        prior={}
        for name in ['exploracion_resultados.csv','replay150k_resultados.csv']:
            import pandas as pd
            prior[name]=pd.read_csv('docs/resultados/'+name).to_dict('records')
        (out/'report.json').write_text(json.dumps(dict(independent_dynamics=checks,free_fall_contact=check_contact,reproductions=results,prior_results=prior),indent=2)+'\n')
        print(json.dumps(dict(reproductions=results,dynamics_max_error=max(r['error_default_max_SI'] for r in checks)),indent=2),flush=True)
    finally:b.close()
if __name__=='__main__':main()
