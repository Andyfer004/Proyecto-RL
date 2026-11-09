"""Animate saved states; no policy selection, simulation, training or new evaluation."""
import argparse,json,os
from pathlib import Path
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
import pandas as pd
from PIL import Image
from thruster_landing.envs import EnvConfig
from thruster_landing.envs.physics import State
from thruster_landing.envs.rendering import Renderer


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,default=Path('artifacts/scope_alignment_v1/final'))
    p.add_argument('--scenario',default='complement-wind_p40-000')
    p.add_argument('--methods',nargs='+',default=['pid','dqn_assisted42','double_assisted42'])
    p.add_argument('--human',action='store_true',help='Real-time replay in a local window; no keyboard control claimed')
    a=p.parse_args();config=EnvConfig.from_json('configs/env.json')
    for method in a.methods:
        data=pd.read_csv(a.results/(method+'.traces.csv'));data=data[data.scenario_id==a.scenario]
        if data.empty:raise ValueError('Scenario absent')
        renderer=Renderer('human' if a.human else 'rgb_array',config);frames=[]
        try:
            for index,row in enumerate(data.itertuples()):
                state=State(**{k:getattr(row,k) for k in ['x','y','vx','vy','theta','omega','gas']})
                frame=renderer.draw(state,max(0,int(row.action)),row.time,row.wind_force_N)
                if a.human:
                    if renderer.close_requested:break
                elif index%2==0 or index==len(data)-1:frames.append(Image.fromarray(frame).resize((630,560)))
        finally:renderer.close()
        if frames:
            path=a.results/(method+'.'+a.scenario+'.gif')
            frames[0].save(path,save_all=True,append_images=frames[1:],duration=100,loop=0)
            print(path)

if __name__=='__main__':main()
