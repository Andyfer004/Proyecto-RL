"""Exact historical sampling reconstruction, verified against every saved replay RNG."""
import argparse
import json
from pathlib import Path
import numpy as np
from thruster_landing.checkpoints import read_checkpoint
from thruster_landing.replay_exposure import absolute_transition_steps


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('artifacts/replay150k_weighted_seed42_100k_v1/baseline_sampling'));args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=Path('artifacts/exploration_comparison_level1_seed42_100k_v1');metadata={}
    for alg in ['dqn','double_dqn']:
        run=root/f'{alg}_weighted_variant';last=read_checkpoint(run/'last.resume.pt','resume');c=last['agent']['config'];rng=np.random.default_rng(c['seed']+1)
        steps=[];ids=[];counts=np.zeros(c['total_steps']+1,dtype=np.int64);verified=[]
        for step in range(1,c['total_steps']+1):
            if step>=c['learning_starts'] and step%c['train_frequency']==0 and min(step,c['buffer_capacity'])>=c['batch_size']:
                indices=rng.choice(min(step,c['buffer_capacity']),c['batch_size'],replace=False)
                sample=absolute_transition_steps(step,indices,c['buffer_capacity']);steps.append(step);ids.append(sample.astype(np.int32));np.add.at(counts,sample,1)
            if step%10000==0:
                checkpoint=read_checkpoint(run/f'step_{step:08d}.resume.pt','resume')
                assert rng.bit_generator.state==checkpoint['agent']['replay']['rng'],(alg,step)
                verified.append(step)
        np.savez_compressed(args.output/f'{alg}.npz',env_steps=np.asarray(steps,dtype=np.int32),transition_steps=np.stack(ids),sample_counts=counts)
        metadata[alg]=dict(verified_checkpoint_steps=verified,updates=len(steps),sampled_transitions=int(counts.sum()),q_at_updates='Not stored historically; use checkpoints on fixed transitions. No baseline retraining.',exact_rng_reconstruction=True)
        print(alg,'sampling reconstructed and checked at all10 checkpoints',flush=True)
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
if __name__=='__main__':main()
