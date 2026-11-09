"""Passive sampling audit. No success labels are used while collecting samples."""
import json
import numpy as np

SAMPLE_DTYPE=np.dtype([('env_steps','<i4'),('transition_step','<i4'),('q','<f4'),('target','<f4')])


def absolute_transition_steps(step, indices, capacity):
    """Map active circular slots to insertion steps (one insertion per env step)."""
    return step-((step-1-np.asarray(indices,dtype=np.int64))%capacity)


class ReplayExposure:
    def __init__(self, agent, runner, output):
        self.agent,self.runner=agent,runner
        saved=runner.get('replay_exposure')
        self.counts=np.zeros(agent.config.total_steps+1,dtype=np.int64)
        if saved:
            previous=np.asarray(saved['sample_counts'],dtype=np.int64)
            self.counts[:len(previous)]=previous
        elif agent.env_steps:
            raise ValueError('Checkpoint sin exposición previa: no inventar conteos')
        assert int(self.counts.sum())==agent.updates*agent.config.batch_size
        self.stream=(output/'learning_samples.bin').open('wb');self.rows=0
        schema=dict(dtype=SAMPLE_DTYPE.descr,meaning='One record per sampled transition, including repeats across updates; q/target before optimizer.step. No labels used to collect.',start_steps=agent.env_steps)
        (output/'learning_samples.schema.json').write_text(json.dumps(schema,indent=2)+'\n')

    def record(self, step, indices, q, targets):
        ids=absolute_transition_steps(step,indices,self.agent.replay.capacity)
        assert np.all((ids>0)&(ids<=step))
        np.add.at(self.counts,ids,1)
        rows=np.empty(len(ids),dtype=SAMPLE_DTYPE)
        rows['env_steps']=step;rows['transition_step']=ids
        rows['q']=q.detach().cpu().numpy();rows['target']=targets.detach().cpu().numpy()
        self.stream.write(rows.tobytes());self.rows+=len(rows)

    def checkpoint(self):
        self.stream.flush()
        self.runner['replay_exposure']=dict(sample_counts=self.counts.tolist(),sampled_transitions=int(self.counts.sum()))

    def close(self, output):
        self.stream.close()
        np.savez_compressed(output/'exposure_counts.npz',sample_counts=self.counts)
