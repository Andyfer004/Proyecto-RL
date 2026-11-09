"""Prepared exploration-only experiment. Same DQN learning and greedy policy."""
import copy
import numpy as np
from .dqn import DQNAgent


def action_probabilities(protocol, action_count=8):
    if protocol.get('schema_version') != 1:
        raise ValueError('Versión de protocolo de exploración inválida')
    probabilities = np.asarray(protocol.get('action_probabilities', []), dtype=np.float64)
    if (probabilities.shape != (action_count,) or not np.isfinite(probabilities).all()
            or (probabilities <= 0).any() or not np.isclose(probabilities.sum(), 1., rtol=0, atol=1e-12)):
        raise ValueError('Se requiere probabilidad positiva para cada acción, sumando 1')
    return probabilities


class WeightedExplorationAgent(DQNAgent):
    def __init__(self, config, protocol, observation_dim=9, action_count=8):
        self.action_probabilities = action_probabilities(protocol, action_count)
        self.exploration_protocol = copy.deepcopy(protocol)
        super().__init__(config, observation_dim, action_count)

    def act(self, observation, explore=True):
        if explore and (self.env_steps < self.config.learning_starts or self.rng.random() < self.epsilon):
            self.last_selection_mode = "exploration"
            return int(self.rng.choice(self.action_count, p=self.action_probabilities))
        action = super().act(observation, explore=False)
        self.last_selection_mode = "exploitation" if explore else "evaluation"
        return action
