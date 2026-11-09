"""Frozen selection body before diagnostic labels, for numerical equivalence."""
import numpy as np
import torch

def act(self, observation, explore=True):
    if explore and (self.env_steps < self.config.learning_starts or self.rng.random() < self.epsilon):
        return int(self.rng.integers(self.action_count))
    was_training = self.online.training
    self.online.eval()
    with torch.inference_mode():
        q = self.online(torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0))
        action = int(q.argmax(dim=1).item())
    self.online.train(was_training)
    return action
