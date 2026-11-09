"""Replay circular uniforme, sin reemplazo dentro de cada minibatch."""
import copy
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, observation_dim, seed=0):
        if capacity < 1 or observation_dim < 1:
            raise ValueError("Dimensiones del replay deben ser positivas")
        self.capacity, self.observation_dim = capacity, observation_dim
        self.rng = np.random.default_rng(seed)
        self.position = self.size = 0
        self.arrays = {
            "observations": np.zeros((capacity, observation_dim), dtype=np.float32),
            "next_observations": np.zeros((capacity, observation_dim), dtype=np.float32),
            "actions": np.zeros(capacity, dtype=np.int64),
            "rewards": np.zeros(capacity, dtype=np.float32),
            "terminated": np.zeros(capacity, dtype=np.bool_),
            "truncated": np.zeros(capacity, dtype=np.bool_),
            "source_levels": np.zeros(capacity, dtype=np.int8),
        }

    def __len__(self):
        return self.size

    def add(self, observation, action, reward, next_observation, terminated, truncated, source_level=0):
        for key, value in (("observations", observation), ("actions", action), ("rewards", reward),
                           ("next_observations", next_observation), ("terminated", terminated), ("truncated", truncated), ("source_levels", source_level)):
            self.arrays[key][self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device="cpu"):
        if not 0 < batch_size <= self.size:
            raise ValueError("No hay suficientes transiciones para el minibatch")
        indices = self.rng.choice(self.size, batch_size, replace=False)
        self.last_sample_indices = indices.copy()  # Diagnostic only; no additional RNG draws.
        return {key: torch.as_tensor(array[indices], device=device) for key, array in self.arrays.items()}

    def state_dict(self):
        return {"capacity": self.capacity, "observation_dim": self.observation_dim,
                "position": self.position, "size": self.size, "rng": copy.deepcopy(self.rng.bit_generator.state),
                "arrays": {key: torch.from_numpy(value[:self.size].copy()) for key, value in self.arrays.items()}}

    def load_state_dict(self, state):
        if (state["capacity"], state["observation_dim"]) != (self.capacity, self.observation_dim):
            raise ValueError("Dimensiones del replay incompatibles")
        self.position, self.size = state["position"], state["size"]
        if not 0 <= self.size <= self.capacity or not 0 <= self.position < self.capacity:
            raise ValueError("Contadores del replay inválidos")
        for key, value in self.arrays.items():
            if key == "source_levels" and key not in state["arrays"]:
                value[:self.size] = 0  # Legacy replay has no curriculum provenance.
                continue
            saved = state["arrays"][key].cpu().numpy()
            if saved.shape != value[:self.size].shape:
                raise ValueError("Forma del replay inválida")
            value[:self.size] = saved
        self.rng.bit_generator.state = state["rng"]
