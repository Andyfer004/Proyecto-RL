"""DQN / Double DQN compartidos: la única diferencia es el target de bootstrap."""
from dataclasses import asdict, dataclass
import copy
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .replay import ReplayBuffer


@dataclass(frozen=True)
class DQNConfig:
    algorithm: str = "dqn"
    seed: int = 42
    device: str = "cpu"
    learning_rate: float = 0.0003
    gamma: float = 0.99
    buffer_capacity: int = 50000
    batch_size: int = 64
    learning_starts: int = 1000
    train_frequency: int = 4
    target_update_frequency: int = 500  # pasos de entorno, copia dura
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 8000  # después de learning_starts
    total_steps: int = 10000
    validation_frequency: int = 5000
    checkpoint_frequency: int = 5000
    log_frequency: int = 100
    gradient_norm_limit: float = 10.0
    hidden_sizes: tuple[int, int] = (128, 128)
    torch_threads: int = 1
    detailed_validation: bool = False  # baseline inicial y trazas inicial/mejor/final
    smooth_l1_beta: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "hidden_sizes", tuple(self.hidden_sizes))
        if not isinstance(self.detailed_validation, bool):
            raise ValueError("detailed_validation debe ser booleano")
        if self.algorithm not in ("dqn", "double_dqn") or self.device not in ("cpu", "cuda", "mps"):
            raise ValueError("Algoritmo o dispositivo inválido")
        for name in ("seed", "buffer_capacity", "batch_size", "learning_starts", "train_frequency",
                     "target_update_frequency", "epsilon_decay_steps", "total_steps", "validation_frequency",
                     "checkpoint_frequency", "log_frequency", "torch_threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name in ("seed", "learning_starts") else 1):
                raise ValueError(f"{name} inválido")
        if self.batch_size > self.buffer_capacity:
            raise ValueError("batch_size supera buffer_capacity")
        for name in ("learning_rate", "gradient_norm_limit", "smooth_l1_beta"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} debe ser positivo y finito")
        if not 0 <= self.gamma <= 1 or not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("gamma o epsilon fuera de rango")
        if len(self.hidden_sizes) != 2 or any(type(size) is not int or size < 1 for size in self.hidden_sizes):
            raise ValueError("Se requieren dos capas ocultas positivas")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text()))


class QNetwork(nn.Sequential):
    def __init__(self, observation_dim, action_count, hidden_sizes=(128, 128)):
        first, second = hidden_sizes
        super().__init__(nn.Linear(observation_dim, first), nn.ReLU(),
                         nn.Linear(first, second), nn.ReLU(), nn.Linear(second, action_count))


@torch.no_grad()
def compute_targets(rewards, terminated, next_online_q, next_target_q, gamma, double_dqn=False):
    """Truncamientos artificiales conservan bootstrap; solo terminated lo anula.

    El llamador conserva la observación final real, nunca la del próximo reset.
    """
    if double_dqn:
        selected = next_online_q.argmax(dim=1, keepdim=True)
        next_values = next_target_q.gather(1, selected).squeeze(1)
    else:
        next_values = next_target_q.max(dim=1).values
    return rewards + gamma * torch.where(terminated.bool(), torch.zeros_like(next_values), next_values)


class DQNAgent:
    def __init__(self, config, observation_dim=9, action_count=8):
        self.config, self.observation_dim, self.action_count = config, observation_dim, action_count
        if config.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA no disponible; usa device=cpu")
        if config.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS no disponible; usa device=cpu")
        self.device = torch.device(config.device)
        torch.set_num_threads(config.torch_threads)
        torch.manual_seed(config.seed)
        self.online = QNetwork(observation_dim, action_count, config.hidden_sizes).to(self.device)
        self.target = copy.deepcopy(self.online).eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.loss_function = nn.SmoothL1Loss(beta=config.smooth_l1_beta, reduction="mean")
        self.rng = np.random.default_rng(config.seed)
        self.replay = ReplayBuffer(config.buffer_capacity, observation_dim, config.seed + 1)
        self.env_steps = self.updates = self.target_syncs = 0

    @property
    def epsilon(self):
        c = self.config
        fraction = np.clip((self.env_steps - c.learning_starts) / c.epsilon_decay_steps, 0, 1)
        return float(c.epsilon_start + fraction * (c.epsilon_end - c.epsilon_start))

    def act(self, observation, explore=True):
        if explore and (self.env_steps < self.config.learning_starts or self.rng.random() < self.epsilon):
            self.last_selection_mode = "exploration"
            return int(self.rng.integers(self.action_count))
        self.last_selection_mode = "exploitation" if explore else "evaluation"
        was_training = self.online.training
        self.online.eval()
        with torch.inference_mode():
            q = self.online(torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0))
            action = int(q.argmax(dim=1).item())
        self.online.train(was_training)
        return action

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())
        self.target_syncs += 1

    def learn(self):
        c = self.config
        batch = self.replay.sample(c.batch_size, self.device)
        q = self.online(batch["observations"]).gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_target = self.target(batch["next_observations"])
            next_online = self.online(batch["next_observations"]) if c.algorithm == "double_dqn" else next_target
            targets = compute_targets(batch["rewards"], batch["terminated"], next_online, next_target,
                                      c.gamma, c.algorithm == "double_dqn")
        observer = getattr(self, "learning_observer", None)
        if observer is not None:
            observer.record(self.env_steps, self.replay.last_sample_indices, q, targets)
        loss = self.loss_function(q, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Loss no finita")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.online.parameters(), c.gradient_norm_limit, error_if_nonfinite=True)
        # Telemetría: solo leer gradientes ya recortados; no altera el optimizador.
        norm_after = torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(parameter.grad.detach())
            for parameter in self.online.parameters() if parameter.grad is not None]))
        self.optimizer.step()
        self.updates += 1
        return {"loss": float(loss.detach()), "mean_abs_td_error": float((targets - q).detach().abs().mean()),
                "mean_abs_q": float(q.detach().abs().mean()), "gradient_norm_before_clip": float(norm),
                "gradient_norm_after_clip": float(norm_after)}

    def state_dict(self):
        return {"config": self.config.to_dict(), "observation_dim": self.observation_dim,
                "action_count": self.action_count, "online": self.online.state_dict(),
                "target": self.target.state_dict(), "optimizer": self.optimizer.state_dict(),
                "env_steps": self.env_steps, "updates": self.updates, "target_syncs": self.target_syncs,
                "rng": copy.deepcopy(self.rng.bit_generator.state), "replay": self.replay.state_dict()}

    def load_state_dict(self, state):
        if DQNConfig(**state["config"]) != self.config:
            # El único cambio admitido es ampliar/reducir el presupuesto total futuro.
            saved = {**state["config"], "total_steps": self.config.total_steps}
            if DQNConfig(**saved) != self.config:
                raise ValueError("Configuración incompatible para reanudar")
        if (state["observation_dim"], state["action_count"]) != (self.observation_dim, self.action_count):
            raise ValueError("Dimensiones del agente incompatibles")
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.env_steps, self.updates, self.target_syncs = state["env_steps"], state["updates"], state["target_syncs"]
        self.rng.bit_generator.state = state["rng"]
        self.replay.load_state_dict(state["replay"])
