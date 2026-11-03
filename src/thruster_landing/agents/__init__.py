"""Controladores; importar este paquete no requiere el extra opcional PyTorch.

DQN/Double DQN se importan explícitamente desde agents.dqn.
"""
from .pid import PIDConfig, PIDController

__all__ = ["PIDConfig", "PIDController"]
