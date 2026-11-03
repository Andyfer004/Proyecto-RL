"""Entorno académico propio; importar este paquete registra el entorno."""
from gymnasium.envs.registration import register

register(id="ThrusterLanding-v0", entry_point="thruster_landing.envs:ThrusterLandingEnv")
