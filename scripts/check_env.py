"""Verificación oficial de contrato Gymnasium, sin interfaz gráfica."""
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import thruster_landing  # Registra el entorno.

if __name__ == "__main__":
    env = gym.make("ThrusterLanding-v0")
    try:
        check_env(env.unwrapped, skip_render_check=True)
        print("Gymnasium: contrato válido")
    finally:
        env.close()
