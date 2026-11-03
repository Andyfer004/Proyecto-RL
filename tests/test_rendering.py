import pytest


def test_rgb_array_and_close(monkeypatch):
    pytest.importorskip("pygame")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from thruster_landing.envs import ThrusterLandingEnv
    env = ThrusterLandingEnv(render_mode="rgb_array")
    env.reset(seed=3)
    frame = env.render()
    assert frame.shape == (800, 900, 3)
    assert frame.dtype.name == "uint8"
    env.step(1)
    assert (env.render() != frame).any()
    env.close()
    env.close()


def test_headless_does_not_import_pygame():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-c", "import sys; from thruster_landing.envs import ThrusterLandingEnv; e=ThrusterLandingEnv(); e.reset(seed=0); e.step(0); assert 'pygame' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
