import subprocess
import sys
from pathlib import Path


def test_imports_do_not_load_isaac_or_register_ids() -> None:
    script = """
import sys
import gymnasium as gym
import gym_softrobot
ids = list(gym.envs.registry.keys())
import gym_softrobot.sim
import gym_softrobot.isaac
assert list(gym.envs.registry.keys()) == ids
bad = [
    name
    for name in sys.modules
    if name.startswith(("isaaclab", "omni", "pxr"))
]
assert bad == [], bad
"""
    repo = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
