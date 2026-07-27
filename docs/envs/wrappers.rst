External integrations
=====================

gym-softrobot contains experimental adapters for multi-agent reinforcement
learning frameworks.

PyMARL
------

``ConvertToPyMarlEnv`` adapts a compatible gym-softrobot multi-agent
environment to PyMARL's expected observation, state, reward, and episode
interface.

.. code-block:: python

   import gymnasium as gym
   import gym_softrobot
   from gym_softrobot.wrapper import ConvertToPyMarlEnv

   base_env = gym.make("OctoCrawl-v0")
   env = ConvertToPyMarlEnv(base_env.unwrapped)

The adapter is experimental and does not make every gym-softrobot environment
multi-agent compatible.

RLlib
-----

The shared-policy adapter requires Ray RLlib. Ray is not part of the core
installation, so install a compatible Ray release before importing that
adapter.
