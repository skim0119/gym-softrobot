import pytest

""" Test Dependency Installation

The purpose is to check if core dependencies are installed properly.
Typically, failure to these tests indicate an incorrection installation 
or wrong activation of the virtual environment (i.e. conda, venv, etc.).

"""
def test_gym():
    import gym
    version = gym.version.VERSION
    assert version

def test_pyelastica():
    import elastica
