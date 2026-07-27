""" Test Dependency Installation

The purpose is to check if core dependencies are installed properly.
Typically, failure to these tests indicate an incorrection installation 
or wrong activation of the virtual environment (i.e. conda, venv, etc.).

"""
def test_gymnasium():
    import gymnasium

    assert gymnasium.__version__

def test_pyelastica():
    from importlib.metadata import version

    assert version("pyelastica")
