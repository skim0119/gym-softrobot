******************
Available Wrappers
******************

Here is the list of available wrappers that we provide. The purpose of the wrapper is to convert the environment
to make it compatible with other external packages. These wrappers should be compatible with typical `OpenAI-gym` 
wrappers, such as `SubprocVecEnv` or `VecFrameStack`, although we highly recommend using our wrapper on the most
outer-layer to be safe. Because of the nature of the wrapper design, we cannot guarantee 100% compatibility with
all other available tools. Please make an GitHub issue if you find any bug in this feature.

PyMARL Converter
================

.. _wrappers:

.. automodule::  gym_softrobot.wrapper

Description
-----------

.. autosummary::
   :nosignatures:

   ConvertToPyMarlEnv

Built-in Wrappers
-----------------

.. autoclass:: ConvertToPyMarlEnv
   :special-members: __init__
   :members:

