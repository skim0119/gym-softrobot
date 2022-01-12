"""
Created on Oct. 19, 2020
@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np

class Constraint(object):
    def __init__(self, constraint_type, constraint_cost_weight):
        self.type = constraint_type
        self.constraint_cost_weight = constraint_cost_weight
        self.cost = np.zeros(1)

class Target(object):
    def __init__(self, target_type, position, director, target_cost_weight):
        self.type = target_type
        self.position = np.array(position)
        self.director = np.array(director)
        self.target_cost_weight = target_cost_weight
        self.cost = np.zeros(1)
