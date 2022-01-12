"""
Created on Fri Apr. 17, 2020
@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np
from numba import njit

from gym_softrobot.utils.actuation.objects.object import Constraint, Target

from elastica._rotations import _inv_rotate

class CylinderConstraint(Constraint):
    def __init__(self, center, normal, radius, height, constraint_cost_weight):
        Constraint.__init__(self, 'cylinder', constraint_cost_weight)
        self.center = np.array(center)
        self.normal = np.array(normal)
        self.normal = self.normal/np.linalg.norm(self.normal)
        self.radius = radius
        self.height = height
        self.indicator = None
        self.closest_position = None

    def running_cost(self, position, radii):
        self.indicator = np.zeros(position.shape[1])
        self.closest_position = np.zeros(position.shape)
        indicator(self.center, self.normal, self.radius, position, radii, self.indicator, self.closest_position)
        constraint_running_cost(self.closest_position, position, self.indicator, self.cost, self.constraint_cost_weight)
        return

    def running_cost_partial_position(self, position):
        return constraint_running_cost_partial_position(self.closest_position, position, self.indicator, self.constraint_cost_weight)

    def calculate_cost(self, position, radii):
        self.cost[0] = 0
        self.running_cost(position, radii)
        return

@njit(cache=True)
def indicator(center, normal, radius, position, radii, indicator_array, closest_position):
    for i in range(indicator_array.shape[0]):
        delta_position = position[:, i] - center
        normal_delta_position = np.dot(normal, delta_position) * normal
        parallel_delta_position = delta_position - normal_delta_position
        temp_center = center + normal_delta_position

        vector_length = np.linalg.norm(parallel_delta_position)
        equivalent_distance = radii[i] + radius
        value = vector_length - equivalent_distance
        closest_position[:, i] = temp_center + (equivalent_distance/vector_length)*parallel_delta_position
        # closest_position[:, i] += center
        if value > 0: # outside
            indicator_array[i] = 1
        elif value < 0: # inside
            indicator_array[i] = -1
        else:
            indicator_array[i] = 0
    return

@njit(cache=True)
def constraint_running_cost(closest_position, position, indicator_array, cost, constraint_cost_weight):
    for i in range(indicator_array.shape[0]):
        if indicator_array[i] < 0:
            cost[0] += 0.5*constraint_cost_weight * \
                ((position[0, i]-closest_position[0, i])**2 + \
                 (position[1, i]-closest_position[1, i])**2 + \
                 (position[2, i]-closest_position[2, i])**2)
    return

@njit(cache=True)
def constraint_running_cost_partial_position(closest_position, position, indicator_array, constraint_cost_weight):
    internal_force = np.zeros(position.shape)
    for i in range(indicator_array.shape[0]):
        if indicator_array[i] < 0:
            internal_force[:, i] = constraint_cost_weight * (closest_position[:, i]-position[:, i])
    return internal_force


class CylinderTarget(Target, CylinderConstraint):
    def __init__(self, position, normal, radius, heights, constraint_cost_weight, target_running_cost_weight, target_cost_weight=0):
        Target.__init__(self, 'cylinder', position, None, target_cost_weight)
        assert (heights[0]*heights[1]) <= 0, \
            "one of the two entries in heights should be nonnegative and the other should be nonpositive."
        assert not ((heights[0] == 0) and (heights[1] == 0)), \
            "two entries in heights can not all be zero."
        self.heights = dict(
            upper=max(heights[0], heights[1]),
            lower=-min(heights[0], heights[1])
            )
        self.normal = np.array(normal)
        self.normal = self.normal/np.linalg.norm(self.normal)
        self.target_running_cost_weight = np.array(target_running_cost_weight)
        CylinderConstraint.__init__(self,
            self.position - self.heights['lower']*self.normal,
            self.normal, radius, self.heights['upper']+self.heights['lower'], constraint_cost_weight)

    def terminal_cost(self, end_position):
        self.cost[0] += 0.5 * self.target_cost_weight * np.linalg.norm(self.position - end_position)**2
        return

    def terminal_cost_partial_end_state(self, end_position, end_director=None):
        terminal_cost_partial_end_position = self.target_cost_weight * (end_position - self.position)

        if not (self.director == None):
            director_collection = np.zeros((3, 3, 2))
            director_collection[:, :, 0] = end_director
            director_collection[:, :, 1] = self.director
            rotate_axis_and_angle = _inv_rotate(director_collection)
            terminal_cost_partial_end_director = self.target_cost_weight * rotate_axis_and_angle[:, 0]
            return terminal_cost_partial_end_position, terminal_cost_partial_end_director

        return terminal_cost_partial_end_position, None
        
    def terminal_cost_partial_end_position(self, end_position):
        return self.target_cost_weight * (end_position - self.position)

    def running_cost(self, position, radii):
        self.indicator = np.zeros(position.shape[1])
        self.closest_position = np.zeros(position.shape)
        indicator(self.center, self.normal, self.radius, position, radii, self.indicator, self.closest_position)
        target_running_cost(self.closest_position, position, self.indicator, self.cost, self.constraint_cost_weight, self.target_running_cost_weight)
        return

    def running_cost_partial_position(self, position):
        return target_running_cost_partial_position(self.closest_position, position, self.indicator, self.constraint_cost_weight, self.target_running_cost_weight)

    def calculate_cost(self, end_position, position, radii):
        self.cost[0] = 0
        self.running_cost(position, radii)
        if not (self.target_cost_weight == 0):
            self.terminal_cost(end_position)
        return

@njit(cache=True)
def target_running_cost(closest_position, position, indicator_array, cost, constraint_cost_weight, target_running_cost_weight):
    for i in range(indicator_array.shape[0]):
        if indicator_array[i] < 0:
            cost[0] += 0.5*constraint_cost_weight * \
                ((position[0, i]-closest_position[0, i])**2 + \
                 (position[1, i]-closest_position[1, i])**2 + \
                 (position[2, i]-closest_position[2, i])**2)
        if indicator_array[i] > 0:
            cost[0] += 0.5*target_running_cost_weight[i] * \
                ((position[0, i]-closest_position[0, i])**2 + \
                 (position[1, i]-closest_position[1, i])**2 + \
                 (position[2, i]-closest_position[2, i])**2)
    return

@njit(cache=True)
def target_running_cost_partial_position(closest_position, position, indicator_array, constraint_cost_weight, target_running_cost_weight):
    internal_force = np.zeros(position.shape)
    for i in range(indicator_array.shape[0]):
        if indicator_array[i] < 0:
            internal_force[:, i] = constraint_cost_weight * (closest_position[:, i]-position[:, i])
        if indicator_array[i] > 0:
            internal_force[:, i] = target_running_cost_weight[i] * (closest_position[:, i]-position[:, i])
    return internal_force
