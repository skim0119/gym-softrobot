"""
Created on Oct. 19, 2020
@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np

from gym_softrobot.utils.actuation.objects.object import Target

from elastica._rotations import _inv_rotate

class PointTarget(Target):
    def __init__(self, position, director=None, target_cost_weight=None):
        Target.__init__(self, 'point', position, director, target_cost_weight)
    
    def __str__(self,):
        result = "Target is a point with position at (%f, %f, %f)." % (self.position[0], self.position[1], self.position[2])
        if not (self.director == None):
            result += "and with target orientation."
        return result

    def terminal_cost(self, end_position, end_director=None):
        self.cost[0] = 0.5 * self.target_cost_weight * np.linalg.norm(self.position - end_position)**2
        if self.director is not None:
            director_collection = np.zeros((3, 3, 2))
            director_collection[:, :, 0] = end_director
            director_collection[:, :, 1] = self.director
            # print(director_collection)
            rotate_axis_and_angle = _inv_rotate(director_collection)
            # print(rotate_axis_and_angle)
            # quit()

        return self.cost[0]

    def terminal_cost_partial_end_state(self, end_position, end_director=None):
        terminal_cost_partial_end_position = self.target_cost_weight * (end_position - self.position)

        if self.director is not None:
            director_collection = np.zeros((3, 3, 2))
            director_collection[:, :, 0] = end_director
            director_collection[:, :, 1] = self.director
            rotate_axis_and_angle = _inv_rotate(director_collection)
            terminal_cost_partial_end_director = self.target_cost_weight * rotate_axis_and_angle[:, 0]
            return terminal_cost_partial_end_position, terminal_cost_partial_end_director

        return terminal_cost_partial_end_position, None

    def calculate_cost(self, position, director, radii):
        return self.terminal_cost(position[:, -1], director[:, :, -1])
