"""
Created on Oct. 19, 2020
@author: Heng-Sheng (Hanson) Chang
"""

from gym_softrobot.utils.actuation.rod_tools import (
    calculate_dilatation,
    sigma_to_shear,
    kappa_to_curvature,
)

import numpy as np


class Algorithm(object):
    def __init__(self, rod, algo_config):
        
        self.rod = rod
        self.config = algo_config


        '''
        make a copy of the required info of the rod 
        so that it will not affect the values of the origin copy
        '''
        
        self.n_elems = rod.n_elems
        self.reference_length = np.sum(rod.rest_lengths)

        # self.ds = 1 / self.n_elems
        # self.dl = self.reference_length / self.n_elems
        self.dl = rod.rest_lengths.copy()
        self.ds = self.dl / self.reference_length
        self.s = np.insert(np.cumsum(self.ds), 0, 0)
        self.s_position = self.s.copy()
        self.s_director = (self.s[:-1] + self.s[1:])/2
        self.s_shear = (self.s[:-1] + self.s[1:])/2
        self.s_kappa = self.s[1:-1]

        self.position = rod.position_collection.copy()
        self.director = rod.director_collection.copy()

        # _, voronoi_dilatation = calculate_dilatation(rod.sigma)
        self.shear = sigma_to_shear(rod.sigma)
        # self.curvature = kappa_to_curvature(rod.kappa, voronoi_dilatation)

        # # self.shear = rod.sigma + np.array([0, 0, 1])[:, None]
        # # voronoi_dilatation = np.ones(rod.voronoi_dilatation.shape)
        # # self.curvature = rod.kappa / voronoi_dilatation

        # _, voronoi_dilatation = calculate_dilatation(rod.rest_sigma)
        self.rest_shear = sigma_to_shear(rod.rest_sigma)
        # self.rest_curvature = kappa_to_curvature(rod.kappa, voronoi_dilatation)

        # # self.rest_shear = rod.rest_sigma.copy()
        # # self.rest_shear[2, :] += 1
        # # self.rest_curvature = rod.rest_kappa / rod.voronoi_dilatation

        self.sigma = rod.sigma.copy()
        self.kappa = rod.kappa.copy()

        self.rest_sigma = rod.rest_sigma.copy()
        self.rest_kappa = rod.rest_kappa.copy()

        self.radius = rod.radius.copy()
        self.shear_matrix = rod.shear_matrix.copy()
        self.bend_matrix = rod.bend_matrix.copy()

        self.internal_force = np.zeros((3, self.n_elems))
        self.internal_couple = np.zeros((3, self.n_elems-1))

        # self.target = algo_config.get('target', None)
        # self.prev_target = None

        # self.s = np.linspace(0, 1, self.n_elems+1)
        # self.s_sigma = (self.s[:-1] + self.s[1:])/2
        # self.s_kappa = self.s[1:-1]

        # self.running_cost = 0
        # self.terminal_cost = 0

    def run(self, plot_flag=False):
        raise NotImplementedError
