"""
Created on Oct. 19, 2020
@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np
from numba import njit
from tqdm import tqdm

from collections import defaultdict
import pickle

import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

from elastica._linalg import _batch_cross, _batch_matvec
from elastica._calculus import quadrature_kernel
from elastica.rod.cosserat_rod import CosseratRod

from gym_softrobot.utils.actuation.algorithms.algorithm import Algorithm
from gym_softrobot.utils.actuation.objects.point import PointTarget
from gym_softrobot.utils.actuation.rod_tools import forward_path, calculate_dilatation#, calculate_potential_energy

label_fontsize = 14

def strain_to_shear_and_curvature(strain):
    n_elems = int((strain.size + 3) / 6)
    shear = np.zeros((3, n_elems))
    curvature = np.zeros((3, n_elems-1))
    index = 0
    for i in range(3):
        shear[i, :] = strain[index:index+n_elems]
        index += n_elems
    for i in range(3):
        curvature[i] = strain[index:index+n_elems-1]
        index += (n_elems-1)
    return shear, curvature

def shear_and_curvature_to_strain(shear, curvature):
    strain = np.zeros(shear.size + curvature.size)
    n_elems = shear.shape[1]
    index = 0
    for i in range(3):
        strain[index:index+n_elems] = shear[i, :]
        index += n_elems
    for i in range(3):
        strain[index:index+n_elems-1] = curvature[i]
        index += (n_elems-1)
    return strain

@njit(cache=True)
def inverse_rigidity_matrix(matrix):
    inverse_matrix = np.zeros(matrix.shape)
    for i in range(inverse_matrix.shape[2]):
        inverse_matrix[:, :, i] = np.linalg.inv(matrix[:, :, i])
    return inverse_matrix

@njit(cache=True)
def backward_path(dl, director, shear, curvature, internal_force, internal_couple):
    blocksize = internal_force.shape[1]
    for i in range(blocksize-1):
        internal_force[:, -1-i-1] = internal_force[:, -1-i] + (
            np.cross(curvature[:, -1-i], internal_force[:, -1-i])
            ) * dl
    nu_cross_n = _batch_cross(shear, internal_force)

    for i in range(blocksize-2):
        kappa_cross_m = np.cross(curvature[:, -1-i], internal_couple[:, -1-i])
        internal_couple[:, -1-i-1] = internal_couple[:, -1-i] + (
            kappa_cross_m
            + (nu_cross_n[:, -1-i] + nu_cross_n[:, -1-i-1])/2
            ) * dl
    pass

@njit(cache=True)
def update_strain(shear, curvature, internal_force, internal_couple, shear_matrix, bend_matrix, eta_shear, eta_curvature):
    shear_intrinsic = np.zeros(shear.shape)
    shear_intrinsic[2, :] = 1
    shear[:, :] += eta_shear * (internal_force - _batch_matvec(shear_matrix, shear-shear_intrinsic))
    shear[2, :] = np.maximum(shear[2, :], 0)
    curvature[:, :] += eta_curvature * (internal_couple - _batch_matvec(bend_matrix, curvature))
    pass

class ForwardBackward(Algorithm):
    def __init__(self, rod, algo_config):
        Algorithm.__init__(self, rod, algo_config)
        self.eta_shear = np.zeros(self.shear.shape)
        self.eta_curvature = np.zeros(self.curvature.shape)

        min_rigidity = np.inf
        for i in range(3):
            min_rigidity_i = min(np.min(self.shear_matrix[i, i, :]), np.min(self.shear_matrix[i, i, :]))
            min_rigidity = min(min_rigidity, min_rigidity_i)
            
        for i in range(3):
            self.eta_shear[i, :] = algo_config.get('eta', 1)/(self.shear_matrix[i, i, :]/min_rigidity)
            self.eta_curvature[i, :] = algo_config.get('eta', 1)/(self.bend_matrix[i, i, :]/min_rigidity)

    def update(self,):
        forward_path(self.dl, self.shear, self.curvature, self.position, self.director)
        terminal_cost_partial_end_position, terminal_cost_partial_end_director = (
            self.target.terminal_cost_partial_end_state(
                self.position[:, -1],
                end_director=self.director[:, :, -1]
            )
        )
        # print(terminal_cost_partial_end_director)
        # print(self.position)

        self.internal_force[:, -1] = - self.director[:, :, -1] @ terminal_cost_partial_end_position
        self.internal_couple[:, -1] = terminal_cost_partial_end_director
        backward_path(self.dl, self.director, self.shear, self.curvature, self.internal_force, self.internal_couple)
        
        update_strain(
            self.shear, self.curvature,
            self.internal_force, self.internal_couple,
            self.shear_matrix, self.bend_matrix,
            self.eta_shear, self.eta_curvature
        )
        
        pass

    def run(self, continuum_target_cost_weight=None, plot_flag=False, iter_number=100000):
        for _ in tqdm(range(iter_number)):
            self.update()
        return

def reach_test_function(
        sampling_number=1000,
        iter_number=600000,
        error_tolerance=1e-6
    ):
    n = 100
    L0 = 0.2

    """ Set up arm params """
    radius_base = 0.012  # radius of the arm at the base
    radius_tip = 0.012  # radius of the arm at the tip
    # radius_tip = 0.001  # radius of the arm at the tip
    radius = np.linspace(radius_base, radius_tip, n+1)
    radius_mean = (radius[:-1]+radius[1:])/2
    damp_coefficient = 0.03

    rod = CosseratRod.straight_rod(
        n_elements=n,
        start=np.zeros((3,)),
        direction=np.array([1.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, -1.0]),
        base_length=L0,
        base_radius=radius_mean.copy(),
        density=700,
        nu=damp_coefficient*((radius_mean/radius_base)**2),
        youngs_modulus=1e4,
        poisson_ratio=0.5,
        nu_for_torques=damp_coefficient*((radius_mean/radius_base)**4),
    )

    target_position = np.array([0.5, 0.5, 0])
    target_director = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    algo_config = dict(
        target=PointTarget(
            position=target_position*L0,
            director=target_director,
            target_cost_weight=100000),
        eta=1e-10,
    )

    algo = ForwardBackward(rod, algo_config)

    # target_cost_weight_array = np.linspace(1000000, 1000000, 1)
    target_cost_weight_array = np.logspace(0, 6, 6)

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(
        nrows=3, ncols=4, figure=fig, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1, 1]
    )
    # ax_2d = fig.add_subplot(gs[0:2, 0:2])
    ax_3d = fig.add_subplot(gs[0:2, 0:2], projection='3d')
    ax_J1 = fig.add_subplot(gs[2, 0])
    ax_J2 = fig.add_subplot(gs[2, 1])
    ax_shear = []
    ax_curvature = []
    for index_i in range(3):
        ax_shear.append(fig.add_subplot(gs[index_i, 2], xlim=[-0.1, 1.1]))
        ax_curvature.append(fig.add_subplot(gs[index_i, 3], xlim=[-0.1, 1.1]))

    # alphas = np.zeros(algo.alpha.shape + target_cost_weight_array.shape)
    running_costs = np.zeros(target_cost_weight_array.shape)
    terminal_costs = np.zeros(target_cost_weight_array.shape)
    # print(algo.position)
    # algo.run(ax_2d, iter_number=100000)
    # print(algo.position)
    for i in tqdm(range(iter_number)):
        # print("123456")
        if (i % sampling_number) == 0:
            ax_3d.plot(
                algo.position[0, :]/L0,
                algo.position[1, :]/L0,
                algo.position[2, :]/L0,
                color='#1f77b4',
                alpha=i/iter_number
            )
            # ax_2d.plot(
            #     algo.position[0, :]/L0,
            #     algo.position[1, :]/L0,
            #     color='#1f77b4',
            #     alpha=i/iter_number
            # )
            # force = -algo.target.terminal_cost_partial_end_state(algo.position[:, -1], algo.director[:, :, -1])
            # v = np.zeros((3, 2))
            # v[:, 0] = algo.position[:, -1]/L0
            # v[:, 1] = algo.position[:, -1]/L0 + force/np.linalg.norm(force)*0.1
            # ax_2d.plot(v[0, :], v[1, :])
            for index_i in range(3):
                ax_shear[index_i].plot(algo.s_shear, algo.shear[index_i, :], color='#1f77b4', alpha=i/iter_number)
                ax_curvature[index_i].plot(algo.s_curvature, algo.curvature[index_i, :], color='#1f77b4', alpha=i/iter_number)
                # print(np.max(algo.curvature[2, :]), np.min(algo.curvature[2, :]))
        # if i==2 :
        #     quit()
        algo.update()


    # for cost_weight_i, target_cost_weight in enumerate(target_cost_weight_array):
    #     algo_config['target'].target_cost_weight = target_cost_weight
    #     algo.run()
    #     # alphas[:, cost_weight_i] = algo.alpha.copy()

    #     running_costs[cost_weight_i] = algo.running_cost
    #     terminal_costs[cost_weight_i] = algo.terminal_cost
        
    #     ax_2d.plot(algo.position_collection[0, :]/L0, algo.position_collection[1, :]/L0)
    #     for index_i in range(3):
    #         ax_strain[index_i].plot(algo.s_strain, algo.strain[index_i, :])
    #         ax_curvature[index_i].plot(algo.s_curvature, algo.curvature[index_i, :])

    # for alpha_i in range(algo.alpha.shape[0]):
    #     ax_J1.plot(target_cost_weight_array, alphas[alpha_i], label=r'$\alpha_{}$'.format(alpha_i+1))
    ax_J1.legend()
    ax_J2.semilogy(target_cost_weight_array, running_costs+terminal_costs, label=r'$\mathsf{J}$', color='black')
    ax_J2.semilogy(target_cost_weight_array, running_costs, label='$V$', color='black', linestyle='--')
    ax_J2.semilogy(target_cost_weight_array, terminal_costs, label='$\Phi$', color='black', linestyle='dotted')
    ax_J2.legend()
    
    y_lim = [-1.1, 1.1]
    ax_J1.set_ylim(y_lim)
    
    # y_lim = np.array(ax_J2.get_ylim())
    # y_lim[0] = min(np.min(running_costs[1:]),np.min(terminal_costs[1:]))
    y_lim = np.array([0.009, 2.1])
    ax_J2.set_ylim(y_lim)
    fig.text(0.3, 0.04, "continuation parameter $\mu$", ha='center', va='center', fontsize=label_fontsize)
    fig.text(0.71, 0.04, '$s$', ha='center', va='center', fontsize=label_fontsize)

    ax_3d.scatter(
        algo.target.position[0]/L0,
        algo.target.position[1]/L0,
        algo.target.position[2]/L0,
        c='r', marker='x'
    )
    # ax_2d.scatter(
    #     algo.target.position[0]/L0,
    #     algo.target.position[1]/L0,
    #     c='r', marker='x'
    # )
    # ax_2d.set_xlim([-0.1, 1.1])
    # ax_2d.set_ylim([-0.1, 1.1])
    ax_3d.set_xlim([-0.1, 1.1])
    ax_3d.set_ylim([-0.1, 1.1])
    ax_3d.set_zlim([-0.6, 0.6])
    ax_shear[0].set_title('shear', fontsize=label_fontsize)
    ax_curvature[0].set_title('curvature', fontsize=label_fontsize)
    for index_i in range(3):
        ax_curvature[index_i].text(1.2, 0, '$d_{}$'.format(index_i+1), ha='center', va='center', fontsize=label_fontsize)
    for index_i in range(2):
        ax_shear[index_i].set_ylim([-0.11, 0.11])
        ax_shear[index_i].ticklabel_format(axis='y', scilimits=(-1, -1))  
        ax_curvature[index_i].set_ylim([-110, 110])
        ax_curvature[index_i].ticklabel_format(axis='y', scilimits=(2, 2))
    ax_shear[2].set_ylim([-0.1, 2.1])
    ax_shear[2].text(0, 2, 'stretch', fontsize=label_fontsize, ha='left', va='top')
    ax_curvature[2].set_ylim([-11, 11])
    ax_curvature[2].ticklabel_format(axis='y', scilimits=(1, 1))
    ax_curvature[2].text(1, 10, 'twist', fontsize=label_fontsize, ha='right', va='top')
    plt.show()

if __name__ == "__main__":
    reach_test_function()
