"""
Created on Jan. 07, 2021
@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np
from numba import njit
import multiprocessing as mp

from elastica._linalg import _batch_cross
from elastica._calculus import quadrature_kernel
from elastica.external_forces import inplace_addition, NoForces

from gym_softrobot.utils.actuation.actuations.actuation import ContinuousActuation, ApplyActuation

from gym_softrobot.utils.actuation.frame_tools import change_box_to_arrow_axes

class Muscle(ContinuousActuation):
    def __init__(self, n_elements):
        ContinuousActuation.__init__(self, n_elements)
        self.n_elements = n_elements
        self.activation = np.zeros(self.n_elements-1)
        self.s = np.linspace(0, 1, self.n_elements+1)[1:-1]

    def set_activation(self, activation):
        self.activation[:] = activation.copy()
        return
    
    def get_activation(self):
        raise NotImplementedError

class MuscleForce(Muscle):
    def __init__(self, n_elements):
        Muscle.__init__(self, n_elements)
        self.distributed_activation = np.zeros(self.n_elements)

    def get_activation(self):
        redistribute_activation(self.activation, self.distributed_activation)
        return self.distributed_activation

@njit(cache=True)
def redistribute_activation(activation, distributed_activation):
    distributed_activation[0] = activation[0]/2
    distributed_activation[-1] = activation[-1]/2
    distributed_activation[1:-1] = (activation[1:] + activation[:-1])/2
    return

class MuscleCouple(Muscle):
    def __init__(self, n_elements):
        Muscle.__init__(self, n_elements)
    
    def get_activation(self):
        return self.activation

class MuscleFibers(object):
    def __init__(self, n_elements, control_numbers):
        self.controls = np.zeros(control_numbers)
        # self.G_activations = np.zeros((control_numbers, n_elements))
        self.G_internal_forces = np.zeros((control_numbers, 3, n_elements))
        self.G_internal_couples = np.zeros((control_numbers, 3, n_elements-1))
        self.G_external_forces = np.zeros((control_numbers, 3, n_elements+1))
        self.G_external_couples = np.zeros((control_numbers, 3, n_elements))
        self.G_flag = False

    @staticmethod
    @njit(cache=True)
    def muscle_fibers_function(
        controls,
        G_internal_forces, G_internal_couples,
        G_external_forces, G_external_couples,
        internal_forces, internal_couples,
        external_forces, external_couples
    ):
        for index in range(controls.shape[0]):
            inplace_addition(internal_forces, controls[index] * G_internal_forces[index, :, :])
            inplace_addition(internal_couples, controls[index] * G_internal_couples[index, :, :])
            inplace_addition(external_forces, controls[index] * G_external_forces[index, :, :])
            inplace_addition(external_couples, controls[index] * G_external_couples[index, :, :])

class ApplyMuscle(ApplyActuation):
    def __init__(self, muscles, step_skip: int, callback_params_list: list):
        ApplyActuation.__init__(self, None, step_skip, None)
        self.muscles = muscles
        self.callback_params_list = callback_params_list


    def apply_torques(self, system, time: np.float = 0.0):
        for muscle in self.muscles:
            muscle(system)
            inplace_addition(system.external_forces, muscle.external_forces)
            inplace_addition(system.external_torques, muscle.external_couples)

        self.make_callback()

    def callback_func(self):
        for muscle, callback_params in zip(self.muscles, self.callback_params_list):
            callback_params['activation'].append(muscle.activation.copy())
            callback_params['internal_force'].append(muscle.internal_forces.copy())
            callback_params['internal_couple'].append(muscle.internal_couples.copy())
            callback_params['external_force'].append(muscle.external_forces.copy())
            callback_params['external_couple'].append(muscle.external_couples.copy())

    def apply_muscles(self, muscle, system):
        muscle(system)
        inplace_addition(system.external_forces, muscle.external_forces)
        inplace_addition(system.external_torques, muscle.external_couples)

@njit(cache=True)
def local_strain(off_center_displacement, strain, curvature):
    return strain + _batch_cross(
        quadrature_kernel(curvature),
        off_center_displacement)

@njit(cache=True)
def force_length_curve_guassian(strain, intrinsic_strain=1, sigma=0.25):
    # blocksize = strain.shape[0]
    # force_weight = np.zeros(blocksize)
    # for i in range(blocksize):
    #     if strain[i] < intrinsic_strain:
    #         force_weight[i] = np.exp(-0.5*((strain[i]-intrinsic_strain)/sigma)**2)
    #     else:
    #         force_weight[i] = np.exp(-0.5*((strain[i]-intrinsic_strain)/(4*sigma))**2)
    force_weight = np.exp(-0.5*((strain-intrinsic_strain)/sigma)**2)
    return force_weight

@njit(cache=True)
def force_velocity_curve_tanh(strain_rate, max_strain_rate=1):
    force_weight = 1 + np.tanh(strain_rate/max_strain_rate)
    return force_weight

@njit(cache=True)
def passive_force_curve(strain, intrinsic_strain=1, coefficient=1, power=2):
    blocksize = strain.shape[0]
    force_weight = np.zeros(blocksize)
    for i in range(blocksize):
        if strain[i] >= intrinsic_strain:
            force_weight[i] = coefficient*(strain[i]-intrinsic_strain)**power
    return force_weight

# force-length curve (x) = 3.06 x^3 - 13.64 x^2 + 18.01 x - 6.44
@njit(cache=True)
def force_length_curve_poly(stretch, intrinsic_stretch=1, f_l_coefficients=np.array([-6.44, 18.01, -13.64, 3.06])):
    degree = f_l_coefficients.shape[0]

    blocksize = stretch.shape[0]
    force_weight = np.zeros(blocksize)
    for i in range(blocksize):
        for power in range(degree):
            force_weight[i] += f_l_coefficients[power] * (stretch[i] ** power)
        force_weight[i] = 0 if force_weight[i] < 0 else force_weight[i]
    return force_weight

@njit(cache=True)
def force_length_curve_hill(stretch, intrinsic_stretch=1):
    strain_rate_min = -42  # 1/sec
    c3 = 1450e3
    c4 = -625e3
    eps_c = 0.773
    c2 = c3 * eps_c / (c3 * eps_c + c4)
    c1 = c3 / (c2 * eps_c ** (c2 - 1))

    l_bz = 0.14e-6
    l_z = 0.06e-6
    # l0_sarc = l_act + l_z + 0.5*l_bz
    l0_sarc = 1.0e-6
    l_act = l0_sarc - l_z - 0.5 * l_bz

    l_myo = 0.7e-6
    D_act = 0.68
    D_myo = 1.90
    C_myo = 0.44
    l_max = l_myo + l_act + l_z
    l_min = l_bz
    f_a = 1
    k = 0.25
    E_comp = 2e4

    blocksize = stretch.shape[0]
    force_weight = np.zeros(blocksize)

    for i in range(blocksize):
        eps_r = stretch[i] - intrinsic_stretch
        l_sarc = l0_sarc + eps_r * l0_sarc
        #     print(l_sarc)
        if l_max < l_sarc:
            f_l = 0
        if l_act + l_bz + l_z <= l_sarc and l_sarc <= l_myo + l_act + l_z:
            f_l = (l_myo + l_act + l_z - l_sarc) / (l_myo - l_bz)
        if l_act + l_z <= l_sarc and l_sarc <= l_act + l_bz + l_z:
            f_l = 1
        if l_myo + l_z <= l_sarc and l_sarc <= l_act + l_z:
            f_l = (l_myo - l_bz - D_act * (l_act + l_z - l_sarc)) / (l_myo - l_bz)
        if l_min <= l_sarc and l_sarc <= l_myo + l_z:
            f_l = (
                l_myo
                - l_bz
                - D_act * (l_act + l_z - l_sarc)
                - D_myo * (l_myo + l_z - l_sarc)
            ) / (l_myo - l_bz) - C_myo * (l_myo + l_z - l_sarc) / (l_myo - l_bz)
        if l_sarc <= l_min:
            f_l = 0
        if f_l < 0:
            f_l = 0

        force_weight[i] = f_l

    return force_weight

def plot_force_length_active_curve():
    strain_range = np.linspace(0, 2, 1000)
    force_length_weight = force_length_curve_guassian(strain_range)
    passive_force_weight = passive_force_curve(strain_range)
    
    figsize = (19.2, 10.8)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(strain_range, force_length_weight, color='C1', linewidth=8, label='active')
    ax.legend(prop={"size":40}, frameon=False)
    ax.set_xlim(-0.1, 2.1)
    ax.tick_params(axis='x', labelsize=48)
    ax.tick_params(axis='y', labelsize=48)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([0, 25, 50, 75, 100])
    ax.tick_params(direction='out', length=20, width=5)
    ax = change_box_to_arrow_axes(
        fig, ax,
        linewidth=5.0,
        overhang=0.0,
        xaxis_ypos=-0.05,
        yaxis_xpos=-0.1,
        x_offset=[-0.05, 0.1],
        y_offset=[-0.05, 0.2]
    )
    return fig, ax

def plot_force_length_curve():
    strain_range = np.linspace(0.5, 2, 1000)
    force_length_weight = force_length_curve_guassian(strain_range)
    passive_force_weight = passive_force_curve(strain_range)
    
    figsize = (19.2, 10.8)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(strain_range, force_length_weight, color='C3', linewidth=8, label='active')
    ax.plot(strain_range, passive_force_weight, color='saddlebrown', linewidth=8, label='passive')
    ax.plot(strain_range, force_length_weight+passive_force_weight, color='black', linewidth=8, label='total')
    ax.legend(prop={"size":40}, frameon=False)
    ax.set_xlim(-0.1, 2.1)
    ax.tick_params(axis='x', labelsize=48)
    ax.tick_params(axis='y', labelsize=48)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([0, 25, 50, 75, 100])
    ax.tick_params(direction='out', length=20, width=5)
    ax = change_box_to_arrow_axes(
        fig, ax,
        linewidth=5.0,
        overhang=0.0,
        xaxis_ypos=-0.05,
        yaxis_xpos=-0.1,
        x_offset=[-0.05, 0.1],
        y_offset=[-0.05, 0.2]
    )
    return fig, ax

def plot_force_velocity_curve():
    strain_rate_range = np.linspace(-2, 2, 1000)
    force_velocity_weight = force_velocity_curve_tanh(strain_rate_range)
    figsize = (19.2, 10.8)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(strain_rate_range, force_velocity_weight, color='C1', linewidth=8, label='active')
    ax.legend(prop={"size":40}, frameon=False)
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(0, 2.1)
    ax.tick_params(axis='x', labelsize=48)
    ax.tick_params(axis='y', labelsize=48)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_yticklabels(['', 50, 100, 150, 200])
    ax.tick_params(direction='inout', length=20, width=5)
    ax = change_box_to_arrow_axes(
        fig, ax,
        linewidth=5.0,
        overhang=0.0,
        xaxis_ypos=0,
        yaxis_xpos=0,
        x_offset=[-0.05, 0.2],
        y_offset=[-0.05, 0.2]
    )
    ax.spines['left'].set_position(('data', 0))
    return fig, ax

def plot_hill_force_length_active_curve():
    
    f_a = 1
    f_v = 1.0
    stress_max = 280e3

    stretch_list = []
    stress_act_list = []
    for stretch in np.linspace(0.0, 2.0, 1001):
        
        f_l = force_length_curve_hill(np.array([stretch]))
        stress_act = f_a * stress_max * f_v * f_l
        stretch_list.append(stretch)
        stress_act_list.append(stress_act)
    
    stretch_array = np.array(stretch_list)
    stress_act_array = np.array(stress_act_list)
    plt.plot(stretch_array, stress_act_array/max(stress_act_array), "k.")

def generate_random_experiment_data(number_of_samples=100, deg=3):
    np.random.seed(314159265)
    sigma_stretch = 0.2
    mu_stretch = 1.0
    stretch_data = sigma_stretch * np.random.randn(number_of_samples) + mu_stretch
    # stretch_data = np.linspace(0, 2, number_of_samples)

    sigma_force_weight = 0.05
    force_weight_data = force_length_curve_hill(stretch_data)
    force_weight_data += sigma_force_weight * np.random.randn(number_of_samples)
    force_weight_data[force_weight_data < 0] = 0

    f_l_coefficients = np.polyfit(stretch_data, force_weight_data, deg=deg)
    result = "model function: "
    for i in range(deg+1):
        if i == deg:
            result += " %.2f " % f_l_coefficients[i]
        elif i == deg-1:
            result += " %.2f x +" % f_l_coefficients[i]
        else:
            result += " %.2f x^%d +" % (f_l_coefficients[i], deg-i)
    print(result)
    
    f_l = np.poly1d(f_l_coefficients) 
    stretch = np.linspace(0, 2.0, 401)
    # stretch = np.linspace(0.4, 1.9, 401)
    force_weight = f_l(stretch)
    # force_weight -= min(force_weight)
    # force_weight /= max(force_weight)
    force_weight[force_weight < 0] = 0

    figsize = (19.2, 10.8)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # ax.scatter(stretch_data, force_weight_data, s=100, color='C2', label='exp')
    ax.plot(stretch, force_weight, color='C1', linewidth=8, label='model')
    # ax.plot(stretch, force_length_curve_hill(stretch), color='C3', linewidth=8, label='Hill')
    ax.legend(prop={"size":40}, frameon=False)
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(-0.1, 1.1)
    ax.tick_params(axis='x', labelsize=48)
    ax.tick_params(axis='y', labelsize=48)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([0, 25, 50, 75, 100])
    ax.tick_params(direction='out', length=20, width=5)
    ax = change_box_to_arrow_axes(
        fig, ax,
        linewidth=5.0,
        overhang=0.0,
        xaxis_ypos=-0.1,
        yaxis_xpos=-0.1,
        x_offset=[-0.05, 0.1],
        y_offset=[-0.05, 0.2]
    )
    return fig, ax

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # plot_force_length_active_curve()
    # plot_force_length_curve()
    # plot_force_velocity_curve()
    # plot_hill_force_length_active_curve()
    generate_random_experiment_data()
    plt.show()
