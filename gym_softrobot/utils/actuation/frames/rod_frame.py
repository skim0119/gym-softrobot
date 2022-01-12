"""
Created on Sep. 23, 2021
@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

from gym_softrobot.utils.actuation.frames.frame import FrameBase
from gym_softrobot.utils.actuation.frames.frame_tools import (
    rod_color, 
    default_label_fontsize,
    process_position,
    process_director
)

class RodFrame(FrameBase):
    def __init__(self, figure_name, folder_name, fig_dict, gs_dict, **kwargs):
        FrameBase.__init__(
            self,
            figure_name=figure_name,
            folder_name=folder_name,
            fig_dict=fig_dict
        )
        self.gs_dict = gs_dict
        self.ax3d_flag = kwargs.get("ax3d_flag", False)
        self.plot_rod = (
            self.plot_rod3d if self.ax3d_flag else self.plot_rod2d
        )

        self.fontsize = kwargs.get("fontsize", default_label_fontsize)
        self.rod_color = kwargs.get("rod_color", rod_color)
        self.offset = kwargs.get("offset", np.zeros(3))
        self.rotation = kwargs.get("rotation", np.identity(3))
        self.reference_total_length = 1
        self.reference_configuration_flag = False
        self.set_n_elems(kwargs.get("n_elems", 100))

    def set_n_elems(self, n_elems):
        self.n_elems = n_elems
        self.s = np.linspace(0, 1, self.n_elems+1)
        self.s_shear = (self.s[:-1] + self.s[1:])/2
        self.s_kappa = self.s[1:-1].copy()

    def reset(self,):
        FrameBase.reset(self,)
        self.gs = gridspec.GridSpec(
            figure=self.fig,
            **self.gs_dict
        )
        self.axes_shear = []
        self.axes_kappa = []
        for i in range(3):
            self.axes_kappa.append(
                self.fig.add_subplot(self.gs[i, 5], xlim=[-0.1, 1.1])
            )
            self.axes_shear.append(
                self.fig.add_subplot(self.gs[i, 4], xlim=[-0.1, 1.1])
            )
        if self.ax3d_flag:
            self.ax_rod = self.fig.add_subplot(
                self.gs[0:3, 0:3], projection='3d'
            )
        else:
            self.ax_rod = self.fig.add_subplot(self.gs[0:3, 0:3])
        
        if self.reference_configuration_flag:
            self.plot_ref_configuration()
        
    def set_ref_configuration(self, position, shear, kappa):
        self.reference_position = position.copy()
        self.reference_shear = shear.copy()
        self.reference_kappa = kappa.copy()
        
        reference_length = np.linalg.norm(
            position[:, 1:]-position[:, :-1], axis=0
        )
        self.reference_total_length = reference_length.sum()
        self.set_n_elems(reference_length.shape[0])
        self.reference_configuration_flag = True
        return self.reference_total_length

    def plot_ref_configuration(self,):
        line_position = process_position(
            self.reference_position,
            self.offset, self.rotation
        ) / self.reference_total_length
        
        if self.ax3d_flag:
            self.ax_rod.plot(
                line_position[0], line_position[1], line_position[2],
                color="grey", linestyle="--"
            )
        else:
            self.ax_rod.plot(
                line_position[0], line_position[1],
                color="grey", linestyle="--"
            )

        for index_i in range(3):
            self.axes_shear[index_i].plot(
                self.s_shear,
                self.reference_shear[index_i],
                color="grey",
                linestyle="--"
            )
            self.axes_kappa[index_i].plot(
                self.s_kappa,
                self.reference_kappa[index_i],
                color="grey",
                linestyle="--"
            )

    def calculate_line_position(self, position, director, radius):
        line_center = process_position(
            position, self.offset, self.rotation
        ) / self.reference_total_length
        line_position = process_position(
            (position[:, :-1] + position[:, 1:])/2,
            self.offset, self.rotation
        )
        line_director = process_director(director, self.rotation)
        line_up = (
            (line_position + line_director[1, :, :] * radius) 
            / self.reference_total_length
        )
        line_down = (
            (line_position - line_director[1, :, :] * radius) 
            / self.reference_total_length
        )
        line_left = (
            (line_position + line_director[0, :, :] * radius)
            / self.reference_total_length
        )
        line_right = (
            (line_position - line_director[0, :, :] * radius)
            / self.reference_total_length
        )
        return line_center, [line_up, line_right, line_down, line_left]


    def plot_rod2d(self, position, director, radius, color=None):
        line_center, lines = self.calculate_line_position(
            position, director, radius
        )
        self.ax_rod.plot(
            lines[0][0], lines[0][1],
            color=self.rod_color if color is None else color
        )
        self.ax_rod.plot(
            lines[2][0], lines[2][1],
            color=self.rod_color if color is None else color
        )
        self.ax_rod.plot(
            lines[1][0], lines[1][1],
            alpha=0.2,
            color=self.rod_color if color is None else color
        )
        self.ax_rod.plot(
            lines[3][0], lines[3][1],
            alpha=0.2,
            color=self.rod_color if color is None else color
        )
        self.ax_rod.plot(
            [lines[0][0, -1], line_center[0, -1], lines[2][0, -1]],
            [lines[0][1, -1], line_center[1, -1], lines[2][1, -1]],
            color=self.rod_color if color is None else color,
            label='sim'
        )
        self.ax_rod.plot(
            [lines[1][0, -1], line_center[0, -1], lines[3][0, -1]],
            [lines[1][1, -1], line_center[1, -1], lines[3][1, -1]],
            alpha=0.2,
            color=self.rod_color if color is None else color
        )
        return self.ax_rod

    def plot_rod3d(self, position, director, radius, color=None):
        line_center, lines = self.calculate_line_position(
            position, director, radius
        )
        self.ax_rod.plot(
            line_center[0], line_center[1], line_center[2],
            color=self.rod_color if color is None else color,
            linestyle="--"
        )
        for line in lines:    
            self.ax_rod.plot(
                line[0], line[1], line[2],
                color=self.rod_color if color is None else color
            ) 
        self.ax_rod.plot(
            [lines[0][0, -1], line_center[0, -1], lines[2][0, -1]],
            [lines[0][1, -1], line_center[1, -1], lines[2][1, -1]],
            [lines[0][2, -1], line_center[2, -1], lines[2][2, -1]],
            color=self.rod_color if color is None else color,
            label='sim'
        )
        self.ax_rod.plot(
            [lines[1][0, -1], line_center[0, -1], lines[3][0, -1]],
            [lines[1][1, -1], line_center[1, -1], lines[3][1, -1]],
            [lines[1][2, -1], line_center[2, -1], lines[3][2, -1]],
            color=self.rod_color if color is None else color
        )
        return self.ax_rod

    def plot_strains(self, shear, kappa, color=None):
        for index_i in range(3):
            self.axes_shear[index_i].plot(
                self.s_shear,
                shear[index_i],
                color=self.rod_color if color is None else color
            )
            self.axes_kappa[index_i].plot(
                self.s_kappa,
                kappa[index_i],
                color=self.rod_color if color is None else color
            )
        return self.axes_shear, self.axes_kappa

    def set_ax_rod_lim(
        self, 
        x_lim=[-1.1, 1.1],
        y_lim=[-1.1, 1.1],
        z_lim=[-1.1, 1.1]
    ):
        self.ax_rod.set_xlim(x_lim)
        self.ax_rod.set_ylim(y_lim)
        if self.ax3d_flag:
            self.ax_rod.set_zlim(z_lim)

    def set_ax_strains_lim(self, axes_shear_lim=None, axes_kappa_lim=None,):
        if axes_shear_lim is None:
            axes_shear_lim = [
                [-0.11, 0.11],
                [-0.11, 0.11],
                [-0.1, 2.1]
            ]
        if axes_kappa_lim is None:
            axes_kappa_lim = [
                [-110, 110],
                [-110, 110],
                [-11, 11],
            ]
        for index_i in range(3):
            shear_mean = np.average(axes_shear_lim[index_i]) if index_i != 2 else 1
            shear_log = np.floor(
                np.log10(axes_shear_lim[index_i][1] - shear_mean)
            )
            kappa_mean = np.average(axes_kappa_lim[index_i])
            kappa_log = np.floor(
                np.log10(axes_kappa_lim[index_i][1] - kappa_mean)
            )
            
            self.axes_shear[index_i].set_ylim(axes_shear_lim[index_i])
            self.axes_kappa[index_i].set_ylim(axes_kappa_lim[index_i])
            self.axes_shear[index_i].ticklabel_format(
                axis='y', scilimits=(shear_log, shear_log),
                useOffset=shear_mean
            )
            self.axes_kappa[index_i].ticklabel_format(
                axis='y', scilimits=(kappa_log, kappa_log),
                useOffset=kappa_mean
            )

    def set_labels(self, time=None):
        if time is not None:
            self.ax_rod.set_title(
                "time={:.2f} [sec]".format(time), 
                fontsize=self.fontsize
            )
        
        self.ax_rod.legend()
        
        for index_i in range(3):
            self.axes_kappa[index_i].set_ylabel(
                "    d$_{}$".format(index_i+1),
                fontsize=self.fontsize,
                rotation=0
            )
            self.axes_kappa[index_i].yaxis.set_label_position("right")

        self.axes_shear[0].set_title("shear", fontsize=self.fontsize)
        ylim = self.axes_shear[2].get_ylim()
        ylim_mean = np.average(ylim)
        position = 0.9 * (ylim[1] - ylim_mean) + ylim_mean
        self.axes_shear[2].text(
            0, position, 'stretch', 
            fontsize=self.fontsize, 
            ha='left', va='top'
        )
        self.axes_shear[2].set_xlabel("$s$", fontsize=self.fontsize)
        
        self.axes_kappa[0].set_title("curvature", fontsize=self.fontsize)
        ylim = self.axes_kappa[2].get_ylim()
        ylim_mean = np.average(ylim)
        position = 0.9 * (ylim[1] - ylim_mean) + ylim_mean
        self.axes_kappa[2].text(
            1, position, 'twist',
            fontsize=self.fontsize,
            ha='right', va='top'
        )
        self.axes_kappa[2].set_xlabel("$s$", fontsize=self.fontsize)
