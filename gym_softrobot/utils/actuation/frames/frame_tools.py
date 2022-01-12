"""
Created on Sep. 23, 2021
@author: Heng-Sheng (Hanson) Chang
"""
import numpy as np
from numba import njit
import matplotlib.colors as mcolors
from numpy.core.shape_base import block

default_colors = mcolors.TABLEAU_COLORS
# default_colors['tab:blue']    # muted blue
# default_colors['tab:orange']  # safety orange
# default_colors['tab:green']   # cooked asparagus green
# default_colors['tab:red']     # brick red
# default_colors['tab:purple']  # muted purple
# default_colors['tab:brown']   # chestnut brown
# default_colors['tab:pink']    # raspberry yogurt pink
# default_colors['tab:gray']    # middle gray
# default_colors['tab:olive']   # curry yellow-green
# default_colors['tab:cyan']    # blue-teal

base_colors = mcolors.BASE_COLORS
# base_colors['b']              # blue
# base_colors['g']              # green
# base_colors['r']              # red
# base_colors['c']              # cyan
# base_colors['m']              # magenta
# base_colors['y']              # yellow
# base_colors['k']              # black
# base_colors['w']              # white
    
default_label_fontsize = 15
paper_label_fontsize = 48
paper_linewidth = 5

rod_color = base_colors['m']
algo_color = base_colors['g']

@njit(cache=True)
def process_position(position, offset, rotation):
    blocksize = position.shape[1]
    output_position = np.zeros((3, blocksize))
    for n in range(blocksize):
        for i in range(3):
            for j in range(3):
                output_position[i, n] += (
                    rotation[i, j] * (position[j, n] - offset[j])
                )
    return output_position

@njit(cache=True)
def process_director(director, rotation):
    blocksize = director.shape[2]
    output_director = np.zeros((3, 3, blocksize))
    for n in range(blocksize):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    output_director[i, j, n] += (
                        rotation[i, k] * director[k, j, n]
                    )
    return output_director

def change_box_to_arrow_axes(
    fig, ax, linewidth=1.0, overhang=0.0,
    xaxis_ypos=0, yaxis_xpos=0,
    x_offset=[0, 0], y_offset=[0, 0]
):
    for spine in ax.spines.values():
        spine.set_visible(False)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # get width and height of axes object to compute
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1./40.*(ymax-ymin)
    hl = 1./40.*(xmax-xmin)
    lw = linewidth # axis line width
    ohg = overhang # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
    ax.arrow(xmin+x_offset[0], xaxis_ypos, xmax-xmin+x_offset[1], 0, fc='k', ec='k', lw = lw, 
                head_width=hw, head_length=hl, overhang = ohg, 
                length_includes_head= True, clip_on = False) 

    ax.arrow(yaxis_xpos, ymin+y_offset[0], 0, ymax-ymin+y_offset[1], fc='k', ec='k', lw = lw, 
                head_width=yhw, head_length=yhl, overhang = ohg, 
                length_includes_head= True, clip_on = False)
    return ax
