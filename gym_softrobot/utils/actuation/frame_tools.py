"""
Created on Mar. 11, 2021
@author: Heng-Sheng (Hanson) Chang
"""

import matplotlib.colors as mcolors

default_colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
    ]
    
default_label_fontsize = 12
paper_label_fontsize = 48
paper_linewidth = 5

rod_color = mcolors.BASE_COLORS['m']
algo_color = mcolors.BASE_COLORS['g']

def change_box_to_arrow_axes(fig, ax, linewidth=1.0, overhang=0.0, xaxis_ypos=0, yaxis_xpos=0, x_offset=[0, 0], y_offset=[0, 0]):
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