__all__ = ["intersection"]
__doc__ = """
This module finds intersection of two curvature in 2D

Code originated from Sukhbinder's intersect.py.
The code is modified to support numba
"""

import numpy as np
from numba import njit

@njit(cache=True)
def _rect_inter_inner(x1, x2):
    n1 = x1.shape[0]-1
    n2 = x2.shape[0]-1
    x1min = np.empty((n1,n2))
    x1max = np.empty((n1,n2))
    x2min = np.empty((n1,n2))
    x2max = np.empty((n1,n2))
    for i in range(n1):
        x1min[i,:] = np.minimum(x1[i], x1[i+1])
        x1max[i,:] = np.maximum(x1[i], x1[i+1])
    for i in range(n2):
        x2min[:,i] = np.minimum(x2[i], x2[i+1])
        x2max[:,i] = np.maximum(x2[i], x2[i+1])
    
    return x1min, x2max, x1max, x2min

@njit(cache=True)
def intersection(position_collection1, position_collection2):
    """
    Computes 2D-coordinate of where two position_collection intersect.
    The position_collection is expected to be from PyElastica rod.
    """
    x1 = position_collection1[0,:]
    y1 = position_collection1[1,:]
    x2 = position_collection2[0,:]
    y2 = position_collection2[1,:]

    S1, S2, S3, S4 = _rect_inter_inner(x1, x2)
    S5, S6, S7, S8 = _rect_inter_inner(y1, y2)

    C1 = np.less_equal(S1, S2)
    C2 = np.greater_equal(S3, S4)
    C3 = np.less_equal(S5, S6)
    C4 = np.greater_equal(S7, S8)

    ii, jj = np.nonzero(C1 & C2 & C3 & C4)
    n = len(ii)

    AA = np.zeros((4, 4, n))
    BB = np.zeros((4, n))
    dxy1 = position_collection1[:,1:] - position_collection1[:,:-1]
    dxy2 = position_collection2[:,1:] - position_collection2[:,:-1]

    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[:,ii]
    AA[1::2, 1, :] = dxy2[:,jj]

    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    T = np.zeros((4, n))
    for i in range(n):
        #try:
        T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        #except ValueError: # Cutoff
        #    T[:, i] = np.Inf

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (
        T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range].T
    return xy0[:, 0], xy0[:, 1]
