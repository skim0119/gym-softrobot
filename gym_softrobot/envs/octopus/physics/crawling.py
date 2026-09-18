from __future__ import annotations

import elastica as ea
import numpy as np
from numba import njit


class SegmentExtensionActuation(ea.NoForces):
    def __init__(
        self,
        start_index: int,
        end_index: int,
        original_shear_matrix: np.ndarray,
        original_bend_matrix: np.ndarray,
        amplitude: callable,
    ) -> None:
        super().__init__()
        self.start_index = int(start_index)
        self.end_index = int(end_index)
        self.original_shear_matrix = original_shear_matrix[..., :end_index]
        self.original_bend_matrix = original_bend_matrix[..., :end_index]
        self.amplitude = amplitude

    def apply_forces(self, system: ea.CosseratRod, time: float = 0.0) -> None:
        stretch_magnitude, stiffness_magnitude, bend_magnitude = self.amplitude()
        self._apply_segment_extension_force(
            self.start_index,
            self.end_index,
            self.original_shear_matrix,
            self.original_bend_matrix,
            system.rest_sigma,
            system.rest_kappa,
            system.shear_matrix,
            system.bend_matrix,
            stretch_magnitude,
            np.pi * bend_magnitude / system.rest_voronoi_lengths[: self.end_index].sum(),
            (1.0 + stiffness_magnitude),
        )

    @staticmethod
    @njit(cache=True)
    def _apply_segment_extension_force(
        start_index: int,
        end_index: int,
        original_shear_matrix: np.ndarray,
        original_bend_matrix: np.ndarray,
        rest_sigma: np.ndarray,
        rest_kappa: np.ndarray,
        shear_matrix: np.ndarray,
        bend_matrix: np.ndarray,
        stretch_scale: float,
        bend_scale: float,
        stiffness_scale: float,
    ) -> None:
        rest_sigma[2, :end_index] = stretch_scale
        rest_kappa[0, :end_index] = bend_scale
        shear_matrix[0, 0, :end_index] = original_shear_matrix[0, 0, :] * stiffness_scale
        shear_matrix[1, 1, :end_index] = original_shear_matrix[1, 1, :] * stiffness_scale
        bend_matrix[0, 0, :end_index] = original_bend_matrix[0, 0, :] * stiffness_scale
        bend_matrix[1, 1, :end_index] = original_bend_matrix[1, 1, :] * stiffness_scale

