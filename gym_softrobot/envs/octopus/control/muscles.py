from __future__ import annotations

import numpy as np
from numba import njit

import elastica as ea
from coomm.actuations.batch_muscle import BatchMuscle

LM_RATIO_MUSCLE_POSITION = 1.0
OM_RATIO_MUSCLE_POSITION = 1.0

AN_RATIO_RADIUS = 0.002
TM_RATIO_RADIUS = 0.045
LM_RATIO_RADIUS = 0.100
OM_RATIO_RADIUS = 0.0375

OM_ROTATION_NUMBER = 6
TM_MAX_MUSCLE_STRESS = 15_000.0 * 2
LM_MAX_MUSCLE_STRESS = 10_000.0 * 200
OM_MAX_MUSCLE_STRESS = 100_000.0 * 100

TM_GROUP_COUNT = 1
LM_GROUP_COUNT = 4
OM_GROUP_COUNT = 2

MUSCLE_GROUP_NAMES: tuple[str, ...] = (
    "TM",
    "LM0",
    "LM1",
    "LM2",
    "LM3",
    "OM+",
    "OM-",
)


def create_octopus_muscle_groups(rod: ea.CosseratRod, activation_array) -> BatchMuscle:
    base_radius = rod.radius[0]

    an_ratio_radius = AN_RATIO_RADIUS / base_radius
    tm_ratio_radius = TM_RATIO_RADIUS / base_radius

    rod_area = np.pi * rod.radius**2
    tm_rest_muscle_area = rod_area * (tm_ratio_radius**2 - an_ratio_radius**2)
    lm_rest_muscle_area = rod_area * (LM_RATIO_RADIUS**2)
    om_rest_muscle_area = rod_area * (OM_RATIO_RADIUS**2)

    muscles = (
        BatchMuscle(num_elements=rod.n_elems)
        .configure(force_length_weight=force_length_weight_poly)
        .add_transverse_muscle(
            rest_muscle_area=tm_rest_muscle_area,
            max_muscle_stress=TM_MAX_MUSCLE_STRESS,
            activation=activation_array[:1],
        )
    )
    for index in range(LM_GROUP_COUNT):
        muscles.add_longitudinal_muscle(
            muscle_init_angle=np.pi * 0.5 * index,
            ratio_muscle_position=LM_RATIO_MUSCLE_POSITION,
            rest_muscle_area=lm_rest_muscle_area,
            max_muscle_stress=LM_MAX_MUSCLE_STRESS,
            activation=activation_array[TM_GROUP_COUNT + index : TM_GROUP_COUNT + index + 1],
        )
    for index in range(OM_GROUP_COUNT):
        muscles.add_oblique_muscle(
            muscle_init_angle=np.pi * 0.5 * index,
            ratio_muscle_position=OM_RATIO_MUSCLE_POSITION,
            rotation_number=OM_ROTATION_NUMBER,
            rest_muscle_area=om_rest_muscle_area,
            max_muscle_stress=OM_MAX_MUSCLE_STRESS,
            activation=activation_array[
                TM_GROUP_COUNT + LM_GROUP_COUNT : TM_GROUP_COUNT + LM_GROUP_COUNT + 1
            ],
        )
    for index in range(OM_GROUP_COUNT):
        muscles.add_oblique_muscle(
            muscle_init_angle=np.pi * 0.5 * index,
            ratio_muscle_position=OM_RATIO_MUSCLE_POSITION,
            rotation_number=-OM_ROTATION_NUMBER,
            rest_muscle_area=om_rest_muscle_area,
            max_muscle_stress=OM_MAX_MUSCLE_STRESS,
            activation=activation_array[
                TM_GROUP_COUNT + LM_GROUP_COUNT + 1 : TM_GROUP_COUNT + LM_GROUP_COUNT + 2
            ],
        )
    muscles.blocking(rod)
    return muscles


@njit(cache=True)
def force_length_weight_poly(muscle_length: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, -5.0 * (muscle_length - 1.0) ** 2 + 1.0)
