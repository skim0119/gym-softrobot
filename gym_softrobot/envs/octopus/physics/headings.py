from __future__ import annotations

import numpy as np

FORWARD_HEADING_XZ = np.array([0.0, -1.0], dtype=np.float64)
HALF_STEP_HEADING_ANGLE = np.pi / 8.0
HEADING_POS_22_5_XZ = np.array(
    [np.sin(HALF_STEP_HEADING_ANGLE), -np.cos(HALF_STEP_HEADING_ANGLE)],
    dtype=np.float64,
)
HEADING_NEG_22_5_XZ = np.array(
    [np.sin(-HALF_STEP_HEADING_ANGLE), -np.cos(-HALF_STEP_HEADING_ANGLE)],
    dtype=np.float64,
)

