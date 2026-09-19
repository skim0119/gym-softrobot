"""Distance and settling reward for the tendon-driven arm reaching task."""

from __future__ import annotations


def tendon_arm_reward(
    *,
    distance: float,
    tip_speed: float = 0.0,
    failure: bool = False,
    velocity_weight: float = 1.0e-3,
    velocity_reference: float = 0.1,
    velocity_gate: float = 0.03,
) -> tuple[float, dict[str, float]]:
    """Reward target proximity and low speed near the target.

    The speed penalty tapers linearly to zero outside ``velocity_gate`` so it
    encourages settling without discouraging motion across the workspace.
    ``velocity_weight`` is in meters and ``velocity_reference`` is in m/s.
    """
    distance_value = -abs(float(distance))
    gate_weight = max(0.0, 1.0 - abs(float(distance)) / velocity_gate)
    velocity_value = -velocity_weight * gate_weight * (
        float(tip_speed) / velocity_reference
    ) ** 2
    failure_value = -50.0 if failure else 0.0
    components = {
        "distance": distance_value,
        "velocity": velocity_value,
        "failure": failure_value,
    }
    return float(sum(components.values())), components
