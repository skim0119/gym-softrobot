"""Phase-driven octopus simulation with independently controlled muscle groups."""

from __future__ import annotations

from gym_softrobot.envs.octopus.control.muscle_policy import OctoArmMusclePolicy
from gym_softrobot.envs.octopus.control.muscles import create_octopus_muscle_groups
from gym_softrobot.envs.octopus.crawling_simulation.phase_simulation import (
    PhaseOctopusSimulation,
)
from gym_softrobot.envs.octopus.physics.forcing import SuckerActuation

from coomm.actuations.batch_muscle import ApplyMuscleActuations


ACTUATION_CHANNEL_NAMES: tuple[str, ...] = (
    "tm",
    "lm0",
    "lm1",
    "lm2",
    "lm3",
    "base_suction",
    "middle_suction",
    "om_positive",
    "om_negative",
)


class PhaseOctopusMuscleSimulation(PhaseOctopusSimulation):
    """One or more octopuses with independent TM/LM/OM muscle-group controls per arm."""

    PolicyCls: type = OctoArmMusclePolicy

    def _add_actuators(
        self,
        simulator,
        rod,
        cfg,
        instance_index: int,
        arm_index: int,
        plane_origin,
        plane_normal,
        policy,
    ) -> None:
        simulator.add_forcing_to(rod).using(
            SuckerActuation,
            k=cfg.sucker_k,
            nu=cfg.sucker_nu,
            k_c=0.0,  # cfg.sucker_contact_k,  # Friction plane already exist
            nu_c=0.0,  # cfg.sucker_contact_nu,
            forces_enabled=cfg.arm_sucker_active(arm_index),
            trigger=lambda inst=instance_index, arm=arm_index: (
                self.instances[inst].policy.middle_suction_active[arm],
                self.instances[inst].policy.base_suction_active[arm],
            ),
            plane_origin=plane_origin,
            plane_normal=plane_normal,
            start_index=[self.middle_range[0], self.base_range[0]],
            end_index=[self.middle_range[1], self.base_range[1]],
        )

        batch_muscles = create_octopus_muscle_groups(
            rod, policy.target_muscle_groups[arm_index]
        )
        simulator.add_forcing_to(rod).using(
            ApplyMuscleActuations,
            batch_muscles,
        )
