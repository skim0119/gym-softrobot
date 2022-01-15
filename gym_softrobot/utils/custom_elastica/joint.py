from elastica import *
from elastica.joint import FreeJoint

from numba import njit

@njit(cache=True)
def z_rotation(vector, theta):
    theta = theta / 180.0 * np.pi
    R = np.array([[np.cos(theta), -np.sin(theta),0.0],[np.sin(theta), np.cos(theta),0],[0.0,0.0,1.0]])
    return np.dot(R,vector.T).T

class FixedJoint2Rigid(FreeJoint):
    """
    The fixed joint class restricts the relative movement and rotation
    between two nodes and elements by applying restoring forces and torques.
    For implementation details, refer to Zhang et al. Nature
    Communications (2019).

        Attributes
        ----------
        k: float
            Stiffness coefficient of the joint.
        nu: float
            Damping coefficient of the joint.
        kt: float
            Rotational stiffness coefficient of the joint.
    """

    def __init__(self, k, nu, kt, angle, radius):
        """

        Parameters
        ----------
        k: float
            Stiffness coefficient of the joint.
        nu: float
            Damping coefficient of the joint.
        kt: float
            Rotational stiffness coefficient of the joint.
        """
        super().__init__(k, nu)
        # additional in-plane constraint through restoring torque
        # stiffness of the restoring constraint -- tuned empirically
        self.kt = kt
        self.angle=angle
        self.radius=radius

    # Apply force is same as free joint
    def apply_forces(self, rod_one, index_one, rod_two, index_two):
        self.rigid_rod_pos = rod_one.position_collection[..., index_one].copy()
        self.rigid_rod_pos[2] = 0.0
        self.rigid_rod_connection_dir = self._apply_forces(
                index_one, index_two,
                self.rigid_rod_pos,
                rod_two.position_collection,
                rod_one.velocity_collection,
                rod_two.velocity_collection,
                rod_one.director_collection[1,...].T,
                rod_one.external_forces,
                rod_two.external_forces,
                self.k, self.nu, self.kt, self.angle, self.radius)

    @staticmethod
    @njit(cache=True)
    def _apply_forces(index_one, index_two, rigid_rod_pos,
            rod_two_position,
            rod_one_velocity,
            rod_two_velocity,
            rod_one_binormal,
            rod_one_external_forces,
            rod_two_external_forces,
            k, nu, kt, angle, radius):
        # return super().apply_forces(rod_one, index_one, rod_two, index_two)
        rigid_rod_connection_dir=-z_rotation(rod_one_binormal,angle)
        #rigid_rod_connection_dir=-z_rotation(rod_one.binormal.T,angle)

        rigid_rod_connection_pt=rigid_rod_connection_dir*radius
        rigid_rod_pos+=rigid_rod_connection_pt[0]
        end_distance_vector = (
                rod_two_position[..., index_two] - rigid_rod_pos
        )
        # Calculate norm of end_distance_vector
        # this implementation timed: 2.48 µs ± 126 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        end_distance = np.sqrt(np.dot(end_distance_vector, end_distance_vector))

        # Below if check is not efficient find something else
        # We are checking if end of rod1 and start of rod2 are at the same point in space
        # If they are at the same point in space, it is a zero vector.
        if end_distance <= np.finfo(np.float64).eps * 1e4:
            normalized_end_distance_vector = np.array([0.0, 0.0, 0.0])
        else:
            normalized_end_distance_vector = end_distance_vector / end_distance

        elastic_force = k * end_distance_vector

        relative_velocity = (
                rod_two_velocity[..., index_two]
                - rod_one_velocity[..., index_one]
        )
        normal_relative_velocity = (
                np.dot(relative_velocity, normalized_end_distance_vector)
                * normalized_end_distance_vector
        )
        damping_force = -nu * normal_relative_velocity

        contact_force = elastic_force + damping_force

        rod_one_external_forces[..., index_one] += contact_force
        rod_two_external_forces[..., index_two] -= contact_force

        return rigid_rod_connection_dir

    def apply_torques(self, rod_one, index_one, rod_two, index_two):
        #self._apply_hard_director_boundary(
        #        index_one,
        #        index_two,
        #        rod_one.director_collection,
        #        rod_two.director_collection,
        #    )
        self._apply_torques(
                index_one, index_two,
                self.rigid_rod_pos,
                self.rigid_rod_connection_dir,
                rod_two.position_collection,
                rod_one.director_collection,
                rod_two.director_collection,
                rod_two.rest_lengths,
                rod_one.external_torques,
                rod_two.external_torques,
                self.k, self.nu, self.kt, self.angle, self.radius)

    @staticmethod
    @njit(cache=True)
    def _apply_hard_director_boundary(
            index_one,
            index_two,
            rod_one_director_collection,
            rod_two_director_collection,
        ):
        pass

    @staticmethod
    @njit(cache=True)
    def _apply_torques(
            index_one, index_two,
            rigid_rod_pos,
            rigid_rod_connection_dir,
            rod_two_position,
            rod_one_director_collection,
            rod_two_director_collection,
            rod_two_rest_lengths,
            rod_one_external_torques,
            rod_two_external_torques,
            k, nu, kt, angle, radius):
        # current direction of the first element of link two
        # also NOTE: - rod two is fixed at first element
        link_direction = (
            rod_two_position[..., index_two + 1]
            - rod_two_position[..., index_two]
        )

        # To constrain the orientation of link two, the second node of link two should align with
        # the direction of link one. Thus, we compute the desired position of the second node of link two
        # as check1, and the current position of the second node of link two as check2. Check1 and check2
        # should overlap.

        # tgt_destination = (
        #     rod_one.position_collection[..., index_one]
        #     + rod_two.rest_lengths[index_two] * rod_one.tangents[..., index_one]
        # )  # dl of rod 2 can be different than rod 1 so use rest length of rod 2

        tgt_destination = (
                rigid_rod_pos
                + rod_two_rest_lengths[index_two] * rigid_rod_connection_dir
        )

        curr_destination = rod_two_position[
            ..., index_two + 1
        ]  # second element of rod2

        # Compute the restoring torque
        forcedirection = -kt * (
            curr_destination - tgt_destination
        )  # force direction is between rod2 2nd element and rod1
        torque = np.cross(link_direction, forcedirection)[0]

        # The opposite torque will be applied on link one
        for i in range(3):
            for j in range(3):
                rod_one_external_torques[i, index_one] -= rod_one_director_collection[i,j,index_one] * torque[j]
                rod_two_external_torques[i, index_two] += rod_two_director_collection[i,j,index_two] * torque[j]
        #rod_one_external_torques[..., index_one] -= (
        #    rod_one_director_collection[..., index_one] @ torque
        #)
        #rod_two_external_torques[..., index_two] += (
        #    rod_two_director_collection[..., index_two] @ torque
        #)
