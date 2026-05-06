import numpy as np
import orbit_lib as ol
import simutils as su
import math

from simutils import Quaternion

satellite_distance = ol.R_E + 400 # From earth center [km]
T = 2 * math.pi * math.sqrt(satellite_distance ** 3 / ol.mu) # Orbital period [s]

###################################
# Assignment 4 | Algorithms       #
###################################

class RigidBody:
    def __init__(self, q0: su.Quaternion, w0: np.ndarray, J: np.ndarray):
        """
        Rigid body class used for attitude simulation.
        :param q0: Initial quaternion (normalized)
        :param w0: Initial angular velocity [rad/s]
        :param J: 3x3 Inertia matrix
        """
        if J.shape != (3, 3):
            raise ValueError("J must be a 3x3 matrix")

        self.q = q0.normalized() # Initial quaternion [array] (normalized creates new class)
        self.w = w0              # Initial angular velocity [array]
        self.tau = np.zeros(3)   # Torque acting on body [array]

        # Maps torque to angular acceleration
        self.J = J  # Inertia matrix [array]
        self.J_inv = np.linalg.inv(self.J) # Inverse (precalculated)

    def update(self, t, dt, tau):
        self.tau = tau # Update torque

        # Create state vector (needed for step)
        x = np.concatenate((self.q[:], self.w))
        x = su.step_RK4(dt, t, x, self.f, tau) # Perform step

        self.q[:] = x[:4] # Update values in Quaternion class
        self.w    = x[4:] # Update angular velocity

        self.q.normalize() # Normalize values in class

    def f(self, t, x, tau: np.ndarray = None)  -> np.ndarray:
        if tau is None:
            tau = np.zeros(3)

        # Temp is used to differentiate it from the class self counterparts
        q_temp = x[:4]
        w_temp = x[4:]

        dq = 0.5 * su.Quaternion(q_temp) @ su.Quaternion(w_temp)
        dw = self.J_inv @ (tau - np.cross(w_temp, self.J @ w_temp))

        return np.concat((dq, dw))


class Satellite:
    def __init__(self):
        pass
    def update(self):
        pass


