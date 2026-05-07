import numpy as np
import orbit_lib as ol
import simutils as su
import math

satellite_distance = ol.R_E + 400 # From earth center [km]
T = 2 * math.pi * math.sqrt(satellite_distance ** 3 / ol.mu) # Orbital period [s]

###################################
# Assignment 4 | Algorithms       #
###################################

class RigidBody:
    def __init__(self, q0: su.Quaternion, w0: np.ndarray, J: np.ndarray) -> None:
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

    def update(self, t: float, dt: float, tau: np.ndarray) -> None:
        """
        Update the state of the rigid body.
        :param t: Time [s]
        :param dt: Time step [s]
        :param tau: Torque vector acting on body [N*m]
        """
        self.tau = tau # Update torque

        # Create state vector (needed for step)
        x = np.concatenate((self.q[:], self.w))
        x = su.step_RK4(dt, t, x, self.f, tau) # Perform step

        self.q[:] = x[:4] # Update values in Quaternion class
        self.w    = x[4:] # Update angular velocity

        self.q.normalize() # Normalize values in class

    def f(self, t: float, x: np.ndarray, tau: np.ndarray | None = None)  -> np.ndarray:
        """
        Compute the time derivative of the state vector.

        The state vector x contains both the quaternion and angular velocity,
        formatted as: [q0, q1, q2, q3, w0, w1, w2].

        :param t: Time [s]
        :param x: State vector containing quaternion and angular velocity
        :param tau: Torque vector acting on body [N*m]
        :return: State vector with the time derivative of x [rad/s | rad/s**2]
        """
        if tau is None: # If torque is not applied
            tau = np.zeros(3)

        # Temp is used to differentiate it from the class self counterparts
        q_temp = x[:4]
        w_temp = x[4:]

        dq = 0.5 * su.Quaternion(q_temp) @ su.Quaternion(w_temp)
        dw = self.J_inv @ (tau - np.cross(w_temp, self.J @ w_temp))

        return np.concatenate((dq, dw))


class Satellite:
    def __init__(self, q0: su.Quaternion, w0: np.ndarray, J: np.ndarray, qd: su.Quaternion, wd: np.ndarray, k1: float = 2.0, k2: float = 1.0) -> None:
        self.body = RigidBody(q0, w0, J)
        self.ri = np.zeros(3)

        # Coefficients for PD controller
        self.k1 = k1
        self.k2 = k2

        # Desired state
        self.qd = qd.normalized()
        self.wd = wd
        print(qd, wd)

    def update(self, t, dt) -> None:
        q = self.body.q
        w = self.body.w

        q_db = self.qd.conjugated() @ q
        if q_db[0] < 0:
            q_db *= -1

        w_db = w - q_db.conjugated().rotate(self.wd)

        tau = -self.k1 * q_db[1:] - self.k2 * w_db
        print(q)
        self.body.update(t, dt, tau)

    def get_state(self) -> tuple[np.ndarray, su.Quaternion]:
        return self.ri, self.body.q


