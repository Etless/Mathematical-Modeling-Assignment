import numpy as np
import orbit_lib as ol
import simutils as su
import math

satellite_distance = ol.R_E + 400 # From earth center [km]
T = 2 * math.pi * math.sqrt(satellite_distance ** 3 / ol.mu) # Orbital period [s]

###################################
# Assignment 5 | Algorithms       #
###################################

class RigidBody:
    def __init__(self, r0: np.ndarray, v0: np.ndarray, m: float, q0: su.Quaternion, w0: np.ndarray, J: np.ndarray) -> None:
        """
        Rigid body class used for attitude simulation.
        :param r0: Initial position vector [km]
        :param v0: Initial velocity vector [km/s]
        :param m: The satellites mass [kg]
        :param q0: Initial quaternion (normalized)
        :param q0: Initial quaternion (normalized)
        :param w0: Initial angular velocity [rad/s]
        :param J: 3x3 Inertia matrix
        """
        if J.shape != (3, 3):
            raise ValueError("J must be a 3x3 matrix")

        #self.ri = r0             # Initial position [array]
        #self.vi = v0             # Initial velocity [array]
        self.force = np.zeros(3) # Force acting on body [array]

        self.x = np.concatenate([r0, v0])

        self.m = m               # Mass of body

        self.q = q0.normalized() # Initial quaternion [array] (normalized creates new class)
        self.w = w0              # Initial angular velocity [array]
        self.tau = np.zeros(3)   # Torque acting on body [array]

        # Maps torque to angular acceleration
        self.J = J  # Inertia matrix [array]
        self.J_inv = np.linalg.inv(self.J) # Inverse (precalculated)

    def update(self, t: float, dt: float, force: np.ndarray,tau: np.ndarray) -> None:
        """
        Update the state of the rigid body.
        :param t: Time [s]
        :param dt: Time step [s]
        :param force: Force vector [N*m]
        :param tau: Torque vector acting on body [N*m]
        """
        self.force = force
        ae = force / self.m # External acceleration
        #self.x = su.step_RK4(dt, t, self.x, su.two_body, ae=ae)
        self.x = su.step_RK4(dt, t, self.x, su.two_body)

        self.tau = tau # Update torque

        # Create state vector (needed for step)
        x = np.concatenate((self.q[:], self.w))
        #x = su.step_RK4(dt, t, x, self.f, tau) # Perform step
        x = su.step_RK4(dt, t, x, self.f)  # Perform step

        self.q[:] = x[:4] # Update values in Quaternion class
        self.w    = x[4:] # Update angular velocity

        self.q.normalize() # Normalize values in class

    def get_state(self):
        return self.x[:3], self.x[3:], self.q, self.w

    # noinspection PyUnusedLocal
    def f(self, t: float, x: np.ndarray, tau: np.ndarray | None = None)  -> np.ndarray:
        """
        Compute the time derivative of the state vector.

        The state vector x contains both the quaternion and angular velocity,
        formatted as: [q0, q1, q2, q3, w0, w1, w2].

        :param t: Time [s]
        :param x: State vector containing quaternion and angular velocity
        :param tau: Torque vector acting on body [N*m]
        :return: Time derivative of the state vector
        """
        if tau is None: # If torque is not applied
            tau = np.zeros(3)

        # Temp is used to differentiate it from the class self counterparts
        q_temp = x[:4]
        w_temp = x[4:]

        dq = 0.5 * (su.Quaternion(q_temp) @ su.Quaternion(w_temp))
        dw = self.J_inv @ (tau - np.cross(w_temp, self.J @ w_temp))

        return np.concatenate((dq, dw))


class Satellite:
    def __init__(self, q_ib, w_bib, J, ri=np.zeros(3), vi=np.zeros(3), m=1, orbit=None, substeps=0) -> None:
        self.orbit = orbit

        if self.orbit is not None:
            ri, vi = self.orbit.get_state()

        self.body = RigidBody(ri, vi, m, q_ib, w_bib, J)
        self.N = substeps + 1
        self.ADCS = ADCS_PD(1e-5, 2e-4, J)

    def update(self, t: float, dt: float) -> None:
        """
        Update the state of the satellite.
        :param t: Time [s]
        :param dt: Time step [s]
        """
        if self.orbit:
            self.update_with_orbit(t, dt)
        else:
            self.update_with_dynamics(t, dt)

    def get_state(self):
        return self.body.get_state()

    def get_orbit_frame(self):
        if self.orbit:
            return self.orbit.get_orbit_frame()
        else:
            ri, vi, _, _ = self.body.get_state()
            return ol.orbit_frame_from_state(ri, vi)

    def update_with_orbit(self, t: float, dt: float) -> None:
        r0, v0 = self.orbit.get_state()
        self.orbit.propagate(dt)
        r1, v1 = self.orbit.get_state()
        t_sub = dt / self.N
        for n in range(0, self.N):
            ri = r0 + n / self.N * (r1 - r0)
            vi = v0 + n / self.N * (v1 - v0)
            _, _, q_ib, w_bib = self.get_state()
            q_io, w_iio, _ = ol.orbit_frame_from_state(ri, vi)
            self.ADCS.update(t, q_ib, w_bib, q_io, w_iio, np.zeros(3))
            tau_u = self.ADCS.get_control()
            self.body.update(t, dt, np.zeros(3), tau_u)
            t += t_sub
        self.body.ri, self.body.vi = self.orbit.get_state()

    def update_with_dynamics(self, t, dt):
        t_sub = dt / self.N
        for n in range(0, self.N):
            ri, vi, q_ib, w_bib = self.get_state()
            q_io, w_iio, dw_iio = self.get_orbit_frame()
            self.ADCS.update(t, q_ib, w_bib, q_io, w_iio, dw_iio)
            tau_u = self.ADCS.get_control()
            f = -ol.mu / np.linalg.norm(ri) ** 3 * ri
            self.body.update(t, t_sub, f, tau_u)
            t += t_sub


###################################
# Assignment 5 | Algorithms       #
###################################

class ADCS_PD:
    def __init__(self, k1, k2, J):
        self.k1 = k1
        self.k2 = k2
        self.J = J
        self.tau = np.zeros(3)

    def update(self, t, q_ib, w_bib, q_io, w_iio, dw_iio):
        # Quaternion error (desired -> body)
        q_db = q_io.conjugated() @ q_ib
        if q_db[0] < 0:  # Shortest way/direction to rotate
            q_db *= -1

        # Angular velocity error (desired -> body)
        w_db = w_bib - q_db.conjugated().rotate(w_iio)

        # Simple PD controller for torque
        self.tau = -self.k1 * q_db[1:] - self.k2 * w_db

    def get_control(self):
        return self.tau

